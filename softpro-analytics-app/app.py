# Softpro Sentiment & Sales Insights – Streamlit Prototype
# -------------------------------------------------------
# Postmortem Shala Style: End-to-end app to
# 1) Upload audio + CSV
# 2) Transcribe audio (Whisper or Vosk)
# 3) Merge with CRM logs
# 4) Sentiment analysis (pretrained OR train custom if labels exist)
# 5) Analytics + Recommendations in Streamlit
#
# How to run (first time):
# -------------------------------------------------------
# 1) Install system deps: ffmpeg (for Whisper). On Ubuntu: sudo apt-get update && sudo apt-get install -y ffmpeg
# 2) Python deps:
#    pip install streamlit pandas numpy scikit-learn transformers torch plotly python-dateutil openai-whisper vosk wordcloud
#    (torch may need a platform-specific install; see https://pytorch.org/ if necessary)
# 3) Run app:
#    streamlit run app.py
#
# Notes:
# - Whisper downloads models automatically on first run. Choose tiny/base/small/medium for speed vs accuracy.
# - Vosk is fully offline but needs a model; download a model (e.g., small-en) and provide its folder path in the sidebar.
# - Expected CSV columns (flexible mapping provided in UI):
#   student_name, year, tech_stack, location (Noida/Lucknow), remarks, call_id, date (YYYY-MM-DD or any parseable date)
#   Optional: label (positive/neutral/negative) for training a custom model.
# - If label column is present, we train TF-IDF + LogisticRegression and use it; otherwise we fallback to a Hugging Face pretrained pipeline.

import os
import io
import tempfile
from datetime import datetime
from dateutil import parser

import numpy as np
import pandas as pd

import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


#D:\Python-with-Datascience\softpro-Analytics\ffmpeg\bin\

#set the Enviroment Variable :-

#D:\Python-with-Datascience\softpro-Analytics\ffmpeg\bin\
os.environ["PATH"] += os.pathsep + r"D:\Python-with-Datascience\softpro-Analytics\ffmpeg\bin"


# Optional imports are wrapped – they may fail gracefully if not installed
# Exceptional imports: Open Source model
try:
    import whisper  # OpenAI Whisper
except Exception:
    whisper = None

try:
    from vosk import Model as VoskModel, KaldiRecognizer
    import wave
except Exception:
    VoskModel = None
    KaldiRecognizer = None
# NLP Pipelines
from transformers import pipeline

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="Softpro Sentiment & Sales Insights", layout="wide")
st.title("Softpro Sentiment & Sales Insights")
st.caption("Audio + CRM logs → Transcripts → Sentiment → Insights → Recommendations")

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("Settings")
asr_engine = st.sidebar.selectbox("ASR Engine (Audio → Text)", ["Whisper", "Vosk (offline)"])
if asr_engine == "Whisper":
    whisper_size = st.sidebar.selectbox("Whisper model size", ["tiny", "base", "small", "medium"], index=1)
else:
    vosk_model_dir = st.sidebar.text_input("Vosk model directory (unzipped)", value="")

# st.sidebar.markdown("---")
# # use_pretrained = st.sidebar.checkbox("Force Pretrained Sentiment (skip training even if labels exist)", value=False)
# 
st.sidebar.markdown("---")
st.sidebar.write("**Export**")
save_intermediate = st.sidebar.checkbox("Save processed CSV", value=True)

# -----------------------------
# Utilities
# -----------------------------

# Whisper loading model
@st.cache_resource(show_spinner=False)
def load_whisper(model_size: str):
    if whisper is None:
        raise RuntimeError("Whisper not installed. pip install openai-whisper and ensure ffmpeg is present.")
    return whisper.load_model(model_size)

# Vosk loading model
@st.cache_resource(show_spinner=False)
def load_vosk(model_dir: str):
    if not model_dir or not os.path.isdir(model_dir):
        raise RuntimeError("Valid Vosk model directory not provided.")
    if VoskModel is None:
        raise RuntimeError("Vosk not installed. pip install vosk")
    return VoskModel(model_dir)

# High frequency pipelines to be loaded

@st.cache_resource(show_spinner=False)
def load_hf_pipeline():
    # Fast, widely used binary sentiment model
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Training of model: test_split_technique:
@st.cache_resource(show_spinner=False)
def train_sklearn_sentiment(texts: pd.Series, labels: pd.Series):
    # labels expected as strings: positive/neutral/negative (case-insensitive is handled)

    y = labels.astype(str).str.lower().replace({
        "pos": "positive",
        "neg": "negative",
        "neu": "neutral",
        "n": "negative",
        "p": "positive"
    })
    X_train, X_test, y_train, y_test = train_test_split(texts, y, test_size=0.2, random_state=42, stratify=y)
    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=50000)
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)
    clf = LogisticRegression(max_iter=200)
    clf.fit(Xtr, y_train)
    y_pred = clf.predict(Xte)
    report = classification_report(y_test, y_pred, output_dict=False)
    return vectorizer, clf, report


# Date handling:

def safe_parse_date(x):
    if pd.isna(x):
        return None
    try:
        return parser.parse(str(x), dayfirst=False, yearfirst=True)
    except Exception:
        return None

# Transcribe with whisper:
def transcribe_with_whisper(audio_bytes: bytes, model, filename: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1] or ".wav") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        path = tmp.name
    try:
        result = model.transcribe(path)
        return result.get("text", "").strip()
    finally:
        try:
            os.remove(path)
        except Exception:
            pass

# Transcribe with vosk:

def transcribe_with_vosk(audio_bytes: bytes, model, filename: str) -> str:
    # Vosk expects WAV PCM 16k mono. We'll try to coerce using wave if already wav; otherwise rely on ffmpeg via whisper isn't possible here.
    # For simplicity: if not WAV, we save and try to open. If not WAV PCM, we warn the user.
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1] or ".wav") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        path = tmp.name
    try:
        if not path.lower().endswith('.wav'):
            return "[Vosk] Please upload WAV PCM audio (16k mono) or use Whisper for auto-conversion."
        wf = wave.open(path, "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            return "[Vosk] WAV must be mono 16-bit PCM. Convert your file or use Whisper."
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)
        text_pieces = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                res = rec.Result()
                text_pieces.append(res)
        final = rec.FinalResult()
        text_pieces.append(final)
        # Combine naive
        return " ".join(text_pieces)
    except Exception as e:
        return f"[Vosk Error] {e}"
    finally:
        try:
            os.remove(path)
        except Exception:
            pass
        print('VOSK Path Removal Error')

# ------------ UI Part ---------------
# -----------------------------
# File Uploaders
# -----------------------------
st.subheader("1) Upload Data")
col1, col2 = st.columns([1,1])
with col1:

    csv_file = st.file_uploader("Upload CSV logs (remarks, student, year, tech stack, location, date, optional label)", type=["csv"]) 

with col2:
    audio_files = st.file_uploader("Upload call recordings (mp3/wav)", type=["mp3", "wav", "m4a", "aac"], accept_multiple_files=True)
# ------------ Data Framing and Column Mapping ------------,
# ------------------------.
# Load DataFrame + Column Mapping,
# -----------------------------
if csv_file is not None:
    try:
        df_raw = pd.read_csv(csv_file)
    except Exception:
        df_raw = pd.read_csv(csv_file, encoding="latin-1")
    st.success(f"CSV loaded with shape {df_raw.shape}")
    with st.expander("Map columns (flexible)"):
        cols = ["<none>"] + list(df_raw.columns)
        map_student = st.selectbox("Student Name column", cols, index=cols.index("student_name") if "student_name" in df_raw.columns else 0)
        map_year = st.selectbox("Year column", cols, index=cols.index("year") if "year" in df_raw.columns else 0)
        map_stack = st.selectbox("Tech Stack column", cols, index=cols.index("tech_stack") if "tech_stack" in df_raw.columns else 0)
        map_loc = st.selectbox("Location column", cols, index=cols.index("location") if "location" in df_raw.columns else 0)
        map_remarks = st.selectbox("Remarks/Notes column", cols, index=cols.index("remarks") if "remarks" in df_raw.columns else 0)
        map_callid = st.selectbox("Call ID column (optional)", cols, index=cols.index("call_id") if "call_id" in df_raw.columns else 0)
        map_date = st.selectbox("Date column (optional)", cols, index=cols.index("date") if "date" in df_raw.columns else 0)
        map_label = st.selectbox("Sentiment label column (optional: positive/neutral/negative)", cols, index=cols.index("label") if "label" in df_raw.columns else 0)

    def pick(colname):
        return None if colname == "<none>" else df_raw[colname]

    df = pd.DataFrame({
        "call_id": pick(map_callid) if map_callid != "<none>" else pd.Series([None]*len(df_raw)),
        "student_name": pick(map_student) if map_student != "<none>" else pd.Series([None]*len(df_raw)),
        "year": pick(map_year) if map_year != "<none>" else pd.Series([None]*len(df_raw)),
        "tech_stack": pick(map_stack) if map_stack != "<none>" else pd.Series([None]*len(df_raw)),
        "location": pick(map_loc) if map_loc != "<none>" else pd.Series([None]*len(df_raw)),
        "remarks": pick(map_remarks) if map_remarks != "<none>" else pd.Series([""]*len(df_raw)),
        "date": pick(map_date) if map_date != "<none>" else pd.Series([None]*len(df_raw)),
        "label": pick(map_label) if map_label != "<none>" else pd.Series([None]*len(df_raw)),
    })
    if "date" in df.columns:
        df["date_parsed"] = df["date"].apply(safe_parse_date)
    else:
        df["date_parsed"] = None
else:
    df = None
#=======================Transcribe Process ================================
# mp3 files (audio file) -> text data 
# -----------------------------
# Transcribe Audio
# -----------------------------
transcripts = []
if audio_files:
    st.subheader("2) Transcribe Audio")

#  Hanlding the Transcribe process as per the model selection:

    if asr_engine == "Whisper":
        try:
            whisper_model = load_whisper(whisper_size)
        except Exception as e:
            st.error(str(e))
            whisper_model = None
    else:
#vosk model :
#         
        try:
            vosk_model = load_vosk(vosk_model_dir)
        except Exception as e:
            st.error(str(e))
            vosk_model = None

# steps for the first time model will be downlaod and stored in the 
# cache memory. till that time your screen will be freezed or not.
# so there should be some mechanism to load the progress. or No.
# Progress bar must be used to share the progress how much model is loaded.
# 0% --> 20% --> 30% ---> 100%

    prog = st.progress(0)
    for i, f in enumerate(audio_files):#generator functions:.
        audio_bytes = f.read()
        if asr_engine == "Whisper" and whisper_model is not None:
            text = transcribe_with_whisper(audio_bytes, whisper_model, f.name)
        elif asr_engine == "Vosk (offline)" and 'vosk_model' in locals() and vosk_model is not None:
            text = transcribe_with_vosk(audio_bytes, vosk_model, f.name)
        else:
            text = "[ASR not available]"
        transcripts.append({
            "call_id": os.path.splitext(os.path.basename(f.name))[0],
            "transcript_text": text
        })
        prog.progress(int(((i+1)/len(audio_files))*100))

# Make new data frame using transcribe process:.
    st.success(f"Transcribed {len(transcripts)} file(s)")

if transcripts:
    df_tr = pd.DataFrame(transcripts)
else:
    df_tr = pd.DataFrame(columns=["call_id", "transcript_text"])  # empty
# merge the transcribe process with csv : data set audio and text files 
# final data set with merging call logs and data set.)
# -----------------------------
# Merge Transcripts with CSV
# -----------------------------
if df is not None:
    st.subheader("3) Merge Logs + Transcripts")
    # Try join on call_id if available, else outer merge on index
    if "call_id" in df.columns:
        df["call_id"] = df["call_id"].astype(str)
    if "call_id" in df_tr.columns:
        df_tr["call_id"] = df_tr["call_id"].astype(str)

    if df["call_id"].notna().any() and not df_tr.empty:
        merged = pd.merge(df, df_tr, on="call_id", how="outer")
    else:
        # append transcripts as new rows if missing call_id
        merged = df.copy()
        if not df_tr.empty:
            extra = pd.DataFrame({
                "call_id": df_tr["call_id"],
                "student_name": None,
                "year": None,
                "tech_stack": None,
                "location": None,
                "remarks": "",
                "date": None,
                "label": None,
                "date_parsed": None,
                "transcript_text": df_tr["transcript_text"]
            })
            merged = pd.concat([merged, extra], ignore_index=True)

        merged["remarks"] = merged.get("remarks", pd.Series([""] * len(merged))).fillna("")
    if "transcript_text" not in merged.columns:
        merged["transcript_text"] = ""
    else:
        merged["transcript_text"] = merged["transcript_text"].fillna("")
        merged["combined_text"] = (merged["remarks"].astype(str) + " " + merged["transcript_text"].astype(str)).str.strip()

    st.dataframe(merged.head(50), use_container_width=True)
else:
    merged = None

# -----------------------------
# Sentiment: Train or Pretrained
# -----------------------------
if merged is not None and len(merged) > 0:
    st.subheader("4) Sentiment Analysis")
    can_train = ("label" in merged.columns) and merged["label"].notna().any() and not use_pretrained

    if can_train:
        st.write("Training custom TF-IDF + LogisticRegression on provided labels…")
        with st.spinner("Training model…"):
            try:
                vectorizer, clf, report = train_sklearn_sentiment(merged["combined_text"].fillna("") , merged["label"])
                st.text("Classification report (hold-out test):\n" + report)
# Prediduction  algorithem for the custom model
#x data  : train
#y data : test
# Predict using trained model:....

                Xall = vectorizer.transform(merged["combined_text"].fillna(""))
                merged["sentiment"] = clf.predict(Xall)
                merged["sentiment_score"] = np.nan  # not provided by sklearn model
                model_used = "custom_sklearn"
            except Exception as e:
                st.error(f"Training failed: {e}. Falling back to pretrained pipeline.")
                can_train = False
 #if training is failes :....

    if not can_train:
        with st.spinner("Running pretrained sentiment model…"):
            nlp = load_hf_pipeline()
            preds = []
            scores = []
            for txt in merged["combined_text"].fillna(""):
                try:
                    r = nlp(txt[:4096])[0]  # limit length for speed   ,# 1024 ,2542 ,5846
                    label = r["label"].lower()
                    #positive ,negative  or netural:......
                    # Map to positive/neutral/negative (HF model is binary pos/neg)

                    if label == "positive":
                        preds.append("positive")
                    elif label == "negative":
                        preds.append("negative")
                    else:
                        preds.append(label)
                    scores.append(float(r.get("score", np.nan)))
                except Exception:
                    preds.append("neutral")
                    scores.append(np.nan)
            merged["sentiment"] = preds
            merged["sentiment_score"] = scores
            model_used = "hf_distilbert"

    st.success(f"Sentiment computed using: {model_used}")

    # -----------------------------
    # Analytics
    # -----------------------------
    #locattion , tech_stack:....

    st.subheader("5) Analytics")
    colA, colB, colC = st.columns(3)
    with colA:
        fig = px.pie(merged, names="sentiment", title="Sentiment Distribution")
        st.plotly_chart(fig, use_container_width=True)
    #locatiom  Analytics:...

    with colB:
        if "location" in merged.columns:
            fig2 = px.bar(merged.fillna({"location": "Unknown"}), x="location", color="sentiment", title="Sentiment by Location")
            st.plotly_chart(fig2, use_container_width=True)
    #tech stak Analytics:...
    with colC:
        if "tech_stack" in merged.columns:
            fig3 = px.bar(merged.fillna({"tech_stack": "Unknown"}), x="tech_stack", color="sentiment", title="Sentiment by Tech Stack")
            st.plotly_chart(fig3, use_container_width=True)

    # Time trend analysis:...
    if "date_parsed" in merged.columns and merged["date_parsed"].notna().any():
        temp = merged.copy()
        temp["month"] = temp["date_parsed"].dt.to_period("M").astype(str)
        ts = temp.groupby(["month", "sentiment"]).size().reset_index(name="count")
        fig4 = px.line(ts, x="month", y="count", color="sentiment", markers=True, title="Monthly Sentiment Trend")
        st.plotly_chart(fig4, use_container_width=True)

    # Top Negative Keywords (TF-IDF on negative rows)
    st.markdown("### Top Negative Keywords")
    neg = merged[merged["sentiment"] == "negative"]["combined_text"].dropna()
    if len(neg) >= 3:
        vec = TfidfVectorizer(max_features=50, stop_words="english")
        X = vec.fit_transform(neg)
        #summations:... 
        # Sum TF-IDF across docs:...
        sums = np.asarray(X.sum(axis=0)).ravel()
        #vocabarly:...........
        
        vocab = np.array(vec.get_feature_names_out())
        kw_df = pd.DataFrame({"keyword": vocab, "score": sums}).sort_values("score", ascending=False).head(20)
        fig5 = px.bar(kw_df, x="keyword", y="score", title="Top Negative Keywords (TF-IDF)")
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info("Not enough negative samples to extract keywords.")

    # -----------------------------
    # Recommendations (rule + keyword heuristics)
    # -----------------------------
    st.subheader("6) Recommendations")

    def gen_recos(df_: pd.DataFrame):
        recos = []
        # Overall negativity
        total = len(df_)
        negc = (df_["sentiment"] == "negative").sum()
        if total > 0 and negc/total > 0.4:
            recos.append("Overall negative sentiment is high (>40%). Consider immediate coaching for counselors and revising scripts.")

        # By location
        if "location" in df_.columns:
            by_loc = df_.groupby("location")["sentiment"].apply(lambda s: (s=="negative").mean()).sort_values(ascending=False)
            for loc, ratio in by_loc.items():
                if pd.notna(loc) and ratio >= 0.35:
                    recos.append(f"{loc}: Negative ratio {ratio:.0%}. Trial: reduce fees, add evening/weekend batches, or senior counselor callbacks.")

        # By stack
        if "tech_stack" in df_.columns:
            by_stack = df_.groupby("tech_stack")["sentiment"].apply(lambda s: (s=="negative").mean()).sort_values(ascending=False)
            for stack, ratio in by_stack.items():
                if pd.notna(stack) and ratio >= 0.35:
                    tips = {
                        "python": "emphasize job outcomes with case studies, add mini capstone demo",
                        "java": "offer installment plans, highlight placement partners",
                        "mern": "show live project repos and alumni testimonials",
                        "ai": "clarify math prerequisites and provide bridge modules"
                    }
                    extra = ""
                    k = str(stack).lower()
                    for key, val in tips.items():
                        if key in k:
                            extra = "; " + val
                            break
                    recos.append(f"{stack}: Negative ratio {ratio:.0%}. Address objections via FAQs{extra}.")

        # Keyword-based
        text_all = " ".join(df_.get("combined_text", pd.Series(dtype=str)).dropna().astype(str).tolist()).lower()
        if any(k in text_all for k in ["fee", "fees", "price", "cost", "expensive"]):
            recos.append("Many fee-related objections → try scholarships, limited-time discounts, or EMI options.")
        if any(k in text_all for k in ["time", "timing", "slot", "schedule", "evening", "weekend"]):
            recos.append("Timing objections → add evening/weekend batches and flexible slots.")
        if any(k in text_all for k in ["location", "distance", "noida", "lucknow", "commute"]):
            recos.append("Location/commute issues → promote online/hybrid option and campus transfer flexibility.")
        if any(k in text_all for k in ["doubt", "support", "mentor", "teacher", "faculty"]):
            recos.append("Learning support concerns → advertise mentorship hours, doubt-solving sessions, and WhatsApp/Slack groups.")
        if any(k in text_all for k in ["job", "placement", "interview", "resume"]):
            recos.append("Career outcomes focus → showcase placement stats, resume/interview prep workshops.")
        return recos

    recommendations = gen_recos(merged)
    if recommendations:
        for r in recommendations:
            st.markdown(f"-{r}")
    else:
        st.info("No strong recommendations detected. With more data, insights will improve.")

    # -----------------------------
    # Downloads
    # -----------------------------
    st.subheader("7) Export")
    if save_intermediate:
        out_csv = merged.copy()
        out_buf = io.StringIO()
        out_csv.to_csv(out_buf, index=False)
        st.download_button("Download processed CSV", data=out_buf.getvalue(), file_name=f"softpro_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")

else:
    st.info("Upload at least a CSV or audio to proceed.")

# End of app

