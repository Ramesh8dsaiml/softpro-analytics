<div align="center">

#  Softpro Analytics  
### **AI-powered Audio Transcription, Sentiment Analysis & Sales Insights Platform**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-ff4b4b)
![NLP](https://img.shields.io/badge/NLP-Transformers-yellow)
![License](https://img.shields.io/badge/License-Academic-green)

</div>

--------------------------------------------------------------------------------

#  Table of Contents
- [Overview](#overview)
- [Features](#features)
- [System Workflow](#system-workflow)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Training Output (PDF Images)](#training-output-pdf-images)
- [How to Run](#how-to-run)
- [Dataset Requirements](#dataset-requirements)
- [Sentiment Model Modes](#sentiment-model-modes)
- [Analytics Provided](#analytics-provided)
- [Recommendation Engine](#recommendation-engine)
- [Future Scope](#future-scope)
- [Author](#author)
- [License](#license)
  

--------------------------------------------------------------------------------
#Project Report:
## click here go to report and Download [Softpro_Analytics_ReportFinal]  ("https://github.com/Ramesh8dsaiml/softpro-analytics/blob/main/Softpro_Analytics_Report%20Final.docx"

<img width="682" height="810" alt="{5416BB94-2D12-4978-B65F-31599041D3AE}" src="https://github.com/user-attachments/assets/35a607bc-800b-40a5-a3a3-7c8a8eaab103" />
<img width="711" height="808" alt="{4CE21652-34D7-4CE2-AC86-928064D6FA01}" src="https://github.com/user-attachments/assets/3a0d9125-043e-4747-98c9-840521935594" />




#  Overview

# softpro-analytics
(company_name - softpro)

#AI-powered platform for audio transcription, sentiment analysis, CRM merging &amp; sales insights.

# Softpro Analytics – Sentiment & Sales Insights

A complete AI-powered Streamlit platform that converts **audio counselling calls + CRM logs** into:  
- Structured transcripts  
- Sentiment insights  
- Negative keyword patterns  
- Tech-stack & location-based analytics  
- Actionable recommendations to improve conversions
- Built using **Whisper/Vosk ASR**, **Transformers**, **Scikit-Learn**, and an **interactive Streamlit dashboard**.
 

-------------------------------------------------------------------> 

#  Features

###  **Audio Processing**
- Upload MP3/WAV call recordings  
- High-quality speech-to-text using **Whisper** (OpenAI)  
- Offline support via **Vosk**

###  **CRM Log Processing**
- Upload CSV logs  
- Auto-map columns  
- Merge call transcripts + counselor remarks

###  **Sentiment Analysis**
- Pretrained DistilBERT (Binary: pos/neg)  
- OR custom ML model (TF-IDF + Logistic Regression)


✔ **Interactive Analytics Dashboard**  
- Sentiment distribution  
- Location-wise analysis  
- Tech-stack-wise performance  
- Monthly sentiment trend  
- Top negative keywords  

✔ **Recommendation Engine**  

Automatically identifies issues:
- Fees  
- Timing  
- Location  
- Placement  
- Faculty support  

And generates **actionable suggestions**.

✔ Export final processed dataset  

------------------------------------------------------------------------------------------------------------------>

##  System Workflow

Audio Files
↓
Transcription (Whisper/Vosk)
↓
CRM Merge
↓
Sentiment Analysis
↓
Analytics Dashboard
↓
Recommendations

---

## 🛠️ Tech Stack

### **Languages & Libraries**
- Python 3.10+
- Streamlit  
- Whisper / Vosk  
- Hugging Face Transformers  
- Scikit-Learn  
- Pandas, NumPy  
- Plotly  

### **AI Models**
- Whisper ASR (tiny/base/small/medium)  
- DistilBERT Sentiment model  
- Logistic Regression (Custom training option)

---

## 📂 Project Structure

softpro-analytics/
│── app.py # Main Streamlit App
│── requirements.txt # Package list
│── sample.csv # Demo CRM Log
│── recordings/ # Demo audio (optional)
│── README.md # Project documentation
│── softpro_page_1.png
│── softpro_page_2.png
│── softpro_page_3.png
│── screenshots/ # Dashboard images



---------------------------------------------------------------------------------------------------------------------------------------------------------------------------@

#  Training Output (From PDF)

###  Page 1  
<img width="1700" height="2200" alt="softpro_page_1" src="https://github.com/user-attachments/assets/178add85-f160-4b48-85e1-d371db9013ab" />
###  Page 2 
<img width="1700" height="2200" alt="softpro_page_2 (1)" src="https://github.com/user-attachments/assets/7703077c-4eee-48d4-90cb-731472c76a4e" />
###  Page 3  
<img width="1700" height="2200" alt="softpro_page_3" src="https://github.com/user-attachments/assets/c2834804-9d1b-490c-aa0d-f0c12374db90" />

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------@

##  How to Run Locally

### **1. Create Virtual Environment**
```bash
python -m venv myenv
myenv\Scripts\activate     # Windows


. Install Dependencies
pip install -r requirements.txt

If Whisper gives FFmpeg error:
Add this to system PATH:


#Run App
streamlit run app.py

-----> Dataset Requirements
CRM Log CSV (Required Columns)
student_name  
year  
tech_stack  
location  
remarks  
call_id  
date  
label (optional: positive/neutral/negative)

----> Audio Support

mp3, wav, m4a, aac

Whisper auto-converts

Vosk requires WAV, 16-bit PCM, mono

-------> Sentiment Model Modes
🔹 Pretrained Mode (Default)

Uses Huggingface DistilBERT

Outputs: positive / negative + confidence

🔹 Custom Training Mode

Triggered when CSV has a label column

Uses TF-IDF + Logistic Regression

Generates Classification Report



------> Analytics Provided

 Sentiment Distribution

 Location-wise Sentiment Comparison

 Tech-stack-wise Analysis

 Monthly Sentiment Trend

 Negative Keyword Extraction (TF-IDF)

 Keyword-based Objection Patterns



------> Recommendation Engine

Automatically detects issues and generates suggestions:

Issue Type	Recommended Action
Fees	EMI plans, scholarships, limited-time offers
Timing	Add evening/weekend batches
Placement	Highlight alumni success, workshops
Location	Provide hybrid/online options
Faculty/Support	Extra mentor hours, doubt sessions



----> Future Scope

Hindi/Hinglish ASR Support

Real-time CRM Integration

Emotional Tone Detection

Dynamic Lead Scoring

Mobile Responsive UI

Author

Ramesh Kumar
B.Tech AI & Data Science (2022–2026)
(AKTU Lucknow)
Future Institute of Engineering & Technology, Bareilly

License

This project is built for academic and educational purposes.








