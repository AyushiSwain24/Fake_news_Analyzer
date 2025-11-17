-----

# Fake News Detection System ⚖️

## 🛠️ Build from Scratch

If you want to build the entire process from the start, please refer to the detailed guide below:
[**View Full Documentation (Google Docs)**](https://docs.google.com/document/d/16poL-YXzdK63F48aiGn_J6kP6ywESZqaW7QF-tu7CpI/edit?usp=sharing)

-----

## 🚀 Quick Start (Essentials Only)

To run the application without rebuilding the models, simply download the essential files and follow the steps below.

### 1\. Download Required Files

Ensure you have the following 6 files in the same directory:

  * **Model Files:**
      * `model_fake.pkl` (Fake News Detector model)
      * `vectorizer_fake.pkl` (Vocabulary for Fake Detector)
      * `model_topic.pkl` (Topic Classifier model)
      * `vectorizer_topic.pkl` (Vocabulary for Topic Classifier)
  * **Application Files:**
      * `app.py`
      * `requirements.txt`

### 2\. Prerequisites

  * **Python Version:** Recommended **Python 3.11** or **3.12**.

### 3\. Installation & Execution

Open your terminal/command prompt, navigate to the directory containing your downloaded files, and run the following commands:

**Install Dependencies:**

```bash
pip install -r requirements.txt
```

**Run the Application:**

```bash
streamlit run app.py
```
