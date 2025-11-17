import streamlit as st
import joblib
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Fake News Analyzer AI | Fake News Detector",
    page_icon="⚖️",
    layout="centered"
)

# --- CUSTOM CSS FOR MODERN LOOK ---
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    .stTextArea textarea {
        font-size: 16px !important;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 50px;
        font-size: 18px;
        font-weight: bold;
        background-color: #FF4B4B;
        color: white;
        border: none;
    }
    .stButton>button:hover {
        background-color: #FF2B2B;
        color: white;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 20px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SETUP & CACHING ---
# We cache resources so they don't reload on every interaction
@st.cache_resource
def load_resources():
    # Download NLTK data
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    
    # Load Models
    model_fake = joblib.load("model_fake.pkl")
    vec_fake = joblib.load("vectorizer_fake.pkl")
    model_topic = joblib.load("model_topic.pkl")
    vec_topic = joblib.load("vectorizer_topic.pkl")
    
    return model_fake, vec_fake, model_topic, vec_topic

model_fake, vec_fake, model_topic, vec_topic = load_resources()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# --- CLEANING FUNCTION ---
def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

# --- MAIN APP LAYOUT ---
st.title("⚖️ Fake News Analyzer AI")
st.caption("Advanced Fake News Detection & Topic Classification System")
st.divider()

# Input Section
st.subheader("Analyze an Article")
user_input = st.text_area("Paste the news headline or article text here:", height=150, placeholder="Ex: Government announces free mars vacation for all citizens...")

if st.button("🔍 Verify Authenticity"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing patterns and topics..."):
            # 1. Preprocess
            cleaned_text = clean_text(user_input)
            
            # 2. Predict Real vs Fake
            vec_f = vec_fake.transform([cleaned_text])
            pred_label = model_fake.predict(vec_f)[0]
            probs = model_fake.predict_proba(vec_f)[0]
            confidence = np.max(probs) * 100
            
            # 3. Predict Topic
            vec_t = vec_topic.transform([cleaned_text])
            pred_topic = model_topic.predict(vec_t)[0]

            # --- DISPLAY RESULTS ---
            st.markdown("### Analysis Results")
            
            # Layout: 2 columns (Verdict on left, Details on right)
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if pred_label == "real":
                    st.success(f"✅ **REAL NEWS**")
                    st.metric(label="Confidence Score", value=f"{confidence:.1f}%")
                else:
                    st.error(f"🚨 **FAKE NEWS**")
                    st.metric(label="Confidence Score", value=f"{confidence:.1f}%")
            
            with col2:
                st.info(f"📂 **Category: {pred_topic}**")
                # Create a simple progress bar for visual flair
                st.progress(int(confidence))
                st.caption("Model Certainty Level")

            # Expandable technical details
            with st.expander("See Technical Details"):
                st.text(f"Processed Text: {cleaned_text}")
                st.json({
                    "prediction": pred_label,
                    "category": pred_topic,
                    "confidence_score": round(confidence, 4),
                    "raw_probabilities": {
                        "fake": round(probs[0], 4),
                        "real": round(probs[1], 4)
                    }
                })