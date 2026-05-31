import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import sys
import os

# Setup cleaning
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
except Exception as e:
    print(f"❌ Error setting up NLTK: {e}")
    sys.exit(1)

def clean_text(text, max_length=5000):
    """Clean text with input validation and security checks"""
    if not isinstance(text, str):
        return ""
    
    # Prevent memory issues
    if len(text) > max_length:
        text = text[:max_length]
    
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 1]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

# Load ALL models with error handling
print("Loading models...")
models_needed = ["model_fake.pkl", "vectorizer_fake.pkl", "model_topic.pkl", "vectorizer_topic.pkl"]
missing_models = [m for m in models_needed if not os.path.exists(m)]

if missing_models:
    print(f"❌ Missing models: {missing_models}")
    print("Please run 'train_models.py' first!")
    sys.exit(1)

try:
    model_fake = joblib.load("model_fake.pkl")
    vec_fake = joblib.load("vectorizer_fake.pkl")
    model_topic = joblib.load("model_topic.pkl")
    vec_topic = joblib.load("vectorizer_topic.pkl")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    sys.exit(1)

def predict_news(text):
    """Predict news authenticity with error handling"""
    if not isinstance(text, str) or not text.strip():
        return "Error: Invalid input"
    
    try:
        cleaned = clean_text(text)
        
        if not cleaned or len(cleaned.split()) < 3:
            return "Error: Text too short after processing"
        
        # 1. Predict Real vs Fake
        vec_f = vec_fake.transform([cleaned])
        prediction_label = model_fake.predict(vec_f)[0]
        
        # Calculate confidence (probability of the predicted class)
        probs = model_fake.predict_proba(vec_f)[0]
        confidence = np.max(probs) * 100
        
        # 2. Predict Category
        vec_t = vec_topic.transform([cleaned])
        category = model_topic.predict(vec_t)[0]
        
        # Format the output string
        return f"{prediction_label.capitalize()}, category: {category}, confidence: {confidence:.1f}%"
    except Exception as e:
        return f"Error during prediction: {str(e)[:50]}"

# Test Cases
samples = [
    "NASA announces new mission to Mars landing in 2030", 
    "Government gives free gold to all citizens starting tomorrow!",
    "Lakers win the championship in a stunning overtime victory",
    "Apple releases new iPhone with holographic display",
    "Senate passes new bill regarding tax reforms for small businesses",
]

print("\n--- RESULTS ---")
for s in samples:
    print(f"Input: {s}")
    result = predict_news(s)
    print(f"Output: {result}")
    print("-" * 50)