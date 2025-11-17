import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np

# Setup cleaning
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

# Load ALL models
print("Loading models...")
model_fake = joblib.load("model_fake.pkl")
vec_fake = joblib.load("vectorizer_fake.pkl")

model_topic = joblib.load("model_topic.pkl")
vec_topic = joblib.load("vectorizer_topic.pkl")

def predict_news(text):
    cleaned = clean_text(text)
    
    # 1. Predict Real vs Fake
    vec_f = vec_fake.transform([cleaned])
    prediction_label = model_fake.predict(vec_f)[0]
    
    # Calculate confidence (probability of the predicted class)
    probs = model_fake.predict_proba(vec_f)[0]
    confidence = np.max(probs) * 100  # Get the highest probability
    
    # 2. Predict Category
    vec_t = vec_topic.transform([cleaned])
    category = model_topic.predict(vec_t)[0]
    
    # Format the output string
    return f"{prediction_label.capitalize()}, category : {category}, confidence : {confidence:.0f}%"

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
    print(f"Output: {predict_news(s)}")
    print("-" * 30)