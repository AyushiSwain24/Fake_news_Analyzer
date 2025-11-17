import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# Setup
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

# 1. PREPARE TOPIC DATA (Only from Real news because they have valid tags)
if os.path.exists("news_dataset.csv"):
    df_real = pd.read_csv("news_dataset.csv")
    # Clean
    df_real['clean_text'] = df_real['text'].apply(clean_text)
    
    # Save for Topic Model
    df_topic = df_real[['clean_text', 'topic']].rename(columns={'clean_text': 'text'})
    df_topic = df_topic.dropna()
    df_topic.to_csv("topic_training_data.csv", index=False)
    print("Saved topic_training_data.csv")
else:
    print("Warning: news_dataset.csv missing")

# 2. PREPARE FAKE/REAL DATA
if os.path.exists("fake_news.csv") and os.path.exists("news_dataset.csv"):
    df_fake = pd.read_csv("fake_news.csv")
    df_real = pd.read_csv("news_dataset.csv") # Load again to be safe
    
    # We only need text and label for this one
    df_combined = pd.concat([df_real[['text', 'label']], df_fake[['text', 'label']]])
    
    df_combined['clean_text'] = df_combined['text'].apply(clean_text)
    final_df = df_combined[['clean_text', 'label']].rename(columns={'clean_text': 'text'})
    
    final_df.to_csv("fakereal_training_data.csv", index=False)
    print("Saved fakereal_training_data.csv")