import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import sys

# Setup
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
except Exception as e:
    print(f"❌ Error setting up NLTK: {e}")
    sys.exit(1)

def clean_text(text):
    """Clean text with input validation"""
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 1]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

# 1. PREPARE TOPIC DATA (Only from Real news because they have valid tags)
if os.path.exists("news_dataset.csv"):
    try:
        df_real = pd.read_csv("news_dataset.csv")
        
        # Validate data
        if 'text' not in df_real.columns or 'topic' not in df_real.columns:
            print("❌ Error: Required columns 'text' and 'topic' not found in news_dataset.csv")
            sys.exit(1)
        
        # Clean
        df_real['clean_text'] = df_real['text'].apply(clean_text)
        
        # Save for Topic Model
        df_topic = df_real[['clean_text', 'topic']].rename(columns={'clean_text': 'text'})
        df_topic = df_topic.dropna()
        
        if len(df_topic) == 0:
            print("❌ Error: No valid data after cleaning for topic model")
            sys.exit(1)
        
        df_topic.to_csv("topic_training_data.csv", index=False)
        print(f"✅ Saved topic_training_data.csv ({len(df_topic)} records)")
    except Exception as e:
        print(f"❌ Error processing news_dataset.csv: {e}")
        sys.exit(1)
else:
    print("⚠️  Warning: news_dataset.csv missing - topic model training skipped")

# 2. PREPARE FAKE/REAL DATA
if os.path.exists("fake_news.csv") and os.path.exists("news_dataset.csv"):
    try:
        df_fake = pd.read_csv("fake_news.csv")
        df_real = pd.read_csv("news_dataset.csv")
        
        # Validate required columns
        if 'text' not in df_fake.columns or 'label' not in df_fake.columns:
            print("❌ Error: Required columns not found in fake_news.csv")
            sys.exit(1)
        if 'text' not in df_real.columns or 'label' not in df_real.columns:
            print("❌ Error: Required columns not found in news_dataset.csv")
            sys.exit(1)
        
        # We only need text and label for this one
        df_combined = pd.concat([df_real[['text', 'label']], df_fake[['text', 'label']]], ignore_index=True)
        
        # Remove duplicates and null values
        df_combined = df_combined.drop_duplicates(subset=['text']).dropna()
        
        df_combined['clean_text'] = df_combined['text'].apply(clean_text)
        final_df = df_combined[['clean_text', 'label']].rename(columns={'clean_text': 'text'})
        
        if len(final_df) == 0:
            print("❌ Error: No valid data after cleaning")
            sys.exit(1)
        
        final_df.to_csv("fakereal_training_data.csv", index=False)
        print(f"✅ Saved fakereal_training_data.csv ({len(final_df)} records)")
    except Exception as e:
        print(f"❌ Error processing datasets: {e}")
        sys.exit(1)
else:
    print("⚠️  Warning: fake_news.csv or news_dataset.csv missing - fake/real model training skipped")