import nltk
import shutil
import os

# 1. Locate NLTK data
data_path = os.path.expanduser('~/nltk_data')
print(f"Checking for NLTK data at: {data_path}")

# 2. Force Delete 'stopwords' if it exists (to fix corruption)
stopwords_path = os.path.join(data_path, 'corpora', 'stopwords')
if os.path.exists(stopwords_path):
    print("Removing corrupted stopwords data...")
    shutil.rmtree(stopwords_path)

zip_path = os.path.join(data_path, 'corpora', 'stopwords.zip')
if os.path.exists(zip_path):
    os.remove(zip_path)

# 3. Re-download fresh
print("Downloading fresh stopwords...")
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

print("\n✅ Fix complete! Now run: streamlit run app.py")