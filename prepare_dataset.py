import sqlite3, os, re, html, unicodedata
import pandas as pd

DB_PATH = "news.db"
DATASET_DIR = "dataset"
MIN_LEN = 2

os.makedirs(DATASET_DIR, exist_ok=True)

def clean_title(text: str) -> str | None:
    if not text:
        return None
    text = html.unescape(str(text))
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\(.*?\)|\[.*?\]", " ", text)
    text = re.sub(r"\b\d{1,2}:\d{2}(?:\s?(?:AM|PM|IST|GMT|UTC))?\b", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"[-–|:]\s*\b[A-Z][A-Za-z]+\b$", "", text)
    text = re.sub(r"[^A-Za-z0-9\s&'’\-:]", " ", text)
    text = text.strip()
    if len(text.split()) < MIN_LEN:
        return None
    return text

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT title, topic FROM articles", conn)
conn.close()

df["title_clean"] = df["title"].apply(clean_title)
df = df.dropna(subset=["title_clean", "topic"])
df["topic"] = df["topic"].str.lower().str.strip()

for topic, group in df.groupby("topic"):
    path = os.path.join(DATASET_DIR, f"{topic}.txt")
    with open(path, "w", encoding="utf-8") as f:
        for title in group["title_clean"].unique():
            f.write(title + "\n")
    print(f"Saved {len(group)} items to {path}")
