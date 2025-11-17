import sqlite3
import pandas as pd

conn = sqlite3.connect("news.db")

# Fetch title AND topic
df = pd.read_sql_query("SELECT title, topic FROM articles", conn)

# Rename for consistency
df = df.rename(columns={"title": "text"})
df["label"] = "real"

df.to_csv("news_dataset.csv", index=False)
print("CSV file created: news_dataset.csv (with topics)")