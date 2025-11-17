import requests
from bs4 import BeautifulSoup
import sqlite3
import datetime
import time
import json

# Headers to mimic a browser
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
}

# Load RSS feeds
with open("feeds.json", "r", encoding="utf-8") as f:
    rss_feeds = json.load(f)

# Database setup
conn = sqlite3.connect("news.db")
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT,
    title TEXT,
    link TEXT UNIQUE,
    published TEXT,
    scraped_at TEXT
)
""")

for topic, urls in rss_feeds.items():
    print(f"🔎 Starting scrape for {topic}...")
    for url in urls:
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "xml")
            items = soup.find_all("item")

            for item in items:
                title = item.title.text.strip() if item.title else None
                link = item.link.text.strip() if item.link else None
                published = item.pubDate.text.strip() if item.pubDate else None
                scraped_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if title and link:
                    try:
                        cursor.execute("""
                            INSERT INTO articles (topic, title, link, published, scraped_at)
                            VALUES (?, ?, ?, ?, ?)
                        """, (topic, title, link, published, scraped_at))
                        print(f"[{topic}] Inserted: {title[:50]}...")
                    except sqlite3.IntegrityError:
                        pass # Skip duplicates
        except Exception as e:
            print(f"[{topic}] Error fetching {url}: {e}")
        time.sleep(1) 

conn.commit()
conn.close()
print("✅ Scraping complete!")