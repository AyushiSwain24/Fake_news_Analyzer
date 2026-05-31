import requests
from bs4 import BeautifulSoup
import sqlite3
import datetime
import time
import json
import sys

# Headers to mimic a browser
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
}

# Load RSS feeds
try:
    with open("feeds.json", "r", encoding="utf-8") as f:
        rss_feeds = json.load(f)
except FileNotFoundError:
    print("❌ Error: feeds.json not found!")
    sys.exit(1)
except json.JSONDecodeError:
    print("❌ Error: feeds.json is not valid JSON!")
    sys.exit(1)

# Database setup with error handling
try:
    conn = sqlite3.connect("news.db", timeout=10)
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
    conn.commit()
except sqlite3.Error as e:
    print(f"❌ Database error: {e}")
    sys.exit(1)

try:
    for topic, urls in rss_feeds.items():
        print(f"🔎 Starting scrape for {topic}...")
        for url in urls:
            try:
                response = requests.get(url, headers=HEADERS, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, "xml")
                items = soup.find_all("item")

                for item in items:
                    try:
                        title = item.title.text.strip() if item.title else None
                        link = item.link.text.strip() if item.link else None
                        published = item.pubDate.text.strip() if item.pubDate else None
                        scraped_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        # Validate data before inserting
                        if title and link and len(title) > 0 and len(link) > 0:
                            cursor.execute("""
                                INSERT INTO articles (topic, title, link, published, scraped_at)
                                VALUES (?, ?, ?, ?, ?)
                            """, (topic, title, link, published, scraped_at))
                            print(f"[{topic}] ✅ Inserted: {title[:50]}...")
                    except sqlite3.IntegrityError:
                        pass  # Skip duplicates
                    except Exception as e:
                        print(f"[{topic}] ⚠️  Error processing item: {str(e)[:50]}")
                
                conn.commit()
            except requests.Timeout:
                print(f"[{topic}] ⏱️  Timeout fetching {url}")
            except requests.RequestException as e:
                print(f"[{topic}] ❌ Error fetching {url}: {str(e)[:50]}")
            except Exception as e:
                print(f"[{topic}] ⚠️  Unexpected error for {url}: {str(e)[:50]}")
            
            time.sleep(2)  # Increased delay to be respectful to servers

    conn.commit()
    print("✅ Scraping complete!")
except KeyboardInterrupt:
    print("\n⏸️  Scraping paused by user")
except Exception as e:
    print(f"❌ Fatal error: {e}")
finally:
    conn.close()
    print("Database connection closed.")