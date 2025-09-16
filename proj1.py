import requests
from bs4 import BeautifulSoup
import sqlite3
import datetime

# -----------------------------
# Database setup
# -----------------------------
conn = sqlite3.connect("news.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT,
    title TEXT,
    link TEXT,
    published TEXT,
    scraped_at TEXT
)
""")

# -----------------------------
# RSS feeds grouped by topic
# -----------------------------
rss_feeds = {
    "Business": [
        "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "https://www.thehindubusinessline.com/feeder/default.rss",
        "https://timesofindia.indiatimes.com/rssfeeds/1898055.cms",
        "https://www.moneycontrol.com/rss/latestnews.xml",
        "https://www.news18.com/rss/business.xml"
    ],
    "Politics": [
        "https://www.thehindu.com/news/national/politics/feeder/default.rss",
        "https://indianexpress.com/section/political-pulse/feed/",
        "https://feeds.feedburner.com/ndtvnews-politics-news",
        "https://www.hindustantimes.com/feeds/rss/politics/rssfeed.xml",
        "https://www.indiatoday.in/rss/1206578"
    ],
    "Technology": [
        "https://indianexpress.com/section/technology/feed/",
        "https://timesofindia.indiatimes.com/rssfeeds/5880659.cms",
        "https://feeds.feedburner.com/NDTVGadgets360-AllTheLatest",
        "https://www.hindustantimes.com/feeds/rss/technology/rssfeed.xml",
        "https://www.news18.com/rss/tech.xml"
    ],
    "Sports": [
        "https://www.espncricinfo.com/rss/content/story/feeds/0.xml",
        "https://www.thehindu.com/sport/feeder/default.rss",
        "https://timesofindia.indiatimes.com/rssfeeds/4719148.cms",
        "https://feeds.feedburner.com/ndtvsports-latest",
        "https://www.news18.com/rss/sports.xml"
    ],
    "Entertainment": [
        "https://indianexpress.com/section/entertainment/feed/",
        "https://timesofindia.indiatimes.com/rssfeeds/1081479906.cms",
        "https://www.hindustantimes.com/feeds/rss/entertainment/rssfeed.xml",
        "https://www.bollywoodhungama.com/rssfeeds/bollywood-news.xml",
        "https://www.news18.com/rss/movies.xml"
    ],
    "Health": [
        "https://www.thehindu.com/sci-tech/health/feeder/default.rss",
        "https://timesofindia.indiatimes.com/rssfeeds/3908999.cms",
        "https://www.hindustantimes.com/feeds/rss/health-fitness/rssfeed.xml",
        "https://www.news18.com/rss/health-fitness.xml",
        "https://medicaldialogues.in/rss"
    ],
    "Education": [
        "https://timesofindia.indiatimes.com/rssfeeds/913168846.cms",
        "https://www.thehindu.com/education/feeder/default.rss",
        "https://www.hindustantimes.com/feeds/rss/education/rssfeed.xml",
        "https://www.indiatoday.in/rss/1206614",
        "https://feeds.feedburner.com/ndtvnews-education-news"
    ],
    "Science": [
        "https://www.thehindu.com/sci-tech/science/feeder/default.rss",
        "https://timesofindia.indiatimes.com/rssfeeds/5880659.cms",
        "https://indianexpress.com/section/technology/science/feed/",
        "https://www.hindustantimes.com/feeds/rss/science/rssfeed.xml",
        "https://www.downtoearth.org.in/rss/science.xml"
    ]
}

# -----------------------------
# Scraping loop
# -----------------------------
for topic, urls in rss_feeds.items():
    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, "xml")
            items = soup.find_all("item")

            for item in items:
                title = item.title.text if item.title else None
                link = item.link.text if item.link else None
                published = item.pubDate.text if item.pubDate else None
                scraped_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if title and link:
                    cursor.execute("""
                        INSERT INTO articles (topic, title, link, published, scraped_at)
                        VALUES (?, ?, ?, ?, ?)
                    """, (topic, title, link, published, scraped_at))

                    print(f"[{topic}] Inserted: {title[:80]}...")

        except Exception as e:
            print(f"[{topic}] Failed to scrape {url}: {e}")

# -----------------------------
# Commit and close
# -----------------------------
conn.commit()
conn.close()
print("✅ Scraping complete!")
