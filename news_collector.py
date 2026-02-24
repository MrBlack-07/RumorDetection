import os
import json
import feedparser
from urllib.parse import urlparse
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)
RAW_FILE = os.path.join(DATA_DIR, 'news_raw.jsonl')

QUERIES = [
    'India', 'World', 'Narendra Modi', 'PM Modi', 'India news', 'BJP', 'Congress'
]

def _get_domain(url):
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ''

def collect_news(max_per_query=20, lang='en'):
    """Collect news from Google News RSS and append to `data/news_raw.jsonl`.
    Returns number of items added."""
    added = 0
    seen_links = set()
    if os.path.exists(RAW_FILE):
        with open(RAW_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    seen_links.add(obj.get('link'))
                except Exception:
                    continue

    with open(RAW_FILE, 'a', encoding='utf-8') as out:
        for q in QUERIES:
            query = q.replace(' ', '+')
            url = f"https://news.google.com/rss/search?q={query}&hl={('en' if lang=='en' else 'en')}&gl=IN&ceid=IN:en"
            feed = feedparser.parse(url)
            count = 0
            for entry in feed.entries:
                if count >= max_per_query:
                    break
                link = entry.get('link')
                if not link or link in seen_links:
                    continue
                item = {
                    'title': entry.get('title', ''),
                    'link': link,
                    'published': entry.get('published', ''),
                    'source': entry.get('source').title if hasattr(entry, 'source') else entry.get('source', ''),
                    'domain': _get_domain(link),
                    'collected_at': datetime.utcnow().isoformat() + 'Z'
                }
                out.write(json.dumps(item, ensure_ascii=False) + '\n')
                seen_links.add(link)
                added += 1
                count += 1
    return added

if __name__ == '__main__':
    n = collect_news()
    print(f"Collected {n} new items")
