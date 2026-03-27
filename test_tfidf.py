from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import feedparser
from urllib.parse import quote_plus

def test_query(query):
    print(f"Query: {query}")
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)
    titles = " ".join([entry.title for entry in feed.entries[:5]])
    print(f"Titles: {titles}")
    
    try:
        vec = TfidfVectorizer(stop_words='english')
        vecs = vec.fit_transform([query, titles])
        sim = cosine_similarity(vecs[0:1], vecs[1:2]).flatten()[0]
        print(f"Similarity: {sim}\n")
    except:
        print("Error\n")

test_query("Did actor Vijay launch a political party?")
test_query("Is actor vijay dead recently?")
test_query("Is Narendra Modi the prime minister of India?")
