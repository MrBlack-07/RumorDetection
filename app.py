from flask import Flask, request, render_template, jsonify
import pickle
import feedparser
from deep_translator import GoogleTranslator
import random
from datetime import datetime, timedelta

app = Flask(__name__)

# --- LOAD MODELS ---
try:
    model = pickle.load(open('pac.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
except:
    print("WARNING: Models not found. Run train.py first!")

# --- RUMOR DATABASE ---
RUMOR_POOL = [
    {"id": 1, "title": "RBI withdrawing ₹500 notes?", "status": "Viral on WhatsApp", "verdict": "FAKE", "content": "A viral message claims RBI is withdrawing ₹500 notes. The PIB Fact Check unit has clarified this is fake."},
    {"id": 2, "title": "Free iPhone 15 Govt Scheme", "status": "Shared 10k times", "verdict": "SCAM", "content": "Malicious links are spreading promising free iPhones. This is a phishing scam to steal data."},
    {"id": 3, "title": "Chennai Metro Ticket Hike", "status": "Trending", "verdict": "REAL", "content": "CMRL has announced a marginal hike in ticket prices for peak hours starting next month."},
    {"id": 4, "title": "New Lockdown in Tamil Nadu?", "status": "Panic Sharing", "verdict": "FAKE", "content": "Old videos of lockdown announcements are being shared as new. No lockdown has been announced."},
    {"id": 5, "title": "Solar Storm to hit Earth?", "status": "Sensationalism", "verdict": "EXAGGERATED", "content": "NASA predicts solar activity, but claims of a 'total internet blackout' are exaggerated."},
    {"id": 6, "title": "Digital ID mandatory for Voting", "status": "Discussion", "verdict": "REAL", "content": "Election commission is piloting digital voter IDs in select constituencies."},
    {"id": 7, "title": "Plastic Rice in Ration Shops", "status": "Viral Video", "verdict": "FAKE", "content": "Videos claiming plastic rice are false; it is actually fortified rice kernels which are healthy."},
    {"id": 8, "title": "WhatsApp '3 Ticks' Rule", "status": "Forwarded Many Times", "verdict": "FAKE", "content": "Government is NOT recording your calls or adding a '3rd Blue Tick' to WhatsApp."},
    {"id": 9, "title": "Free Laptop Scheme 2026", "status": "Link Scams", "verdict": "SCAM", "content": "Fake websites are collecting student data promising free laptops. Only use official .gov portals."},
    {"id": 10, "title": "Tsunami Warning for Chennai", "status": "Old Alert", "verdict": "OUTDATED", "content": "An old tsunami drill warning is being circulated as a real alert. There is no threat currently."}
]

# --- GLOBAL VARS ---
current_rumors = []
next_update_time = datetime.now()

def update_rumors_if_needed():
    global current_rumors, next_update_time
    if not current_rumors or datetime.now() > next_update_time:
        current_rumors = random.sample(RUMOR_POOL, 3)
        next_update_time = datetime.now() + timedelta(hours=2)

def fetch_news_rss(query, lang='en'):
    clean_query = query.replace(" ", "+")
    if lang == 'ta':
        url = f"https://news.google.com/rss/search?q={clean_query}&hl=ta&gl=IN&ceid=IN:ta"
    else:
        url = f"https://news.google.com/rss/search?q={clean_query}&hl=en-IN&gl=IN&ceid=IN:en"
    
    feed = feedparser.parse(url)
    posts = []
    for entry in feed.entries[:10]:
        posts.append({
            'title': entry.title,
            'link': entry.link,
            'published': entry.published,
            'source': entry.source.title if hasattr(entry, 'source') else 'Google News'
        })
    return posts

# --- ROUTES ---
@app.route('/')
def home():
    update_rumors_if_needed()
    news = fetch_news_rss("Tamil Nadu", lang='ta')
    return render_template('home.html', news_data=news, current_feed="Tamil News", rumors=current_rumors)

@app.route('/get_feed', methods=['POST'])
def get_feed():
    """API for JavaScript to fetch news without reloading"""
    feed_type = request.json.get('type')
    if feed_type == 'tamil':
        news = fetch_news_rss("Tamil Nadu", lang='ta')
    else:
        news = fetch_news_rss("India", lang='en')
    return jsonify(news)

@app.route('/search_news', methods=['POST'])
def search_news():
    update_rumors_if_needed()
    raw_query = request.form.get('query')
    lang_mode = request.form.get('search_lang')
    
    search_query = raw_query
    if lang_mode == 'tanglish':
        try:
            search_query = GoogleTranslator(source='auto', target='en').translate(raw_query)
        except:
            pass
            
    news = fetch_news_rss(search_query, lang='en')
    return render_template('home.html', news_data=news, current_feed=f"Results: {raw_query}", search_val=raw_query, rumors=current_rumors)

@app.route('/predict', methods=['POST'])
def predict():
    update_rumors_if_needed()
    if request.method == 'POST':
        news_text = request.form['news']
        language = request.form.get('language')
        
        final_text = news_text
        if language == 'tanglish':
            try:
                final_text = GoogleTranslator(source='auto', target='en').translate(news_text)
            except:
                pass

        vec_text = vectorizer.transform([final_text])
        prediction = model.predict(vec_text)
        
        return render_template('result.html', prediction=prediction[0], original=news_text, translated=final_text)

if __name__ == '__main__':
    app.run(debug=True)