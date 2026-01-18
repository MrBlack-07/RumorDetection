from flask import Flask, request, render_template, jsonify
import pickle
import feedparser
from deep_translator import GoogleTranslator
import random

app = Flask(__name__)

# --- LOAD MODELS ---
try:
    model = pickle.load(open('pac.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
except:
    print("WARNING: Models not found. Run train.py first!")

# --- HELPER FUNCTIONS ---
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

def get_top_rumors():
    """
    Returns a list of trending rumors for the demo. 
    You can update these strings to match current events!
    """
    return [
        {
            "text": "RBI to withdraw ₹500 notes next month?",
            "status": "Viral on WhatsApp",
            "verdict": "FAKE"
        },
        {
            "text": "New 'Cyber-Kidnap' scam targeting Chennai parents",
            "status": "Trending Now",
            "verdict": "REAL THREAT"
        },
        {
            "text": "Free iPhone 15 giveaway link by Govt of India",
            "status": "Shared 10k times",
            "verdict": "SCAM"
        }
    ]

# --- ROUTES ---
@app.route('/')
def home():
    news = fetch_news_rss("Tamil Nadu", lang='ta')
    rumors = get_top_rumors() # Get the top rumors
    return render_template('home.html', news_data=news, current_feed="Tamil News", rumors=rumors)

@app.route('/get_feed', methods=['POST'])
def get_feed():
    feed_type = request.json.get('type')
    if feed_type == 'tamil':
        news = fetch_news_rss("Tamil Nadu", lang='ta')
    else:
        news = fetch_news_rss("Tamil Nadu", lang='en')
    return jsonify(news)

@app.route('/search_news', methods=['POST'])
def search_news():
    raw_query = request.form.get('query')
    lang_mode = request.form.get('search_lang')
    
    search_query = raw_query
    if lang_mode == 'tanglish':
        try:
            search_query = GoogleTranslator(source='auto', target='en').translate(raw_query)
        except:
            pass
            
    news = fetch_news_rss(search_query, lang='en')
    rumors = get_top_rumors()
    return render_template('home.html', news_data=news, current_feed=f"Results for: {raw_query}", search_val=raw_query, rumors=rumors)

@app.route('/predict', methods=['POST'])
def predict():
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