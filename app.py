from flask import Flask, request, render_template, jsonify
import pickle
import feedparser
from deep_translator import GoogleTranslator
import random
from datetime import datetime, timedelta
import os
import json
import re
from urllib.parse import quote_plus, urlencode
from urllib.parse import urlparse
from news_collector import collect_news
from update_dataset import build_combined_csv
from qa_engine import get_qa_answer

app = Flask(__name__)

# --- RELIABLE NEWS SOURCES DATABASE ---
RELIABLE_SOURCES = {
    # High Reliability (Government & Major News Agencies)
    'reliable': [
        'pib.gov.in', 'ndtv.com', 'thehindu.com', 'indianexpress.com',
        'timesofindia.indiatimes.com', 'hindustantimes.com', 'news18.com',
        'reuters.com', 'apnews.com', 'bbc.com', 'bbc.news',
        'npr.org', 'wsj.com', 'nytimes.com', 'guardian.com',
        'economist.com', 'forbes.com', 'bloomberg.com',
        'pib', 'press information bureau', 'ani', 'pti',
        'doordarshan', 'all india radio', 'prasar bharati'
    ],
    # Medium Reliability
    'medium': [
        'indiatoday.in', 'abplive.com', 'zee.news', 'zeenews.india.com',
        'aajtak.in', 'teel.tv', 'newsx.com', 'ddnews.gov.in',
        'swarajya.id', 'thewire.in', 'scroll.in', 'print.in',
        'cnn.com', 'cnbc.com', 'aljazeera.com', 'dw.com'
    ],
    # Low Reliability / Fact Check Sites
    'factcheck': [
        'snopes.com', 'factcheck.org', 'politifact.com',
        'boomlive.in', 'altnews.in', 'factly.in', 'check4spam.com',
        'trapfalsenews.com', 'misdisinfo.com'
    ],
    # Known Unreliable Sources
    'unreliable': [
        'fakenews', 'hoax', 'bin', 'viral', 'WhatsApp',
        'facebook.com', 'twitter.com', 'instagram.com',
        'whatsapp', 'forward', 'share', 'viral message'
    ]
}

# --- FACT CHECK FUNCTION ---
def fact_check_news(query):
    """
    Real-time fact checking using Google News and source analysis
    Returns: dict with fact-check results
    """
    try:
        results = {
            'query': query,
            'verdict': 'UNKNOWN',
            'confidence': 0,
            'sources_found': [],
            'fact_check_links': [],
            'reliability_score': 0
        }
        
        # 1. Search Google News for the query
        search_url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(search_url)
        
        reliable_count = 0
        medium_count = 0
        unreliable_count = 0
        source_list = []
        
        for entry in feed.entries[:15]:
            source = entry.source.title if hasattr(entry, 'source') else 'Unknown'
            title = entry.title
            link = entry.link
            source_lower = source.lower()
            
            source_list.append({
                'title': title,
                'source': source,
                'link': link
            })
            
            # Check reliability
            is_reliable = False
            for rel_source in RELIABLE_SOURCES['reliable']:
                if rel_source in source_lower:
                    reliable_count += 1
                    is_reliable = True
                    break
            
            if not is_reliable:
                for med_source in RELIABLE_SOURCES['medium']:
                    if med_source in source_lower:
                        medium_count += 1
                        break
            
            # Check for fact-check sites
            for fc_source in RELIABLE_SOURCES['factcheck']:
                if fc_source in source_lower:
                    results['fact_check_links'].append({
                        'source': source,
                        'link': link,
                        'title': title
                    })
        
        results['sources_found'] = source_list
        
        # 2. Calculate reliability score
        total_sources = reliable_count + medium_count + unreliable_count
        if total_sources > 0:
            results['reliability_score'] = (reliable_count * 100 + medium_count * 50) / total_sources
        
        # 3. Determine verdict based on sources
        if reliable_count >= 3:
            results['verdict'] = 'LIKELY REAL'
            results['confidence'] = min(80 + reliable_count * 5, 95)
        elif reliable_count >= 1 and medium_count >= 2:
            results['verdict'] = 'LIKELY REAL'
            results['confidence'] = min(60 + reliable_count * 10, 85)
        elif results['fact_check_links']:
            # Check if fact-check sites have articles about this
            for fc in results['fact_check_links']:
                fc_title = fc['title'].lower()
                if any(word in fc_title for word in ['fake', 'false', 'hoax', 'misleading']):
                    results['verdict'] = 'LIKELY FAKE'
                    results['confidence'] = 75
                    break
                elif any(word in fc_title for word in ['true', 'real', 'accurate', 'fact check']):
                    results['verdict'] = 'CONFIRMED REAL'
                    results['confidence'] = 85
                    break
        elif medium_count >= 3:
            results['verdict'] = 'POSSIBLY REAL'
            results['confidence'] = 55
        else:
            results['verdict'] = 'CANNOT VERIFY'
            results['confidence'] = 30
        
        return results
        
    except Exception as e:
        print(f"Fact check error: {e}")
        return {
            'query': query,
            'verdict': 'ERROR',
            'confidence': 0,
            'sources_found': [],
            'fact_check_links': [],
            'reliability_score': 0,
            'error': str(e)
        }

# --- LOAD MODELS SAFELY (ORIGINAL + INDIA) ---
models = {
    'original': {'model': None, 'vectorizer': None, 'status': 'NOT_FOUND'},
    'india': {'model': None, 'vectorizer': None, 'status': 'NOT_FOUND'}
}
model_error = None

def load_model(name, model_path, vec_path):
    """Helper function to load a model and its vectorizer."""
    global model_error
    try:
        if os.path.exists(model_path) and os.path.exists(vec_path):
            with open(model_path, 'rb') as f_model, open(vec_path, 'rb') as f_vec:
                models[name]['model'] = pickle.load(f_model)
                models[name]['vectorizer'] = pickle.load(f_vec)
            models[name]['status'] = 'READY'
            print(f"[OK] Model '{name}' loaded successfully from {model_path}.")
            return True
        else:
            models[name]['status'] = 'NOT_FOUND'
            print(f"[!] Model '{name}' not found. Searched for {model_path} and {vec_path}.")
            return False
    except Exception as e:
        models[name]['status'] = 'ERROR'
        model_error = str(e)
        print(f"[ERROR] ERROR loading model '{name}': {e}")
        return False

# Load original model (default)
load_model('original', 'pac.pkl', 'vectorizer.pkl')
# Load India-specific model (assumes these filenames from train_india_model.py)
load_model('india', 'pac_india.pkl', 'vectorizer_india.pkl')

model_status = "READY" if any(m['status'] == 'READY' for m in models.values()) else "ERROR"

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
    try:
        if not current_rumors or datetime.now() > next_update_time:
            current_rumors = random.sample(RUMOR_POOL, min(3, len(RUMOR_POOL)))
            next_update_time = datetime.now() + timedelta(hours=2)
    except Exception as e:
        print(f"Error updating rumors: {e}")
        current_rumors = RUMOR_POOL[:3] # Fallback

def fetch_news_rss(query, lang='en'):
    try:
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
    except Exception as e:
        print(f"RSS Error: {e}")
        return []

# --- ROUTES ---
@app.route('/')
def home():
    try:
        update_rumors_if_needed()
        news = fetch_news_rss("Tamil Nadu", lang='ta')
        
        # Model limitations
        limitations = """
        MODEL LIMITATIONS & DATASET INFO:
        - Original Model: Trained on US news (True.csv, Fake.csv) - 44,898 articles
        - India Model: Trained on combined US + India datasets - 45,048 articles
        - Best for: English-language news articles
        - Limitations: Works better with longer articles (150+ words), US-centric training
        - Accuracy: ~99% on original dataset, varies on domain-specific news
        """
        
        return render_template('home.html', 
                             news_data=news, 
                             current_feed="Tamil News", 
                             rumors=current_rumors,
                             model_status=model_status,
                             model_error=model_error,
                             limitations=limitations)
    except Exception as e:
        return f"App Error: {str(e)}"

@app.route('/get_feed', methods=['POST'])
def get_feed():
    try:
        feed_type = request.json.get('type')
        if feed_type == 'tamil':
            news = fetch_news_rss("Tamil Nadu", lang='ta')
        else:
            news = fetch_news_rss("India", lang='en')
        return jsonify({'status': 'success', 'data': news})
    except Exception as e:
        print(f"Feed Error: {e}")
        return jsonify({'status': 'error', 'data': [], 'message': 'Failed to fetch news feed'})

@app.route('/search_news', methods=['POST'])
def search_news():
    try:
        update_rumors_if_needed()
        raw_query = request.form.get('query', '')
        lang_mode = request.form.get('search_lang')
        
        search_query = raw_query
        if lang_mode == 'tanglish':
            try:
                search_query = GoogleTranslator(source='auto', target='en').translate(raw_query)
            except Exception as e:
                print(f"Translation failed: {e}")
                
        news = fetch_news_rss(search_query, lang='en')
        return render_template('home.html', 
                             news_data=news, 
                             current_feed=f"Results: {raw_query}", 
                             search_val=raw_query, 
                             rumors=current_rumors,
                             model_status=model_status)
    except Exception as e:
        return render_template('home.html', 
                             news_data=[], 
                             error="Search failed.",
                             model_status=model_status)

@app.route('/api/model-status', methods=['GET'])
def get_model_status():
    """API endpoint to check status of all available models."""
    return jsonify({
        'overall_status': 'READY' if any(m['status'] == 'READY' for m in models.values()) else 'ERROR',
        'models': {name: {'status': info['status']} for name, info in models.items()},
        'error': model_error
    })


@app.route('/api/analyze', methods=['POST'])
def analyze_hybrid():
    """
    Hybrid Analysis API: Combines ML prediction with real-time fact-checking.
    INPUT: {"text": "...", "model": "original" | "india"}
    OUTPUT: Comprehensive analysis JSON
    """
    data = request.get_json() or {}
    text = data.get('text', '').strip()
    model_choice = data.get('model', 'original') # Default to original model

    if len(text) < 15:
        return jsonify({'error': 'Input text is too short. Please provide at least 15 characters.'}), 400

    # --- Step 1: ML Content Analysis ---
    ml_result = {'verdict': 'N/A', 'confidence': 0, 'model_used': model_choice}
    
    selected_model_info = models.get(model_choice)
    if not selected_model_info or selected_model_info['status'] != 'READY':
        # Fallback to original if chosen one is unavailable
        if model_choice != 'original' and models['original']['status'] == 'READY':
            selected_model_info = models['original']
            ml_result['model_used'] = 'original (fallback)'
        else:
            return jsonify({'error': f"Model '{model_choice}' is not available. No models ready."}), 503

    try:
        model = selected_model_info['model']
        vectorizer = selected_model_info['vectorizer']
        vec_text = vectorizer.transform([text])
        ml_result['verdict'] = model.predict(vec_text)[0]
        decision_score = model.decision_function(vec_text)[0]
        ml_result['confidence'] = min(abs(decision_score) * 10, 99)
    except Exception as e:
        ml_result['error'] = str(e)

    # --- Step 2: Real-time Source Analysis ---
    source_result = fact_check_news(text)

    # --- Step 3: Combine Results (Hybrid Logic) ---
    final_verdict = "UNCERTAIN"
    final_confidence = 50
    reasoning = []

    ml_verdict = ml_result.get('verdict')
    ml_conf = ml_result.get('confidence', 0)
    source_verdict = source_result.get('verdict')
    source_conf = source_result.get('confidence', 0)

    # High-confidence agreements
    if ml_verdict == 'FAKE' and ml_conf > 70 and source_verdict in ['LIKELY FAKE', 'CANNOT VERIFY']:
        final_verdict = 'LIKELY FAKE'
        final_confidence = (ml_conf + source_conf) / 2 + 10
        reasoning.append("ML model predicts FAKE with high confidence and source analysis concurs or cannot find reliable sources.")
    elif ml_verdict == 'REAL' and ml_conf > 70 and source_verdict in ['LIKELY REAL', 'CONFIRMED REAL']:
        final_verdict = 'LIKELY REAL'
        final_confidence = (ml_conf + source_conf) / 2 + 10
        reasoning.append("ML model predicts REAL and source analysis found multiple reliable sources.")
    
    # Conflicts
    elif ml_verdict == 'FAKE' and source_verdict in ['LIKELY REAL', 'CONFIRMED REAL']:
        final_verdict = 'UNCERTAIN'
        final_confidence = 40
        reasoning.append("CONFLICT: ML model suggests FAKE, but real-time search found reliable sources covering the topic. The news might be real but written in a way that triggers the model.")
    elif ml_verdict == 'REAL' and source_verdict == 'LIKELY FAKE':
        final_verdict = 'UNCERTAIN'
        final_confidence = 45
        reasoning.append("CONFLICT: ML model suggests REAL, but real-time search found fact-checks or unreliable sources indicating it might be fake.")

    # Low-confidence or one-sided evidence
    elif ml_verdict == 'FAKE' and ml_conf > 60:
        final_verdict = 'LIKELY FAKE'
        final_confidence = ml_conf
        reasoning.append("ML model predicts FAKE. Source analysis was inconclusive.")
    elif source_verdict == 'LIKELY FAKE':
        final_verdict = 'LIKELY FAKE'
        final_confidence = source_conf
        reasoning.append("Source analysis found evidence of it being fake (e.g., fact-checks). ML analysis was inconclusive.")
    elif source_verdict in ['LIKELY REAL', 'CONFIRMED REAL']:
        final_verdict = 'LIKELY REAL'
        final_confidence = source_conf
        reasoning.append("Source analysis found reliable sources. ML analysis was inconclusive.")
    else:
        reasoning.append("Neither the ML model nor the source analysis could provide a confident verdict. Treat with caution.")

    final_confidence = min(max(round(final_confidence, 2), 5), 99)

    return jsonify({
        'final_verdict': final_verdict,
        'final_confidence': final_confidence,
        'reasoning': ' '.join(reasoning),
        'ml_analysis': ml_result,
        'source_analysis': source_result
    })


@app.route('/api/ask', methods=['POST'])
def ask_question():
    """
    Conversational QA endpoint.
    INPUT: {"query": "is the indian prime minister right?"}
    OUTPUT: {"verdict": "Yes/No", "explanation": "...", "confidence": 95.0}
    """
    data = request.get_json() or {}
    query = data.get('query', '').strip()

    if not query or len(query) < 5:
        return jsonify({'error': 'Question is too short. Please ask a full question.'}), 400

    # Get answer from QA engine based on Indian database
    answer = get_qa_answer(query)
    
    return jsonify(answer)

@app.route('/api/collect', methods=['POST'])
def api_collect():
    """Trigger news collection and dataset update. Returns counts."""
    try:
        added = collect_news()
        csv_path = build_combined_csv()
        return jsonify({'status':'ok','added':added,'combined_csv': csv_path})
    except Exception as e:
        return jsonify({'status':'error','error':str(e)}), 500

@app.route('/api/predict_quick', methods=['POST'])
def predict_quick():
    try:
        data = request.get_json() or {}
        text = data.get('text', '').strip()
        if not text:
            return jsonify({'prediction': 'UNCERTAIN'})
            
        word_count = len(text.split())
        if word_count < 3:
            return jsonify({
                'prediction': 'UNCERTAIN',
                'explanation': 'Your input is too short. Machine Learning models require at least a full sentence or a few words of context to make an accurate prediction.',
                'confidence': 0,
                'source_title': 'Input Too Short'
            })
            
        answer = get_qa_answer(text)
        
        if answer['verdict'] == 'Yes':
            prediction = 'REAL'
        elif answer['verdict'] == 'No':
            prediction = 'FAKE'
        else:
            prediction = 'UNCERTAIN'
            
        return jsonify({
            'prediction': prediction,
            'confidence': answer.get('confidence', 50.0),
            'explanation': answer.get('explanation', 'No explanation available.'),
            'source_title': answer.get('source_title', 'Unknown Source')
        })
    except Exception as e:
        print(f"Quick Predict Error: {e}")
        return jsonify({
            'prediction': 'UNCERTAIN',
            'explanation': f'An error occurred: {e}',
            'confidence': 0,
            'source_title': 'Error'
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        update_rumors_if_needed()
        if request.method == 'POST':
            news_text = request.form.get('news', '').strip()
            
            if not news_text:
                news = fetch_news_rss("Tamil Nadu", lang='ta')
                return render_template('home.html', 
                                     news_data=news, 
                                     rumors=current_rumors,
                                     model_status=model_status, 
                                     error="⚠️ Please enter some text to analyze.")

            language = request.form.get('language')            
            final_text = news_text
            translated = False
            
            if language == 'tanglish':
                try:
                    final_text = GoogleTranslator(source='auto', target='en').translate(news_text)
                    translated = True
                except Exception as e:
                    print(f"Translation Error: {e}")
                    final_text = news_text

            # --- USE NEW HYBRID QA ENGINE INSTEAD OF LEGACY ML ---
            answer = get_qa_answer(final_text)
            
            # Map QA Engine 'Yes/No' back to 'REAL/FAKE' for the legacy result.html layout
            if answer['verdict'] == 'Yes':
                prediction = 'REAL'
            elif answer['verdict'] == 'No':
                prediction = 'FAKE'
            else:
                prediction = 'UNCERTAIN'
                
            confidence = answer.get('confidence', 50.0)
            explanation = answer.get('explanation', 'No explanation available.')
            source_title = answer.get('source_title', 'Unknown Source')

            return render_template('result.html', 
                                 prediction=prediction, 
                                 original=news_text, 
                                 translated=final_text,
                                 was_translated=translated,
                                 confidence=round(confidence, 2),
                                 explanation=explanation,
                                 source_title=source_title)
            
    except Exception as e:
        print(f"Prediction Error: {e}")
        news = fetch_news_rss("Tamil Nadu", lang='ta')
        return render_template('home.html', 
                             news_data=news, 
                             rumors=current_rumors,
                             model_status=model_status,
                             error="[ERROR] Analysis failed. Please try again.")

if __name__ == '__main__':
    app.run(debug=True)