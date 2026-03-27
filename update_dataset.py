import os
import json
import pandas as pd
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import pickle

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
RAW_FILE = os.path.join(DATA_DIR, 'news_raw.jsonl')
COMBINED_CSV = os.path.join(DATA_DIR, 'combined_recent.csv')

RELIABLE_DOMAINS = set(['pib.gov.in','ndtv.com','thehindu.com','indianexpress.com','timesofindia.indiatimes.com','hindustantimes.com','reuters.com','bbc.com'])
UNRELIABLE_KEYWORDS = set(['viral','whatsapp','facebook','instagram','forward'])

def _get_domain(link):
    try:
        return urlparse(link).netloc.lower()
    except Exception:
        return ''

def load_raw_items():
    items = []
    if os.path.exists(RAW_FILE):
        with open(RAW_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    items.append(json.loads(line))
                except Exception:
                    continue
    return items

def pseudo_label_items(items, model=None, vectorizer=None):
    rows = []
    for it in items:
        title = it.get('title','')
        link = it.get('link','')
        domain = _get_domain(link)
        label = 'UNKNOWN'
        # Heuristic labeling
        if any(d in domain for d in RELIABLE_DOMAINS):
            label = 'REAL'
        elif any(k in title.lower() for k in UNRELIABLE_KEYWORDS):
            label = 'FAKE'
        elif model and vectorizer:
            try:
                X = vectorizer.transform([title])
                pred = model.predict(X)[0]
                label = pred
            except Exception:
                label = 'UNKNOWN'
        rows.append({'text': title, 'label': label, 'link': link, 'domain': domain})
    return pd.DataFrame(rows)

def build_combined_csv():
    items = load_raw_items()
    model = None
    vec = None
    # Load existing vectorizer and model if available
    try:
        vec = pickle.load(open(os.path.join(BASE_DIR, 'models', 'vectorizer.pkl'),'rb'))
        model = pickle.load(open(os.path.join(BASE_DIR, 'models', 'pac.pkl'),'rb'))
    except Exception:
        vec = None
        model = None

    df_recent = pseudo_label_items(items, model=model, vectorizer=vec)
    # Filter unknowns
    df_recent = df_recent[df_recent['label']!='UNKNOWN']
    if df_recent.empty:
        print('No recent pseudo-labeled items to add.')
        return None

    # Load existing True/Fake datasets if present
    frames = [df_recent[['text','label']]]
    for fname in ['True.csv','Fake.csv','True_India_Comprehensive.csv','Fake_India_Comprehensive.csv']:
        filepath = os.path.join(BASE_DIR, 'datasets', fname)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                if 'title' in df.columns and 'text' in df.columns:
                    df['text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
                if 'text' in df.columns and 'label' in df.columns:
                    frames.append(df[['text','label']])
            except Exception:
                continue

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=['text'])
    combined.to_csv(COMBINED_CSV, index=False)
    print(f'Wrote combined dataset to {COMBINED_CSV} ({len(combined)} rows)')
    return COMBINED_CSV

if __name__ == '__main__':
    build_combined_csv()
