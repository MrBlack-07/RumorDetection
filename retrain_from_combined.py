import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

BASE = os.path.dirname(__file__)
COMBINED = os.path.join(BASE, 'data', 'combined_recent.csv')

def load_data():
    if os.path.exists(COMBINED):
        df = pd.read_csv(COMBINED)
        if 'label' in df.columns and 'text' in df.columns:
            return df['text'], df['label']
    # fallback to original
    df_true = pd.read_csv(os.path.join(BASE, 'datasets', 'True.csv')) if os.path.exists(os.path.join(BASE, 'datasets', 'True.csv')) else None
    df_fake = pd.read_csv(os.path.join(BASE, 'datasets', 'Fake.csv')) if os.path.exists(os.path.join(BASE, 'datasets', 'Fake.csv')) else None
    frames = []
    if df_true is not None:
        df_true['text'] = df_true.get('title','').fillna('') + ' ' + df_true.get('text','').fillna('')
        df_true['label'] = 'REAL'
        frames.append(df_true[['text','label']])
    if df_fake is not None:
        df_fake['text'] = df_fake.get('title','').fillna('') + ' ' + df_fake.get('text','').fillna('')
        df_fake['label'] = 'FAKE'
        frames.append(df_fake[['text','label']])
    if frames:
        df = pd.concat(frames, ignore_index=True).dropna()
        return df['text'], df['label']
    raise RuntimeError('No training data found')

def train_and_save():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    vec = TfidfVectorizer(strip_accents='unicode', ngram_range=(1,2), max_features=15000, stop_words='english', sublinear_tf=True)
    X_train_tfidf = vec.fit_transform(X_train)
    X_test_tfidf = vec.transform(X_test)
    clf = SGDClassifier(loss='hinge', penalty=None, random_state=42, max_iter=1000, tol=1e-3)
    clf.fit(X_train_tfidf, y_train)
    pred = clf.predict(X_test_tfidf)
    print('Accuracy:', accuracy_score(y_test, pred))
    print('Precision:', precision_score(y_test, pred, pos_label='FAKE'))
    print('Recall:', recall_score(y_test, pred, pos_label='FAKE'))
    print('F1:', f1_score(y_test, pred, pos_label='FAKE'))
    pickle.dump(clf, open(os.path.join(BASE, 'models', 'pac.pkl'),'wb'))
    pickle.dump(vec, open(os.path.join(BASE, 'models', 'vectorizer.pkl'),'wb'))
    print('Saved models/pac.pkl and models/vectorizer.pkl')

if __name__ == '__main__':
    train_and_save()
