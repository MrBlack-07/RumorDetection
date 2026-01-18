import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import pickle

# --- 1. DATA PREPARATION ---
# For a real project, download 'news.csv' from Kaggle. 
# For this demo, we use a small internal dataset to make it run INSTANTLY.
data = {
    'text': [
        "Aliens have landed in Chennai and are demanding coffee.",
        "Government announces new tax reforms for small businesses.",
        "Drinking water cures all diseases instantly according to new study.",
        "The Prime Minister inaugurated the new metro line today.",
        "Scientists discover a tree that grows gold coins.",
        "RBI keeps repo rate unchanged in latest monetary policy.",
        "NASA confirms earth is actually flat and rests on a turtle.",
        "Election commission announces dates for upcoming assembly polls."
    ],
    'label': [
        "FAKE", "REAL", "FAKE", "REAL", "FAKE", "REAL", "FAKE", "REAL"
    ]
}

df = pd.DataFrame(data)

# --- 2. TRAINING ---
# Initialize Vectorizer (Stop words removes 'the', 'is', etc.)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Convert text to numbers
tfidf_train = tfidf_vectorizer.fit_transform(df['text'])

# Initialize Passive Aggressive Classifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, df['label'])

# --- 3. SAVE THE MODEL ---
pickle.dump(pac, open('pac.pkl', 'wb'))
pickle.dump(tfidf_vectorizer, open('vectorizer.pkl', 'wb'))

print("Model trained successfully! 'pac.pkl' and 'vectorizer.pkl' created.")