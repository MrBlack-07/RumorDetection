import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import os

print("--- STARTING SMART TRAINING ---")

# 1. LOAD DATA (Crucial: You must have True.csv and Fake.csv)
try:
    print("Loading datasets...")
    df_true = pd.read_csv('True.csv')
    df_fake = pd.read_csv('Fake.csv')
    
    df_true['label'] = 'REAL'
    df_fake['label'] = 'FAKE'
    
    # Merge and Shuffle
    df = pd.concat([df_true, df_fake])
    df = df.sample(frac=1).reset_index(drop=True)
    print(f"Loaded {len(df)} articles. The AI is learning...")
except:
    print("ERROR: True.csv or Fake.csv not found!")
    print("Please download them from Kaggle and put them in this folder.")
    exit()

# 2. TRAIN WITH N-GRAMS (The "Word Arrangement" Fix)
print("Training brain...")

# ngram_range=(1,2) -> Teaches AI to read "Prime Minister" as a pair, not just separate words
# max_features=10000 -> Keeps the brain file small enough for GitHub (approx 50MB)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, max_features=10000, ngram_range=(1,2))

# Convert text to numbers
tfidf_train = tfidf_vectorizer.fit_transform(df['text'])

# Train the Classifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, df['label'])

# 3. SAVE
print("Saving smart brain files...")
pickle.dump(pac, open('pac.pkl', 'wb'))
pickle.dump(tfidf_vectorizer, open('vectorizer.pkl', 'wb'))

print("SUCCESS! Your AI can now handle different word arrangements.")