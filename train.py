import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

print("--- STARTING TRAINING ---")

# 1. LOAD DATA
try:
    print("Loading True.csv and Fake.csv...")
    df_true = pd.read_csv('True.csv')
    df_fake = pd.read_csv('Fake.csv')
    
    df_true['label'] = 'REAL'
    df_fake['label'] = 'FAKE'
    
    df = pd.concat([df_true, df_fake])
    df = df.sample(frac=1).reset_index(drop=True)
    print(f"Loaded {len(df)} articles.")
except:
    print("ERROR: Could not find True.csv or Fake.csv in this folder.")
    print("Please download them from Kaggle!")
    exit()

# 2. TRAIN (With limits to keep file size small)
print("Training model...")
# max_features=10000 keeps the brain size under 50MB so GitHub accepts it
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, max_features=10000)

tfidf_train = tfidf_vectorizer.fit_transform(df['text'])
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, df['label'])

# 3. SAVE
print("Saving brain files...")
pickle.dump(pac, open('pac.pkl', 'wb'))
pickle.dump(tfidf_vectorizer, open('vectorizer.pkl', 'wb'))

print("SUCCESS! New 'pac.pkl' and 'vectorizer.pkl' created.")
print("CHECK: Your 'pac.pkl' should now be around 10MB - 50MB size.")