import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

# --- 1. DATA PREPARATION (For Dataset Option 2) ---
print("Loading dataset...")

try:
    # Read the two separate files
    df_true = pd.read_csv('True.csv')
    df_fake = pd.read_csv('Fake.csv')
    
    # Add labels (1 for Real, 0 for Fake) or text labels
    df_true['label'] = 'REAL'
    df_fake['label'] = 'FAKE'
    
    # Merge them into one dataset
    df = pd.concat([df_true, df_fake])
    
    # Shuffle the data (so it's not all Real then all Fake)
    df = df.sample(frac=1).reset_index(drop=True)
    
    print(f"Success! Loaded {len(df)} articles.")
    
except FileNotFoundError:
    print("ERROR: Could not find 'True.csv' or 'Fake.csv'.")
    print("Please download them from Kaggle and put them in this folder.")
    exit()

# --- 2. TRAINING ---
print("Training model (this may take a minute)...")

# Initialize Vectorizer
# We use 'text' column because this dataset has full article text
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Convert text to numbers
tfidf_train = tfidf_vectorizer.fit_transform(df['text'])

# Initialize Passive Aggressive Classifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, df['label'])

# --- 3. SAVE THE MODEL ---
pickle.dump(pac, open('pac.pkl', 'wb'))
pickle.dump(tfidf_vectorizer, open('vectorizer.pkl', 'wb'))

print("DONE! 'pac.pkl' and 'vectorizer.pkl' have been saved.")
print("Now run 'python app.py' to start your site.")