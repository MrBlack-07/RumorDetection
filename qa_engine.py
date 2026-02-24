import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import feedparser
import requests
from urllib.parse import quote_plus
import requests
import json

# Create an instance of the QA engine
class QAEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.df = None
        self.tfidf_matrix = None
        self.is_ready = False
        self.load_data()

    def load_data(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        true_path = os.path.join(base_dir, 'True_India.csv')
        fake_path = os.path.join(base_dir, 'Fake_India.csv')
        
        # Load comprehensive if they exist, else primary
        true_comp_path = os.path.join(base_dir, 'True_India_Comprehensive.csv')
        fake_comp_path = os.path.join(base_dir, 'Fake_India_Comprehensive.csv')
        
        if os.path.exists(true_comp_path):
            true_path = true_comp_path
        if os.path.exists(fake_comp_path):
            fake_path = fake_comp_path

        try:
            if not os.path.exists(true_path) or not os.path.exists(fake_path):
                print("Warning: Indian dataset files not found. QA engine will be disabled.")
                return

            df_true = pd.read_csv(true_path)
            df_fake = pd.read_csv(fake_path)

            df_true['label'] = 'REAL'
            df_fake['label'] = 'FAKE'

            # Combine and clean
            self.df = pd.concat([df_true, df_fake], ignore_index=True)
            self.df = self.df.dropna(subset=['text', 'title'])
            
            # Create a combined text field for better search
            self.df['search_text'] = self.df['title'] + " " + self.df['text']
            
            # Fit and transform
            self.tfidf_matrix = self.vectorizer.fit_transform(self.df['search_text'])
            self.is_ready = True
            print(f"[OK] QA Engine initialized with {len(self.df)} Indian articles.")
        except Exception as e:
            print(f"Error initializing QA Engine: {e}")

    def answer_question(self, query):
        if not self.is_ready:
            return {
                "verdict": "ERROR",
                "explanation": "QA database is not available."
            }

        # Vectorize query
        query_vec = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        if len(similarities) == 0:
             return {
                "verdict": "UNKNOWN",
                "explanation": "No relevant information found in the database."
            }

        best_idx = similarities.argmax()
        best_score = similarities[best_idx]
        
        # If confidence is too low, fallback to WIKIPEDIA then LIVE SEARCH
        # 0.50 threshold ensures only highly confident contextual matches use local data
        if best_score < 0.50:
            return self.wikipedia_fallback(query)

        best_article = self.df.iloc[best_idx]
        label = best_article['label']
        title = best_article['title']
        text = str(best_article['text'])

        # Extract first 2-3 sentences for explanation
        # Split by typical sentence boundaries
        sentences = [s.strip() for s in text.replace('?', '.').replace('!', '.').split('.') if len(s.strip()) > 10]
        context_sentences = sentences[:3]
        context = ". ".join(context_sentences) + "."
        
        query_lower = f" {query.lower()} "
        negations = [' not ', " isn't ", " aren't ", " wasn't ", " weren't ", ' never ', ' fake ', ' false ']
        has_negation = any(neg in query_lower for neg in negations)
        
        if label == 'REAL':
            if has_negation:
                verdict = "No"
                explanation = f"Your statement contains a negation, but our verified database confirms the opposite: {context}"
            else:
                verdict = "Yes"
                explanation = f"{context}"
        else:
            if has_negation:
                verdict = "Yes"
                explanation = f"Correct. The underlying claim is recognized as fake or misleading in our database. {context}"
            else:
                verdict = "No"
                explanation = f"This information is recognized as fake or misleading in our database. {context}"

        return {
            "verdict": verdict,
            "explanation": explanation,
            "source_title": title + " (Local DB)",
            "confidence": round(float(best_score) * 100, 2)
        }
        
        

    def wikipedia_fallback(self, query):
        """Search Wikipedia first for factual/timeless queries before hitting Live News."""
        
        # Step 0: Check for breaking news keywords. If it's time-sensitive, Wikipedia won't help.
        query_words = query.lower().split()
        breaking_keywords = ['today', 'yesterday', 'now', 'breaking', 'latest', 'update', 'recently', 'current']
        if any(kw in query_words for kw in breaking_keywords):
            return self.live_search_fallback(query)
            
        try:
            # Step 1: Search Wikipedia for the closest article title
            search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={quote_plus(query)}&utf8=&format=json"
            headers = {'User-Agent': 'TruthLensAI/1.0'}
            resp = requests.get(search_url, headers=headers, timeout=5).json()
            
            search_results = resp.get('query', {}).get('search', [])
            if not search_results:
                return self.live_search_fallback(query)
                
            # Get the top hit title
            top_title = search_results[0]['title']
            
            # Step 2: Fetch the summary of that article
            summary_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro=1&explaintext=1&titles={quote_plus(top_title)}&format=json"
            summary_resp = requests.get(summary_url, headers=headers, timeout=5).json()
            
            pages = summary_resp.get('query', {}).get('pages', {})
            page = list(pages.values())[0]
            extract = page.get('extract', '').strip()
            
            if not extract or len(extract) < 50:
                return self.live_search_fallback(query)
                
            # Step 3: We found a solid Wikipedia article! Let's do NLP vector math to see if it matches the query's claim.
            # Use NLP to check if the Wikipedia extract actually addresses the specific question.
            try:
                temp_vec = TfidfVectorizer(stop_words='english')
                vecs = temp_vec.fit_transform([query, extract])
                sim = cosine_similarity(vecs[0:1], vecs[1:2]).flatten()[0]
                
                # If similarity is too low, Wikipedia just defines the entity but doesn't answer the specific claim.
                if sim < 0.10:
                    return self.live_search_fallback(query)
            except Exception as ml_err:
                print(f"Wikipedia similarity error: {ml_err}")
                
            # Clean up the query and extract
            query_lower = f" {query.lower()} "
            negations = [' not ', " isn't ", " aren't ", " wasn't ", " weren't ", ' never ', ' fake ', ' false ']
            has_negation = any(neg in query_lower for neg in negations)
            
            sentences = [s.strip() for s in extract.replace('?', '.').replace('!', '.').split('.') if len(s.strip()) > 10][:3]
            context = ". ".join(sentences) + "."
            
            # Since Wikipedia is factual, if there's no negation in the query, the fact is generally TRUE.
            # If the query contains a negation (e.g., "The earth is NOT round"), then it contradicts the factual Wikipedia text.
            if has_negation:
                verdict = "No"
                explanation = f"Your statement contradicts known facts. According to Wikipedia: {context}"
                confidence = 80.0
            else:
                verdict = "Yes"
                explanation = f"Factually accurate. According to Wikipedia: {context}"
                confidence = 85.0
                
            return {
                "verdict": verdict,
                "explanation": explanation,
                "source_title": top_title + " (Wikipedia)",
                "confidence": confidence
            }
            
        except Exception as e:
            print(f"Wikipedia API error: {e}")
            return self.live_search_fallback(query)

    def live_search_fallback(self, query):
        """Fallback to Google News RSS if local database lacks confidence."""
        try:
            search_url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-IN&gl=IN&ceid=IN:en"
            feed = feedparser.parse(search_url)
            
            if not feed.entries:
                return {
                    "verdict": "UNKNOWN",
                    "explanation": "I couldn't find relevant Indian news in my database OR online to confidently answer this question. It might be too obscure or highly specific.",
                    "source_title": "No Sources Found",
                    "confidence": 0
                }
                
            # Smart NLP Ranking for Top 5 Live News
            top_entries = feed.entries[:5]
            best_entry = top_entries[0]
            
            try:
                descriptions = [getattr(entry, 'description', getattr(entry, 'summary', entry.title)) for entry in top_entries]
                texts = [query] + [f"{e.title} {d}" for e, d in zip(top_entries, descriptions)]
                
                # Fresh vectorizer for isolated comparison
                temp_vec = TfidfVectorizer(stop_words='english')
                vecs = temp_vec.fit_transform(texts)
                sims = cosine_similarity(vecs[0:1], vecs[1:]).flatten()
                
                max_idx = sims.argmax()
                best_entry = top_entries[max_idx]
            except Exception as e:
                print(f"Live Search NLP Error: {e}")
                
            title = best_entry.title
            source = best_entry.source.title if hasattr(best_entry, 'source') else 'Google News'
            
            # Simple heuristic for live search
            source_lower = source.lower()
            title_lower = title.lower()
            
            fact_checkers = ['altnews', 'boom', 'factly', 'factcheck', 'snopes', 'pib fact check']
            reliable = ['hindu', 'ndtv', 'times of india', 'indian express', 'pib', 'reuters', 'bbc', 'ani', 'pti']
            
            is_fact_check = any(fc in source_lower for fc in fact_checkers) or any(word in title_lower for word in ['fact check', 'fake', 'hoax', 'busted'])
            is_reliable = any(r in source_lower for r in reliable)
            
            if is_fact_check:
                verdict = "No"
                explanation = f"Fact-checkers have recently addressed this. According to {source}, the truth is: {title}."
                confidence = 85.0
            elif is_reliable:
                verdict = "Yes"
                explanation = f"According to real-time reports from reliable sources like {source}, the latest news is: {title}."
                confidence = 80.0
            else:
                verdict = "Yes" # Default to yes for general news, but with lower confidence
                explanation = f"I found recent news matching your query online. According to {source}: {title}."
                confidence = 65.0
                
            return {
                "verdict": verdict,
                "explanation": explanation,
                "source_title": title + " (Live Online Search)",
                "confidence": confidence
            }
            
        except Exception as e:
            return {
                "verdict": "ERROR",
                "explanation": f"Live search failed: {e}",
                "source_title": "Error",
                "confidence": 0
            }

# Global instance
qa_system = QAEngine()

def get_qa_answer(query):
    return qa_system.answer_question(query)

if __name__ == '__main__':
    # Test
    res = get_qa_answer("indian prime minister is right?")
    print(res)
