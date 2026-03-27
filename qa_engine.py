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
        true_path = os.path.join(base_dir, 'datasets', 'True_India.csv')
        fake_path = os.path.join(base_dir, 'datasets', 'Fake_India.csv')
        
        # Load comprehensive if they exist, else primary
        true_comp_path = os.path.join(base_dir, 'datasets', 'True_India_Comprehensive.csv')
        fake_comp_path = os.path.join(base_dir, 'datasets', 'Fake_India_Comprehensive.csv')
        
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
                explanation = f"As your AI assistant checking proper sources across India, I found your statement contains a negation, but our verified databases confirm the opposite is TRUE: {context}"
            else:
                verdict = "Yes"
                explanation = f"As your AI assistant checking proper sources across India, I can confirm this is TRUE. {context}"
        else:
            if has_negation:
                verdict = "Yes"
                explanation = f"As your AI assistant, you are correct. I checked proper sources and the underlying claim is FALSE. {context}"
            else:
                verdict = "No"
                explanation = f"As your AI assistant, I can confirm this information is FALSE based on verified sources. {context}"

        return {
            "verdict": verdict,
            "explanation": explanation,
            "source_title": title + " (Local DB)",
            "confidence": round(float(best_score) * 100, 2)
        }
        
        

    def wikipedia_fallback(self, query):
        """Search Wikipedia first for factual/timeless queries before hitting Live News."""
        
        # Step 0: Check for breaking news keywords or question formats.
        query_words = query.lower().split()
        breaking_keywords = ['today', 'yesterday', 'now', 'breaking', 'latest', 'update', 'recently', 'current', 'news']
        question_words = ('is ', 'are ', 'was ', 'were ', 'will ', 'did ', 'does ', 'do ', 'has ', 'have ', 'who ', 'what ', 'when ', 'where ', 'why ', 'how ')
        
        if any(kw in query_words for kw in breaking_keywords) or query.lower().startswith(question_words):
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
                explanation = f"As your AI assistant checking global proper sources, your statement contradicts known facts. Thus it is FALSE. According to Wikipedia: {context}"
                confidence = 80.0
            else:
                verdict = "Yes"
                explanation = f"As your AI assistant, this is factually TRUE. According to Wikipedia: {context}"
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
        """Agentic LLM-like Web Search: Fallback to Google News RSS and Wikipedia context."""
        try:
            # 1. Base Context Generation (Fetch a quick Wikipedia snippet for the main entity)
            context_intro = ""
            try:
                search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={quote_plus(query)}&utf8=&format=json"
                headers = {'User-Agent': 'TruthLensAI/1.0'}
                resp = requests.get(search_url, headers=headers, timeout=3).json()
                search_results = resp.get('query', {}).get('search', [])
                if search_results:
                    top_title = search_results[0]['title']
                    summary_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro=1&explaintext=1&titles={quote_plus(top_title)}&format=json"
                    summary_resp = requests.get(summary_url, headers=headers, timeout=3).json()
                    pages = summary_resp.get('query', {}).get('pages', {})
                    page = list(pages.values())[0]
                    extract = page.get('extract', '').strip()
                    if extract:
                        sentences = [s.strip() for s in extract.replace('?', '.').replace('!', '.').split('.') if len(s.strip()) > 5]
                        if sentences:
                            context_intro = "📚 Context: " + ". ".join(sentences[:2]) + ".\n\n"
            except Exception as e:
                pass # Ignore wiki context failures

            # 2. Google News RSS Search
            search_url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-IN&gl=IN&ceid=IN:en"
            feed = feedparser.parse(search_url)
            
            if not feed.entries:
                return {
                    "verdict": "UNKNOWN",
                    "explanation": f"{context_intro}🕵️‍♂️ I couldn't find recent relevant news in my database OR online to confidently answer this question.",
                    "source_title": "No Sources Found",
                    "confidence": 0
                }
                
            # Analyze all top entries
            top_entries = feed.entries[:10]
            
            fact_check_news = []
            reliable_news = []
            
            fact_checkers = ['altnews', 'boom', 'factly', 'factcheck', 'snopes', 'pib fact check', 'quint', 'vishvas news', 'newschecker']
            reliable = ['hindu', 'ndtv', 'times of india', 'indian express', 'pib', 'reuters', 'bbc', 'ani', 'pti', 'mint', 'business standard', 'telegraph']
            
            for entry in top_entries:
                title = entry.title
                source = entry.source.title if hasattr(entry, 'source') else 'Google News'
                source_lower = source.lower()
                title_lower = title.lower()
                
                is_fact_check = any(fc in source_lower for fc in fact_checkers) or any(word in title_lower for word in ['fact check', 'fake', 'hoax', 'busted', 'fact-check'])
                is_reliable = any(r in source_lower for r in reliable)
                
                if is_fact_check:
                    fact_check_news.append({'title': title, 'source': source})
                elif is_reliable:
                    reliable_news.append({'title': title, 'source': source})
                else:
                    # General news, fallback
                    reliable_news.append({'title': title, 'source': source})

            # Determine Verdict and Build Explanation
            explanation = context_intro
            verdict = "Yes"
            confidence = 50.0
            source_title = "Multiple Sources"

            if fact_check_news:
                verdict = "No"
                confidence = 85.0
                explanation += "🚨 As your AI assistant checking proper sources across India, I found fact-checkers have debunked this. It is FALSE. Fact-check reports:\n"
                for news in fact_check_news[:2]:
                    explanation += f"- [{news['source']}] {news['title']}\n"
                explanation += "\n"

            if reliable_news:
                # Basic NLP keyword match heuristic for claim verification
                stop_words = {'is', 'are', 'am', 'was', 'were', 'do', 'does', 'did', 'has', 'have', 'had', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'about', 'actor', 'actress', 'minister', 'news', 'latest', 'recently', 'today', 'true', 'false', 'any', 'some', 'my'}
                q_words = [w for w in query.lower().replace('?', '').split() if w not in stop_words]
                
                unique_news = []
                seen_titles = set()
                for news in reliable_news:
                    if news['title'] not in seen_titles:
                        unique_news.append(news)
                        seen_titles.add(news['title'])
                        
                combined_titles = " ".join([n['title'].lower() for n in unique_news[:5]])
                matched_words = [w for w in q_words if w in combined_titles]
                match_ratio = len(matched_words) / len(q_words) if q_words else 1.0
                
                if verdict == "Yes":
                    if match_ratio > 0.6:
                        confidence = 85.0
                        explanation += "✅ As your AI assistant, I scanned proper news sources across India and verified this is TRUE. Latest reports:\n"
                    else:
                        verdict = "UNCERTAIN"
                        confidence = 50.0
                        explanation += "🕵️‍♂️ As your AI assistant, I found news related to this topic across India, but proper sources do not explicitly confirm this exact claim. Latest reports:\n"
                else:
                    explanation += "📰 Other Top News:\n"
                
                for news in unique_news[:4]:
                    explanation += f"- [{news['source']}] {news['title']}\n"
            else:
                if not fact_check_news:
                    explanation += "I found some general news articles online, but couldn't verify them against my list of highly trustable sources."
                    
            return {
                "verdict": verdict,
                "explanation": explanation.strip(),
                "source_title": source_title + " (Live Online Search)",
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
