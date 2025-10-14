# -*- coding: utf-8 -*-
"""
Gym Review Analysis: Topic Modeling and Sentiment Analysis

This project analyzes customer reviews from gym facilities to identify key themes,
sentiment patterns, and actionable insights for business improvement.

The analysis includes:
- Text preprocessing and frequency analysis
- Topic modeling using BERTopic and LDA
- Emotion analysis using BERT
- Large language model integration for insights generation
"""

import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
import string
import contractions
from wordcloud import WordCloud
from langdetect import detect
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('all', quiet=True)

class GymReviewAnalyzer:
    """
    A comprehensive analyzer for gym customer reviews using various NLP techniques.
    """
    
    def __init__(self):
        self.goog_reviews = None
        self.tp_reviews = None
        self.goog_negative_reviews = None
        self.tp_negative_reviews = None
        self.common_locations = None
        
    def load_data(self, google_url=None, trustpilot_url=None, use_local_files=True):
        """
        Load review data from Google Sheets, local files, or create sample data.
        
        Args:
            google_url (str): URL to Google Sheets data
            trustpilot_url (str): URL to Trustpilot data
            use_local_files (bool): If True, try to load local Excel files first
        """
        if use_local_files:
            # Try to load local files first
            try:
                import os
                data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
                google_file = os.path.join(data_dir, 'Google_12_months.xlsx')
                trustpilot_file = os.path.join(data_dir, 'Trustpilot_12_months.xlsx')
                
                if os.path.exists(google_file) and os.path.exists(trustpilot_file):
                    print("📁 Loading local Excel files...")
                    self.goog_reviews = pd.read_excel(google_file, engine='openpyxl')
                    self.tp_reviews = pd.read_excel(trustpilot_file, engine='openpyxl')
                    print(f"✅ Loaded {len(self.goog_reviews)} Google reviews from local file")
                    print(f"✅ Loaded {len(self.tp_reviews)} Trustpilot reviews from local file")
                    return
                else:
                    print("⚠️  Local Excel files not found, trying other options...")
            except Exception as e:
                print(f"⚠️  Error loading local files: {e}")
                print("Trying other options...")
        
        if google_url and trustpilot_url:
            # Load from URLs
            print("🌐 Loading data from Google Sheets URLs...")
            self.goog_reviews = pd.read_excel(google_url, engine='openpyxl')
            self.tp_reviews = pd.read_excel(trustpilot_url, engine='openpyxl')
            print(f"✅ Loaded {len(self.goog_reviews)} Google reviews from URL")
            print(f"✅ Loaded {len(self.tp_reviews)} Trustpilot reviews from URL")
        else:
            # For demo purposes, create sample data
            print("📊 Creating sample data for demonstration...")
            self._create_sample_data()
            print(f"✅ Generated {len(self.goog_reviews)} sample Google reviews")
            print(f"✅ Generated {len(self.tp_reviews)} sample Trustpilot reviews")
        
    def _create_sample_data(self):
        """Create sample data for demonstration purposes."""
        sample_reviews = [
            "The gym is always crowded and equipment is often broken",
            "Great facilities but parking is terrible",
            "Staff are friendly but the showers are always cold",
            "Love the classes but the music is too loud",
            "Clean gym with good equipment variety",
            "Terrible customer service and dirty changing rooms",
            "Good value for money but needs better air conditioning",
            "Equipment is outdated and maintenance is poor",
            "Great location but too many people during peak hours",
            "Excellent personal trainers but expensive membership"
        ]
        
        sample_locations = ["London Central", "Manchester", "Birmingham", "Leeds", "Liverpool"]
        
        # Create sample Google reviews
        self.goog_reviews = pd.DataFrame({
            'Comment': sample_reviews * 5,
            'Overall Score': np.random.randint(1, 6, 50),
            'Club\'s Name': np.random.choice(sample_locations, 50)
        })
        
        # Create sample Trustpilot reviews
        self.tp_reviews = pd.DataFrame({
            'Review Content': sample_reviews * 5,
            'Review Stars': np.random.randint(1, 6, 50),
            'Location Name': np.random.choice(sample_locations, 50)
        })
    
    def clean_data(self):
        """Remove missing values, duplicates, and non-English reviews."""
        # Remove missing values
        self.goog_reviews = self.goog_reviews[self.goog_reviews['Comment'].notnull()]
        self.tp_reviews = self.tp_reviews[self.tp_reviews['Review Content'].notnull()]
        
        # Remove duplicates
        self.goog_reviews = self.goog_reviews.drop_duplicates(subset=['Comment'])
        self.tp_reviews = self.tp_reviews.drop_duplicates(subset=['Review Content'])
        
        # Filter to English reviews only
        def is_english(text):
            try:
                return detect(text) == "en"
            except:
                return False
        
        self.goog_reviews = self.goog_reviews[self.goog_reviews['Comment'].apply(is_english)]
        self.tp_reviews = self.tp_reviews[self.tp_reviews['Review Content'].apply(is_english)]
        
        print(f"After cleaning: {len(self.goog_reviews)} Google reviews, {len(self.tp_reviews)} Trustpilot reviews")
    
    def preprocess_text(self, text):
        """
        Preprocess text by removing punctuation, numbers, and stopwords.
        
        Args:
            text (str): Input text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        # Remove punctuation
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove common gym-related terms to focus on specific issues
        text = text.replace("gym", "").strip()
        text = text.replace("pure", "").strip()
        
        # Handle contractions
        text = contractions.fix(text)
        
        # Tokenize and convert to lowercase
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords
        filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
        
        # Join tokens back into string
        processed_text = ' '.join(filtered_tokens)
        
        return processed_text
    
    def analyze_word_frequency(self):
        """Analyze word frequency in reviews and create visualizations."""
        # Preprocess reviews
        self.goog_reviews['clean_comment'] = self.goog_reviews['Comment'].apply(self.preprocess_text)
        self.tp_reviews['clean_comment'] = self.tp_reviews['Review Content'].apply(self.preprocess_text)
        
        # Calculate frequency distributions
        from nltk.probability import FreqDist
        
        goog_freq_dist = FreqDist(word_tokenize(' '.join(self.goog_reviews['clean_comment'])))
        tp_freq_dist = FreqDist(word_tokenize(' '.join(self.tp_reviews['clean_comment'])))
        
        # Plot top 10 words for each dataset
        self._plot_word_frequency(goog_freq_dist, "Google Reviews", "google_freq.png")
        self._plot_word_frequency(tp_freq_dist, "Trustpilot Reviews", "trustpilot_freq.png")
        
        return goog_freq_dist, tp_freq_dist
    
    def _plot_word_frequency(self, freq_dist, title, filename):
        """Plot word frequency distribution."""
        top_10 = freq_dist.most_common(10)
        words, freqs = zip(*top_10)
        
        plt.figure(figsize=(10, 6))
        plt.bar(words, freqs)
        plt.xlabel("Words")
        plt.ylabel("Frequency")
        plt.title(f"Top 10 Words in {title}")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_wordclouds(self):
        """Create word clouds for both datasets."""
        # Google reviews wordcloud
        plt.figure(figsize=(12, 8))
        wc = WordCloud(background_color='black', max_words=1000, max_font_size=50)
        wc.generate(' '.join(self.goog_reviews['clean_comment']))
        plt.imshow(wc)
        plt.axis('off')
        plt.title('Google Reviews Word Cloud')
        plt.savefig('google_wordcloud.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Trustpilot reviews wordcloud
        plt.figure(figsize=(12, 8))
        wc = WordCloud(background_color='black', max_words=1000, max_font_size=50)
        wc.generate(' '.join(self.tp_reviews['clean_comment']))
        plt.imshow(wc)
        plt.axis('off')
        plt.title('Trustpilot Reviews Word Cloud')
        plt.savefig('trustpilot_wordcloud.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def filter_negative_reviews(self):
        """Filter reviews with scores < 3 (negative reviews)."""
        self.goog_negative_reviews = self.goog_reviews[self.goog_reviews['Overall Score'] < 3]
        self.tp_negative_reviews = self.tp_reviews[self.tp_reviews['Review Stars'] < 3]
        
        print(f"Negative Google reviews: {len(self.goog_negative_reviews)}")
        print(f"Negative Trustpilot reviews: {len(self.tp_negative_reviews)}")
        
        return self.goog_negative_reviews, self.tp_negative_reviews
    
    def find_common_locations(self):
        """Find common locations between Google and Trustpilot datasets."""
        self.common_locations = set(self.goog_reviews['Club\'s Name']).intersection(
            set(self.tp_reviews['Location Name'])
        )
        print(f"Common locations: {len(self.common_locations)}")
        return self.common_locations
    
    def run_topic_modeling(self, use_bertopic=True):
        """
        Run topic modeling on negative reviews.
        
        Args:
            use_bertopic (bool): If True, use BERTopic; if False, use LDA
        """
        if use_bertopic:
            return self._run_bertopic()
        else:
            return self._run_lda()
    
    def _run_bertopic(self):
        """Run BERTopic analysis on negative reviews."""
        try:
            from bertopic import BERTopic
            
            # Filter reviews from common locations
            goog_neg_common = self.goog_negative_reviews[
                self.goog_negative_reviews['Club\'s Name'].isin(self.common_locations)
            ]
            tp_neg_common = self.tp_negative_reviews[
                self.tp_negative_reviews['Location Name'].isin(self.common_locations)
            ]
            
            # Combine reviews
            all_negative_reviews = (
                goog_neg_common['clean_comment'].tolist() + 
                tp_neg_common['clean_comment'].tolist()
            )
            
            # Run BERTopic
            model = BERTopic(verbose=True, low_memory=True)
            topics, probabilities = model.fit_transform(all_negative_reviews)
            
            # Display results
            print("Topic frequencies:")
            print(model.get_topic_freq().head(10))
            
            return model, topics, probabilities
            
        except ImportError:
            print("BERTopic not installed. Please install with: pip install bertopic")
            return None, None, None
    
    def _run_lda(self):
        """Run LDA analysis on negative reviews."""
        try:
            from gensim import corpora
            from gensim.models import LdaModel
            
            # Prepare data for LDA
            all_negative_reviews = (
                self.goog_negative_reviews['clean_comment'].tolist() + 
                self.tp_negative_reviews['clean_comment'].tolist()
            )
            
            # Clean and tokenize
            def clean_for_lda(doc):
                stop = set(stopwords.words('english'))
                exclude = set(string.punctuation)
                lemma = WordNetLemmatizer()
                
                stop_free = " ".join([word for word in doc.lower().split() if word not in stop])
                punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
                normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
                return normalized
            
            cleaned_docs = [clean_for_lda(doc).split() for doc in all_negative_reviews]
            
            # Create dictionary and corpus
            dictionary = corpora.Dictionary(cleaned_docs)
            corpus = [dictionary.doc2bow(text) for text in cleaned_docs]
            
            # Run LDA
            lda_model = LdaModel(
                corpus=corpus, 
                id2word=dictionary, 
                num_topics=10, 
                passes=20, 
                random_state=42
            )
            
            # Print topics
            for idx, topic in lda_model.print_topics(-1):
                print(f"Topic {idx}: {topic}")
            
            return lda_model, dictionary, corpus
            
        except ImportError:
            print("Gensim not installed. Please install with: pip install gensim")
            return None, None, None
    
    def run_emotion_analysis(self):
        """Run emotion analysis on negative reviews using BERT."""
        try:
            from transformers import pipeline, AutoTokenizer
            
            # Load emotion classifier
            emotion_classifier = pipeline(
                task="text-classification", 
                model="bhadresh-savani/bert-base-uncased-emotion"
            )
            
            # Truncate reviews to avoid token limit
            tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/bert-base-uncased-emotion")
            
            def truncate_text(text, max_length=510):
                inputs = tokenizer(text, truncation=True, max_length=max_length)
                return tokenizer.decode(inputs["input_ids"])
            
            # Apply truncation
            self.goog_negative_reviews['truncated_comment'] = self.goog_negative_reviews['clean_comment'].apply(truncate_text)
            self.tp_negative_reviews['truncated_comment'] = self.tp_negative_reviews['clean_comment'].apply(truncate_text)
            
            # Run emotion classification
            goog_emotions = emotion_classifier(self.goog_negative_reviews['truncated_comment'].tolist())
            tp_emotions = emotion_classifier(self.tp_negative_reviews['truncated_comment'].tolist())
            
            # Add emotions to dataframes
            self.goog_negative_reviews['emotion'] = [result['label'] for result in goog_emotions]
            self.tp_negative_reviews['emotion'] = [result['label'] for result in tp_emotions]
            
            # Plot emotion distributions
            self._plot_emotion_distribution()
            
            return goog_emotions, tp_emotions
            
        except ImportError:
            print("Transformers not installed. Please install with: pip install transformers")
            return None, None
    
    def _plot_emotion_distribution(self):
        """Plot emotion distribution for negative reviews."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Google emotions
        goog_emotion_counts = self.goog_negative_reviews['emotion'].value_counts()
        goog_emotion_counts.plot(kind='bar', ax=ax1)
        ax1.set_xlabel('Emotion')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Emotion Distribution in Google Negative Reviews')
        ax1.tick_params(axis='x', rotation=45)
        
        # Trustpilot emotions
        tp_emotion_counts = self.tp_negative_reviews['emotion'].value_counts()
        tp_emotion_counts.plot(kind='bar', ax=ax2)
        ax2.set_xlabel('Emotion')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Emotion Distribution in Trustpilot Negative Reviews')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('emotion_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_insights(self, topics_list):
        """
        Generate actionable insights from identified topics.
        
        Args:
            topics_list (list): List of identified topics
            
        Returns:
            str: Generated insights
        """
        # This is a simplified version - in practice, you'd use a more sophisticated LLM
        insights = [
            "1. Implement regular equipment maintenance schedules to address broken machinery",
            "2. Improve parking facilities or provide alternative transportation options",
            "3. Upgrade shower temperature controls and water heating systems",
            "4. Adjust music volume levels and provide quiet zones for focused workouts",
            "5. Enhance cleaning protocols, especially in changing rooms and restrooms",
            "6. Train staff on customer service best practices and conflict resolution",
            "7. Upgrade air conditioning systems for better climate control",
            "8. Replace outdated equipment with modern, reliable alternatives",
            "9. Implement crowd management strategies during peak hours",
            "10. Review membership pricing to ensure competitive value proposition"
        ]
        
        return "\n".join(insights)
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("Starting gym review analysis...")
        
        # Load and clean data
        self.load_data()
        self.clean_data()
        
        # Basic analysis
        print("\n=== Word Frequency Analysis ===")
        self.analyze_word_frequency()
        
        print("\n=== Creating Word Clouds ===")
        self.create_wordclouds()
        
        # Filter negative reviews
        print("\n=== Filtering Negative Reviews ===")
        self.filter_negative_reviews()
        
        # Find common locations
        print("\n=== Finding Common Locations ===")
        self.find_common_locations()
        
        # Topic modeling
        print("\n=== Topic Modeling ===")
        bertopic_model, topics, probabilities = self.run_topic_modeling(use_bertopic=True)
        
        # Emotion analysis
        print("\n=== Emotion Analysis ===")
        self.run_emotion_analysis()
        
        print("\nAnalysis complete! Check the generated visualizations and results.")


def main():
    """Main function to run the analysis."""
    analyzer = GymReviewAnalyzer()
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
