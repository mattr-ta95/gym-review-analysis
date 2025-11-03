# -*- coding: utf-8 -*-
"""
Gym Review Analysis: Topic Modeling and Sentiment Analysis

This project analyzes customer reviews from gym facilities to identify key themes,
sentiment patterns, and actionable insights for business improvement.

The analysis includes:
- Text preprocessing and frequency analysis
- Topic modeling using BERTopic and LDA
- Emotion analysis using BERT
- Insight generation based on identified topics
"""

import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import string
import contractions
from wordcloud import WordCloud
from langdetect import detect, LangDetectException
import warnings
import logging
from typing import Optional, Tuple, List, Set, Union
from pathlib import Path

# Configure warnings - only suppress specific categories
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download required NLTK data only
REQUIRED_NLTK_PACKAGES = ['stopwords', 'punkt', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4']
for package in REQUIRED_NLTK_PACKAGES:
    try:
        nltk.data.find(f'corpora/{package}' if package in ['stopwords', 'wordnet', 'omw-1.4'] else f'tokenizers/{package}')
    except LookupError:
        logger.info(f"Downloading NLTK package: {package}")
        nltk.download(package, quiet=True)


class GymReviewAnalyzer:
    """
    A comprehensive analyzer for gym customer reviews using various NLP techniques.

    Attributes:
        negative_threshold (int): Score threshold for negative reviews (default: 3)
        lda_topics (int): Number of topics for LDA modeling (default: 10)
        lda_passes (int): Number of passes for LDA training (default: 20)
        terms_to_remove (List[str]): Generic terms to remove during preprocessing
        google_text_col (str): Column name for Google review text
        google_score_col (str): Column name for Google review scores
        google_location_col (str): Column name for Google locations
        trustpilot_text_col (str): Column name for Trustpilot review text
        trustpilot_score_col (str): Column name for Trustpilot review scores
        trustpilot_location_col (str): Column name for Trustpilot locations
    """

    def __init__(
        self,
        negative_threshold: int = 3,
        lda_topics: int = 10,
        lda_passes: int = 20,
        terms_to_remove: Optional[List[str]] = None,
        google_text_col: str = 'Comment',
        google_score_col: str = 'Overall Score',
        google_location_col: str = "Club's Name",
        trustpilot_text_col: str = 'Review Content',
        trustpilot_score_col: str = 'Review Stars',
        trustpilot_location_col: str = 'Location Name'
    ):
        """
        Initialize the GymReviewAnalyzer.

        Args:
            negative_threshold: Reviews with scores below this are considered negative
            lda_topics: Number of topics to extract in LDA modeling
            lda_passes: Number of training passes for LDA
            terms_to_remove: List of generic terms to remove during preprocessing
            google_text_col: Column name for Google review text
            google_score_col: Column name for Google review scores
            google_location_col: Column name for Google location names
            trustpilot_text_col: Column name for Trustpilot review text
            trustpilot_score_col: Column name for Trustpilot review scores
            trustpilot_location_col: Column name for Trustpilot location names
        """
        # Configuration
        self.negative_threshold = negative_threshold
        self.lda_topics = lda_topics
        self.lda_passes = lda_passes
        self.terms_to_remove = terms_to_remove or ['gym', 'pure']

        # Column mappings
        self.google_text_col = google_text_col
        self.google_score_col = google_score_col
        self.google_location_col = google_location_col
        self.trustpilot_text_col = trustpilot_text_col
        self.trustpilot_score_col = trustpilot_score_col
        self.trustpilot_location_col = trustpilot_location_col

        # Data storage
        self.goog_reviews: Optional[pd.DataFrame] = None
        self.tp_reviews: Optional[pd.DataFrame] = None
        self.goog_negative_reviews: Optional[pd.DataFrame] = None
        self.tp_negative_reviews: Optional[pd.DataFrame] = None
        self.common_locations: Optional[Set[str]] = None

        # Model cache
        self._emotion_classifier = None

    @property
    def emotion_classifier(self):
        """Lazy-load and cache the emotion classifier model."""
        if self._emotion_classifier is None:
            try:
                from transformers import pipeline
                logger.info("Loading emotion classification model...")
                self._emotion_classifier = pipeline(
                    task="text-classification",
                    model="bhadresh-savani/bert-base-uncased-emotion"
                )
                logger.info("Emotion classifier loaded successfully")
            except ImportError:
                logger.error("Transformers not installed. Install with: pip install transformers")
                raise
            except Exception as e:
                logger.error(f"Error loading emotion classifier: {e}")
                raise
        return self._emotion_classifier

    def load_data(
        self,
        google_url: Optional[str] = None,
        trustpilot_url: Optional[str] = None,
        use_local_files: bool = True
    ) -> None:
        """
        Load review data from Google Sheets, local files, or create sample data.

        Args:
            google_url: URL to Google Sheets data
            trustpilot_url: URL to Trustpilot data
            use_local_files: If True, try to load local Excel files first

        Raises:
            FileNotFoundError: If local files specified but not found
            ValueError: If data format is invalid
        """
        if use_local_files:
            # Try to load local files first
            try:
                data_dir = Path(__file__).parent.parent / 'data'
                google_file = data_dir / 'Google_12_months.xlsx'
                trustpilot_file = data_dir / 'Trustpilot_12_months.xlsx'

                if google_file.exists() and trustpilot_file.exists():
                    logger.info("Loading local Excel files...")
                    self.goog_reviews = pd.read_excel(google_file, engine='openpyxl')
                    self.tp_reviews = pd.read_excel(trustpilot_file, engine='openpyxl')

                    # Validate data structure
                    self._validate_data_structure()

                    logger.info(f"Loaded {len(self.goog_reviews)} Google reviews from local file")
                    logger.info(f"Loaded {len(self.tp_reviews)} Trustpilot reviews from local file")
                    return
                else:
                    logger.warning("Local Excel files not found, trying other options...")
            except Exception as e:
                logger.warning(f"Error loading local files: {e}")
                logger.info("Trying other options...")

        if google_url and trustpilot_url:
            # Load from URLs
            try:
                logger.info("Loading data from Google Sheets URLs...")
                self.goog_reviews = pd.read_excel(google_url, engine='openpyxl')
                self.tp_reviews = pd.read_excel(trustpilot_url, engine='openpyxl')

                # Validate data structure
                self._validate_data_structure()

                logger.info(f"Loaded {len(self.goog_reviews)} Google reviews from URL")
                logger.info(f"Loaded {len(self.tp_reviews)} Trustpilot reviews from URL")
            except Exception as e:
                logger.error(f"Error loading from URLs: {e}")
                raise
        else:
            # For demo purposes, create sample data
            logger.info("Creating sample data for demonstration...")
            self._create_sample_data()
            logger.info(f"Generated {len(self.goog_reviews)} sample Google reviews")
            logger.info(f"Generated {len(self.tp_reviews)} sample Trustpilot reviews")

    def _validate_data_structure(self) -> None:
        """
        Validate that loaded data has required columns.

        Raises:
            ValueError: If required columns are missing
        """
        # Check Google reviews
        required_google_cols = [self.google_text_col, self.google_score_col, self.google_location_col]
        missing_google = [col for col in required_google_cols if col not in self.goog_reviews.columns]
        if missing_google:
            raise ValueError(f"Google reviews missing required columns: {missing_google}")

        # Check Trustpilot reviews
        required_tp_cols = [self.trustpilot_text_col, self.trustpilot_score_col, self.trustpilot_location_col]
        missing_tp = [col for col in required_tp_cols if col not in self.tp_reviews.columns]
        if missing_tp:
            raise ValueError(f"Trustpilot reviews missing required columns: {missing_tp}")

        logger.info("Data structure validation passed")

    def _create_sample_data(self) -> None:
        """Create sample data for demonstration purposes."""
        np.random.seed(42)  # Fixed seed for reproducibility

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
        num_samples = 50
        self.goog_reviews = pd.DataFrame({
            self.google_text_col: sample_reviews * 5,
            self.google_score_col: np.random.randint(1, 6, num_samples),
            self.google_location_col: np.random.choice(sample_locations, num_samples)
        })

        # Create sample Trustpilot reviews
        self.tp_reviews = pd.DataFrame({
            self.trustpilot_text_col: sample_reviews * 5,
            self.trustpilot_score_col: np.random.randint(1, 6, num_samples),
            self.trustpilot_location_col: np.random.choice(sample_locations, num_samples)
        })

    def clean_data(self) -> None:
        """
        Remove missing values, duplicates, and non-English reviews.

        Raises:
            ValueError: If no data has been loaded yet
        """
        if self.goog_reviews is None or self.tp_reviews is None:
            raise ValueError("No data loaded. Call load_data() first.")

        logger.info("Cleaning data...")
        initial_goog = len(self.goog_reviews)
        initial_tp = len(self.tp_reviews)

        # Remove missing values
        self.goog_reviews = self.goog_reviews[self.goog_reviews[self.google_text_col].notnull()].copy()
        self.tp_reviews = self.tp_reviews[self.tp_reviews[self.trustpilot_text_col].notnull()].copy()

        # Remove duplicates
        self.goog_reviews = self.goog_reviews.drop_duplicates(subset=[self.google_text_col]).copy()
        self.tp_reviews = self.tp_reviews.drop_duplicates(subset=[self.trustpilot_text_col]).copy()

        # Filter to English reviews only
        def is_english(text: str) -> bool:
            """Check if text is in English."""
            if not isinstance(text, str) or len(text.strip()) == 0:
                return False
            try:
                return detect(text) == "en"
            except LangDetectException:
                return False

        self.goog_reviews = self.goog_reviews[
            self.goog_reviews[self.google_text_col].apply(is_english)
        ].copy()
        self.tp_reviews = self.tp_reviews[
            self.tp_reviews[self.trustpilot_text_col].apply(is_english)
        ].copy()

        logger.info(f"Google reviews: {initial_goog} -> {len(self.goog_reviews)} "
                   f"(removed {initial_goog - len(self.goog_reviews)})")
        logger.info(f"Trustpilot reviews: {initial_tp} -> {len(self.tp_reviews)} "
                   f"(removed {initial_tp - len(self.tp_reviews)})")

    def preprocess_text(self, text: Union[str, None]) -> str:
        """
        Preprocess text by removing punctuation, numbers, and stopwords.

        Args:
            text: Input text to preprocess

        Returns:
            Preprocessed text

        Raises:
            TypeError: If text is not a string or None
        """
        # Input validation
        if text is None:
            return ""

        if not isinstance(text, str):
            raise TypeError(f"Expected string or None, got {type(text)}")

        if len(text.strip()) == 0:
            return ""

        try:
            # Remove punctuation
            translator = str.maketrans('', '', string.punctuation)
            text = text.translate(translator)

            # Remove numbers
            text = re.sub(r'\d+', '', text)

            # Remove common terms to focus on specific issues
            for term in self.terms_to_remove:
                text = text.replace(term, "").strip()

            # Handle contractions
            text = contractions.fix(text)

            # Tokenize and convert to lowercase
            tokens = word_tokenize(text.lower())

            # Remove stopwords
            stop_words = set(stopwords.words('english'))
            filtered_tokens = [token for token in tokens if token not in stop_words and len(token) > 2]

            # Join tokens back into string
            processed_text = ' '.join(filtered_tokens)

            return processed_text
        except Exception as e:
            logger.warning(f"Error preprocessing text: {e}")
            return ""

    def analyze_word_frequency(self) -> Tuple[object, object]:
        """
        Analyze word frequency in reviews and create visualizations.

        Returns:
            Tuple of (google_freq_dist, trustpilot_freq_dist)

        Raises:
            ValueError: If data hasn't been loaded
        """
        if self.goog_reviews is None or self.tp_reviews is None:
            raise ValueError("No data loaded. Call load_data() first.")

        logger.info("Analyzing word frequencies...")

        # Preprocess reviews
        self.goog_reviews.loc[:, 'clean_comment'] = self.goog_reviews[self.google_text_col].apply(
            self.preprocess_text
        )
        self.tp_reviews.loc[:, 'clean_comment'] = self.tp_reviews[self.trustpilot_text_col].apply(
            self.preprocess_text
        )

        # Calculate frequency distributions
        from nltk.probability import FreqDist

        goog_freq_dist = FreqDist(word_tokenize(' '.join(self.goog_reviews['clean_comment'])))
        tp_freq_dist = FreqDist(word_tokenize(' '.join(self.tp_reviews['clean_comment'])))

        # Plot top 10 words for each dataset
        self._plot_word_frequency(goog_freq_dist, "Google Reviews", "google_freq.png")
        self._plot_word_frequency(tp_freq_dist, "Trustpilot Reviews", "trustpilot_freq.png")

        logger.info("Word frequency analysis complete")
        return goog_freq_dist, tp_freq_dist

    def _plot_word_frequency(self, freq_dist: object, title: str, filename: str) -> None:
        """
        Plot word frequency distribution.

        Args:
            freq_dist: NLTK FreqDist object
            title: Plot title
            filename: Output filename
        """
        if not freq_dist or len(freq_dist) == 0:
            logger.warning(f"No frequency data to plot for {title}")
            return

        top_10 = freq_dist.most_common(10)
        if not top_10:
            logger.warning(f"No words found in frequency distribution for {title}")
            return

        words, freqs = zip(*top_10)

        plt.figure(figsize=(10, 6))
        plt.bar(words, freqs)
        plt.xlabel("Words")
        plt.ylabel("Frequency")
        plt.title(f"Top 10 Words in {title}")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved word frequency plot: {filename}")

    def create_wordclouds(self) -> None:
        """
        Create word clouds for both datasets.

        Raises:
            ValueError: If data hasn't been analyzed yet
        """
        if 'clean_comment' not in self.goog_reviews.columns:
            raise ValueError("Data not preprocessed. Call analyze_word_frequency() first.")

        logger.info("Creating word clouds...")

        # Google reviews wordcloud
        goog_text = ' '.join(self.goog_reviews['clean_comment'])
        if goog_text.strip():
            plt.figure(figsize=(12, 8))
            wc = WordCloud(background_color='black', max_words=1000, max_font_size=50)
            wc.generate(goog_text)
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.title('Google Reviews Word Cloud')
            plt.savefig('google_wordcloud.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Saved Google reviews word cloud")

        # Trustpilot reviews wordcloud
        tp_text = ' '.join(self.tp_reviews['clean_comment'])
        if tp_text.strip():
            plt.figure(figsize=(12, 8))
            wc = WordCloud(background_color='black', max_words=1000, max_font_size=50)
            wc.generate(tp_text)
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.title('Trustpilot Reviews Word Cloud')
            plt.savefig('trustpilot_wordcloud.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Saved Trustpilot reviews word cloud")

    def filter_negative_reviews(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filter reviews with scores below the negative threshold.

        Returns:
            Tuple of (google_negative_reviews, trustpilot_negative_reviews)

        Raises:
            ValueError: If data hasn't been loaded
        """
        if self.goog_reviews is None or self.tp_reviews is None:
            raise ValueError("No data loaded. Call load_data() first.")

        logger.info(f"Filtering negative reviews (score < {self.negative_threshold})...")

        self.goog_negative_reviews = self.goog_reviews[
            self.goog_reviews[self.google_score_col] < self.negative_threshold
        ].copy()
        self.tp_negative_reviews = self.tp_reviews[
            self.tp_reviews[self.trustpilot_score_col] < self.negative_threshold
        ].copy()

        logger.info(f"Negative Google reviews: {len(self.goog_negative_reviews)}")
        logger.info(f"Negative Trustpilot reviews: {len(self.tp_negative_reviews)}")

        return self.goog_negative_reviews, self.tp_negative_reviews

    def find_common_locations(self) -> Set[str]:
        """
        Find common locations between Google and Trustpilot datasets.

        Returns:
            Set of common location names

        Raises:
            ValueError: If data hasn't been loaded
        """
        if self.goog_reviews is None or self.tp_reviews is None:
            raise ValueError("No data loaded. Call load_data() first.")

        self.common_locations = set(self.goog_reviews[self.google_location_col]).intersection(
            set(self.tp_reviews[self.trustpilot_location_col])
        )
        logger.info(f"Common locations: {len(self.common_locations)}")
        return self.common_locations

    def run_topic_modeling(self, use_bertopic: bool = True) -> Union[Tuple, Tuple[None, None, None]]:
        """
        Run topic modeling on negative reviews.

        Args:
            use_bertopic: If True, use BERTopic; if False, use LDA

        Returns:
            For BERTopic: (model, topics, probabilities)
            For LDA: (lda_model, dictionary, corpus)
            Returns (None, None, None) on error

        Raises:
            ValueError: If negative reviews haven't been filtered
        """
        if self.goog_negative_reviews is None or self.tp_negative_reviews is None:
            raise ValueError("Negative reviews not filtered. Call filter_negative_reviews() first.")

        if use_bertopic:
            return self._run_bertopic()
        else:
            return self._run_lda()

    def _run_bertopic(self) -> Union[Tuple, Tuple[None, None, None]]:
        """
        Run BERTopic analysis on negative reviews.

        Returns:
            Tuple of (model, topics, probabilities) or (None, None, None) on error
        """
        try:
            from bertopic import BERTopic
            from tqdm import tqdm

            logger.info("Running BERTopic analysis...")

            # Filter reviews from common locations
            if self.common_locations is None:
                self.find_common_locations()

            goog_neg_common = self.goog_negative_reviews[
                self.goog_negative_reviews[self.google_location_col].isin(self.common_locations)
            ]
            tp_neg_common = self.tp_negative_reviews[
                self.tp_negative_reviews[self.trustpilot_location_col].isin(self.common_locations)
            ]

            # Combine reviews
            all_negative_reviews = (
                goog_neg_common['clean_comment'].tolist() +
                tp_neg_common['clean_comment'].tolist()
            )

            # Filter out empty reviews
            all_negative_reviews = [r for r in all_negative_reviews if r and len(r.strip()) > 0]

            if len(all_negative_reviews) == 0:
                logger.warning("No negative reviews found for topic modeling")
                return None, None, None

            logger.info(f"Analyzing {len(all_negative_reviews)} negative reviews...")

            # Run BERTopic
            model = BERTopic(verbose=True, low_memory=True)
            topics, probabilities = model.fit_transform(all_negative_reviews)

            # Display results
            logger.info("Topic modeling complete. Top topics:")
            topic_freq = model.get_topic_freq().head(10)
            for _, row in topic_freq.iterrows():
                logger.info(f"Topic {row['Topic']}: {row['Count']} documents")

            return model, topics, probabilities

        except ImportError:
            logger.error("BERTopic not installed. Install with: pip install bertopic")
            return None, None, None
        except Exception as e:
            logger.error(f"Error running BERTopic: {e}")
            return None, None, None

    def _run_lda(self) -> Union[Tuple, Tuple[None, None, None]]:
        """
        Run LDA analysis on negative reviews.

        Returns:
            Tuple of (lda_model, dictionary, corpus) or (None, None, None) on error
        """
        try:
            from gensim import corpora
            from gensim.models import LdaModel

            logger.info("Running LDA analysis...")

            # Prepare data for LDA
            all_negative_reviews = (
                self.goog_negative_reviews['clean_comment'].tolist() +
                self.tp_negative_reviews['clean_comment'].tolist()
            )

            # Filter out empty reviews
            all_negative_reviews = [r for r in all_negative_reviews if r and len(r.strip()) > 0]

            if len(all_negative_reviews) == 0:
                logger.warning("No negative reviews found for topic modeling")
                return None, None, None

            # Clean and tokenize
            def clean_for_lda(doc: str) -> str:
                """Additional cleaning for LDA."""
                stop = set(stopwords.words('english'))
                exclude = set(string.punctuation)
                lemma = WordNetLemmatizer()

                stop_free = " ".join([word for word in doc.lower().split() if word not in stop])
                punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
                normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
                return normalized

            cleaned_docs = [clean_for_lda(doc).split() for doc in all_negative_reviews]
            cleaned_docs = [doc for doc in cleaned_docs if len(doc) > 0]

            if len(cleaned_docs) == 0:
                logger.warning("No valid documents after cleaning")
                return None, None, None

            # Create dictionary and corpus
            dictionary = corpora.Dictionary(cleaned_docs)
            corpus = [dictionary.doc2bow(text) for text in cleaned_docs]

            # Run LDA
            logger.info(f"Training LDA model with {self.lda_topics} topics...")
            lda_model = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=self.lda_topics,
                passes=self.lda_passes,
                random_state=42
            )

            # Print topics
            logger.info("LDA topics identified:")
            for idx, topic in lda_model.print_topics(-1):
                logger.info(f"Topic {idx}: {topic}")

            return lda_model, dictionary, corpus

        except ImportError:
            logger.error("Gensim not installed. Install with: pip install gensim")
            return None, None, None
        except Exception as e:
            logger.error(f"Error running LDA: {e}")
            return None, None, None

    def run_emotion_analysis(self) -> Union[Tuple[List, List], Tuple[None, None]]:
        """
        Run emotion analysis on negative reviews using BERT.

        Returns:
            Tuple of (google_emotions, trustpilot_emotions) or (None, None) on error

        Raises:
            ValueError: If negative reviews haven't been filtered
        """
        if self.goog_negative_reviews is None or self.tp_negative_reviews is None:
            raise ValueError("Negative reviews not filtered. Call filter_negative_reviews() first.")

        if len(self.goog_negative_reviews) == 0 and len(self.tp_negative_reviews) == 0:
            logger.warning("No negative reviews to analyze")
            return None, None

        try:
            from transformers import AutoTokenizer
            from tqdm import tqdm

            logger.info("Running emotion analysis...")

            # Get the emotion classifier (uses cached version if available)
            classifier = self.emotion_classifier

            # Truncate reviews to avoid token limit
            tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/bert-base-uncased-emotion")

            def truncate_text(text: str, max_length: int = 510) -> str:
                """Truncate text to maximum token length."""
                if not text or len(text.strip()) == 0:
                    return ""
                inputs = tokenizer(text, truncation=True, max_length=max_length)
                return tokenizer.decode(inputs["input_ids"], skip_special_tokens=True)

            # Apply truncation
            goog_reviews_list = self.goog_negative_reviews['clean_comment'].tolist()
            tp_reviews_list = self.tp_negative_reviews['clean_comment'].tolist()

            # Filter empty reviews
            goog_reviews_list = [r if r and len(r.strip()) > 0 else "no comment" for r in goog_reviews_list]
            tp_reviews_list = [r if r and len(r.strip()) > 0 else "no comment" for r in tp_reviews_list]

            goog_truncated = [truncate_text(r) for r in tqdm(goog_reviews_list, desc="Truncating Google reviews")]
            tp_truncated = [truncate_text(r) for r in tqdm(tp_reviews_list, desc="Truncating Trustpilot reviews")]

            # Run emotion classification
            logger.info("Classifying emotions in Google reviews...")
            goog_emotions = classifier(goog_truncated)

            logger.info("Classifying emotions in Trustpilot reviews...")
            tp_emotions = classifier(tp_truncated)

            # Add emotions to dataframes (use .loc to avoid SettingWithCopyWarning)
            self.goog_negative_reviews = self.goog_negative_reviews.copy()
            self.tp_negative_reviews = self.tp_negative_reviews.copy()

            self.goog_negative_reviews.loc[:, 'emotion'] = [result['label'] for result in goog_emotions]
            self.tp_negative_reviews.loc[:, 'emotion'] = [result['label'] for result in tp_emotions]

            # Plot emotion distributions
            self._plot_emotion_distribution()

            logger.info("Emotion analysis complete")
            return goog_emotions, tp_emotions

        except ImportError:
            logger.error("Transformers not installed. Install with: pip install transformers")
            return None, None
        except Exception as e:
            logger.error(f"Error running emotion analysis: {e}")
            return None, None

    def _plot_emotion_distribution(self) -> None:
        """Plot emotion distribution for negative reviews."""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Google emotions
            if 'emotion' in self.goog_negative_reviews.columns:
                goog_emotion_counts = self.goog_negative_reviews['emotion'].value_counts()
                goog_emotion_counts.plot(kind='bar', ax=ax1)
                ax1.set_xlabel('Emotion')
                ax1.set_ylabel('Frequency')
                ax1.set_title('Emotion Distribution in Google Negative Reviews')
                ax1.tick_params(axis='x', rotation=45)

            # Trustpilot emotions
            if 'emotion' in self.tp_negative_reviews.columns:
                tp_emotion_counts = self.tp_negative_reviews['emotion'].value_counts()
                tp_emotion_counts.plot(kind='bar', ax=ax2)
                ax2.set_xlabel('Emotion')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Emotion Distribution in Trustpilot Negative Reviews')
                ax2.tick_params(axis='x', rotation=45)

            plt.tight_layout()
            plt.savefig('emotion_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Saved emotion distribution plot")
        except Exception as e:
            logger.error(f"Error plotting emotion distribution: {e}")

    def generate_insights(self, topics_list: Optional[List] = None) -> str:
        """
        Generate actionable insights from identified topics.

        Args:
            topics_list: List of identified topics (currently not used, reserved for future LLM integration)

        Returns:
            String containing numbered insights

        Note:
            This is a simplified version with template-based insights.
            Future versions could integrate with LLMs for dynamic insight generation.
        """
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

    def run_complete_analysis(self) -> None:
        """
        Run the complete analysis pipeline.

        This method orchestrates all analysis steps in sequence.
        """
        try:
            logger.info("Starting gym review analysis pipeline...")

            # Load and clean data
            if self.goog_reviews is None:
                self.load_data()
            self.clean_data()

            # Basic analysis
            logger.info("=== Word Frequency Analysis ===")
            self.analyze_word_frequency()

            logger.info("=== Creating Word Clouds ===")
            self.create_wordclouds()

            # Filter negative reviews
            logger.info("=== Filtering Negative Reviews ===")
            self.filter_negative_reviews()

            # Find common locations
            logger.info("=== Finding Common Locations ===")
            self.find_common_locations()

            # Topic modeling
            logger.info("=== Topic Modeling ===")
            bertopic_model, topics, probabilities = self.run_topic_modeling(use_bertopic=True)

            # Emotion analysis
            logger.info("=== Emotion Analysis ===")
            self.run_emotion_analysis()

            logger.info("Analysis complete! Check the generated visualizations and results.")

        except Exception as e:
            logger.error(f"Error in analysis pipeline: {e}")
            raise


def main():
    """Main function to run the analysis."""
    analyzer = GymReviewAnalyzer()
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
