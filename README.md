# Gym Review Analysis

A comprehensive Natural Language Processing (NLP) toolkit for analyzing customer reviews from gym facilities. This project uses advanced topic modeling, sentiment analysis, and emotion detection techniques to extract actionable insights from customer feedback.

## Features

- **Text Preprocessing**: Advanced text cleaning, tokenization, and normalization
- **Word Frequency Analysis**: Statistical analysis of most common terms in reviews
- **Topic Modeling**: 
  - BERTopic for modern neural topic modeling
  - LDA (Latent Dirichlet Allocation) for traditional topic modeling
- **Emotion Analysis**: BERT-based emotion classification
- **Visualization**: Word clouds, frequency plots, and topic visualizations
- **Insight Generation**: Template-based actionable recommendations (extensible for LLM integration)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gym-review-analysis.git
cd gym-review-analysis
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download NLTK data (if not already downloaded):
```python
import nltk
nltk.download('all')
```

## Quick Start

### Basic Usage

```python
from src.gym_review_analysis import GymReviewAnalyzer

# Initialize the analyzer
analyzer = GymReviewAnalyzer()

# Load sample data (or provide your own data URLs)
analyzer.load_data()

# Run complete analysis
analyzer.run_complete_analysis()
```

### Using Your Own Data

```python
# Load data from Google Sheets or Excel files
google_url = "https://docs.google.com/spreadsheets/d/your-google-sheet-url/export?format=xlsx"
trustpilot_url = "https://docs.google.com/spreadsheets/d/your-trustpilot-url/export?format=xlsx"

analyzer = GymReviewAnalyzer()
analyzer.load_data(google_url, trustpilot_url)
analyzer.run_complete_analysis()
```

## Data Format

The analyzer expects data in the following format:

### Google Reviews
- `Comment`: Review text content
- `Overall Score`: Rating (1-5)
- `Club's Name`: Gym location name

### Trustpilot Reviews
- `Review Content`: Review text content
- `Review Stars`: Rating (1-5)
- `Location Name`: Gym location name

## Analysis Components

### 1. Text Preprocessing
- Removes punctuation, numbers, and stopwords
- Handles contractions and tokenization
- Filters non-English content

### 2. Word Frequency Analysis
- Identifies most common terms in reviews
- Creates frequency distribution plots
- Generates word clouds for visual representation

### 3. Topic Modeling
- **BERTopic**: Uses transformer-based embeddings for topic discovery
- **LDA**: Traditional probabilistic topic modeling
- Interactive visualizations of topic clusters

### 4. Emotion Analysis
- BERT-based emotion classification (joy, anger, fear, sadness, etc.)
- Emotion distribution analysis
- Focus on negative review emotions

### 5. Insight Generation
- Template-based actionable recommendations
- Business-focused improvement suggestions
- Data-driven decision support
- Extensible architecture for future LLM integration

## Project Structure

```
gym-review-analysis/
├── src/                    # Source code
│   ├── __init__.py
│   └── gym_review_analysis.py
├── data/                   # Data files and examples
├── notebooks/              # Jupyter notebooks for exploration
├── tests/                  # Unit tests
├── docs/                   # Documentation
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

## Examples

### Running Topic Modeling

```python
# Run BERTopic analysis
model, topics, probabilities = analyzer.run_topic_modeling(use_bertopic=True)

# Run LDA analysis
lda_model, dictionary, corpus = analyzer.run_topic_modeling(use_bertopic=False)
```

### Emotion Analysis

```python
# Analyze emotions in negative reviews
goog_emotions, tp_emotions = analyzer.run_emotion_analysis()
```

### Custom Analysis

```python
# Filter negative reviews
analyzer.filter_negative_reviews()

# Find common locations
common_locs = analyzer.find_common_locations()

# Generate insights
insights = analyzer.generate_insights(topics_list)
print(insights)
```

## Output Files

The analysis generates several output files:

- `google_freq.png`: Word frequency plot for Google reviews
- `trustpilot_freq.png`: Word frequency plot for Trustpilot reviews
- `google_wordcloud.png`: Word cloud for Google reviews
- `trustpilot_wordcloud.png`: Word cloud for Trustpilot reviews
- `emotion_distribution.png`: Emotion distribution plots

## Dependencies

### Core Libraries
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computing
- `matplotlib` & `seaborn`: Data visualization
- `nltk`: Natural language processing

### Advanced NLP
- `bertopic`: Neural topic modeling
- `transformers`: Hugging Face transformers
- `gensim`: Traditional topic modeling
- `wordcloud`: Word cloud generation

### Machine Learning
- `torch`: PyTorch for deep learning
- `sentence-transformers`: Sentence embeddings
- `datasets`: Hugging Face datasets

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Hugging Face for transformer models and datasets
- BERTopic developers for the excellent topic modeling library
- NLTK community for natural language processing tools
- The open-source Python ecosystem

## Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/gym-review-analysis/issues) page
2. Create a new issue with detailed information
3. Contact the maintainer at matthew.russell@example.com

## Changelog

### Version 1.0.0
- Initial release
- Basic topic modeling with BERTopic and LDA
- Emotion analysis with BERT
- Word frequency analysis and visualization
- Sample data generation for testing
