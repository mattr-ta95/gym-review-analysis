"""
Unit tests for the Gym Review Analyzer
"""

import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from gym_review_analysis import GymReviewAnalyzer


class TestGymReviewAnalyzer(unittest.TestCase):
    """Test cases for GymReviewAnalyzer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = GymReviewAnalyzer()
    
    def test_initialization(self):
        """Test analyzer initialization"""
        self.assertIsNotNone(self.analyzer)
        self.assertIsNone(self.analyzer.goog_reviews)
        self.assertIsNone(self.analyzer.tp_reviews)
    
    def test_load_sample_data(self):
        """Test loading sample data"""
        self.analyzer.load_data()
        self.assertIsNotNone(self.analyzer.goog_reviews)
        self.assertIsNotNone(self.analyzer.tp_reviews)
        self.assertGreater(len(self.analyzer.goog_reviews), 0)
        self.assertGreater(len(self.analyzer.tp_reviews), 0)
    
    def test_preprocess_text(self):
        """Test text preprocessing functionality"""
        test_text = "This is a TEST review with numbers 123 and punctuation!!!"
        processed = self.analyzer.preprocess_text(test_text)
        
        # Should be lowercase
        self.assertEqual(processed, processed.lower())
        # Should not contain numbers
        self.assertFalse(any(char.isdigit() for char in processed))
        # Should not contain punctuation
        self.assertFalse(any(char in "!@#$%^&*()_+-=[]{}|;':\",./<>?" for char in processed))
    
    def test_clean_data(self):
        """Test data cleaning functionality"""
        self.analyzer.load_data()
        initial_goog_count = len(self.analyzer.goog_reviews)
        initial_tp_count = len(self.analyzer.tp_reviews)
        
        self.analyzer.clean_data()
        
        # Should have fewer or equal reviews after cleaning
        self.assertLessEqual(len(self.analyzer.goog_reviews), initial_goog_count)
        self.assertLessEqual(len(self.analyzer.tp_reviews), initial_tp_count)
    
    def test_filter_negative_reviews(self):
        """Test filtering negative reviews"""
        self.analyzer.load_data()
        self.analyzer.clean_data()
        
        goog_neg, tp_neg = self.analyzer.filter_negative_reviews()
        
        # All negative reviews should have scores < 3
        if len(goog_neg) > 0:
            self.assertTrue(all(score < 3 for score in goog_neg['Overall Score']))
        if len(tp_neg) > 0:
            self.assertTrue(all(score < 3 for score in tp_neg['Review Stars']))
    
    def test_find_common_locations(self):
        """Test finding common locations"""
        self.analyzer.load_data()
        self.analyzer.clean_data()
        
        common_locs = self.analyzer.find_common_locations()
        self.assertIsInstance(common_locs, set)
        self.assertGreaterEqual(len(common_locs), 0)
    
    def test_generate_insights(self):
        """Test insight generation"""
        sample_topics = ["equipment", "cleanliness", "staff"]
        insights = self.analyzer.generate_insights(sample_topics)
        
        self.assertIsInstance(insights, str)
        self.assertGreater(len(insights), 0)
        # Should contain numbered list
        self.assertIn("1.", insights)


if __name__ == '__main__':
    unittest.main()
