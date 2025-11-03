#!/usr/bin/env python3
"""
Main script to run gym review analysis
"""

import sys
import os
import logging
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from gym_review_analysis import GymReviewAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main function to run the analysis"""
    logger.info("=" * 60)
    logger.info("Gym Review Analysis Tool")
    logger.info("=" * 60)

    # Initialize analyzer
    analyzer = GymReviewAnalyzer()

    # Check if local Excel files exist
    data_dir = Path(__file__).parent / 'data'
    google_file = data_dir / 'Google_12_months.xlsx'
    trustpilot_file = data_dir / 'Trustpilot_12_months.xlsx'

    if google_file.exists() and trustpilot_file.exists():
        logger.info("Found local Excel files - loading them automatically...")
        analyzer.load_data(use_local_files=True)
    else:
        logger.info("\nChoose data source:")
        logger.info("1. Use sample data (for demonstration)")
        logger.info("2. Provide Google Sheets URLs")

        choice = input("\nEnter your choice (1 or 2): ").strip()

        if choice == "2":
            google_url = input("Enter Google Sheets URL: ").strip()
            trustpilot_url = input("Enter Trustpilot Sheets URL: ").strip()

            if not google_url or not trustpilot_url:
                logger.warning("URLs cannot be empty. Using sample data instead.")
                analyzer.load_data(use_local_files=False)
            else:
                try:
                    analyzer.load_data(google_url, trustpilot_url, use_local_files=False)
                except Exception as e:
                    logger.error(f"Failed to load from URLs: {e}")
                    logger.info("Falling back to sample data...")
                    analyzer.load_data(use_local_files=False)
        else:
            logger.info("Loading sample data...")
            analyzer.load_data(use_local_files=False)

    # Run analysis
    logger.info("\nStarting analysis...")
    try:
        analyzer.run_complete_analysis()

        logger.info("\n" + "=" * 60)
        logger.info("Analysis completed successfully!")
        logger.info("=" * 60)
        logger.info("\nGenerated files:")
        logger.info("  - google_freq.png")
        logger.info("  - trustpilot_freq.png")
        logger.info("  - google_wordcloud.png")
        logger.info("  - trustpilot_wordcloud.png")
        logger.info("  - emotion_distribution.png")

        return 0

    except Exception as e:
        logger.error(f"\nError during analysis: {str(e)}", exc_info=True)
        logger.error("Please check your data format and try again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
