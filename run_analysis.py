#!/usr/bin/env python3
"""
Main script to run gym review analysis
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from gym_review_analysis import GymReviewAnalyzer


def main():
    """Main function to run the analysis"""
    print("🏋️‍♂️ Gym Review Analysis Tool")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = GymReviewAnalyzer()
    
    # Check if local Excel files exist
    import os
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    google_file = os.path.join(data_dir, 'Google_12_months.xlsx')
    trustpilot_file = os.path.join(data_dir, 'Trustpilot_12_months.xlsx')
    
    if os.path.exists(google_file) and os.path.exists(trustpilot_file):
        print("📁 Found local Excel files - loading them automatically...")
        analyzer.load_data(use_local_files=True)
    else:
        print("\nChoose data source:")
        print("1. Use sample data (for demonstration)")
        print("2. Provide Google Sheets URLs")
        
        choice = input("\nEnter your choice (1 or 2): ").strip()
        
        if choice == "2":
            google_url = input("Enter Google Sheets URL: ").strip()
            trustpilot_url = input("Enter Trustpilot Sheets URL: ").strip()
            
            if not google_url or not trustpilot_url:
                print("❌ URLs cannot be empty. Using sample data instead.")
                analyzer.load_data(use_local_files=False)
            else:
                analyzer.load_data(google_url, trustpilot_url, use_local_files=False)
        else:
            print("📊 Loading sample data...")
            analyzer.load_data(use_local_files=False)
    
    # Run analysis
    print("\n🔍 Starting analysis...")
    try:
        analyzer.run_complete_analysis()
        print("\n✅ Analysis completed successfully!")
        print("\nGenerated files:")
        print("- google_freq.png")
        print("- trustpilot_freq.png") 
        print("- google_wordcloud.png")
        print("- trustpilot_wordcloud.png")
        print("- emotion_distribution.png")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        print("Please check your data format and try again.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
