# Data Directory

This directory contains data files and examples for the Gym Review Analysis project.

## Data Format Requirements

### Google Reviews Data
Your Google reviews data should be in Excel format (.xlsx) with the following columns:

- **Comment**: The review text content
- **Overall Score**: Rating from 1-5 (where <3 is considered negative)
- **Club's Name**: The gym location name

### Trustpilot Reviews Data
Your Trustpilot reviews data should be in Excel format (.xlsx) with the following columns:

- **Review Content**: The review text content  
- **Review Stars**: Rating from 1-5 (where <3 is considered negative)
- **Location Name**: The gym location name

## Sample Data

The analyzer includes built-in sample data for demonstration purposes. When you run the analysis without providing URLs, it will automatically generate sample reviews to demonstrate the functionality.

## Data Sources

### Option 1: Google Sheets
If you have your data in Google Sheets:

1. Make sure your sheet is publicly accessible
2. Get the export URL in this format:
   ```
   https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/export?format=xlsx
   ```
3. Use this URL when running the analysis

### Option 2: Local Files
You can also place your Excel files directly in this directory and modify the code to load them locally.

## Data Privacy

- Ensure your data doesn't contain any personally identifiable information (PII)
- Consider anonymizing customer names or other sensitive details
- The analyzer filters out non-English reviews automatically

## Data Quality Tips

1. **Remove duplicates**: The analyzer automatically removes duplicate reviews
2. **Handle missing values**: Reviews with empty text are filtered out
3. **Language filtering**: Only English reviews are processed
4. **Score validation**: Ensure ratings are numeric values between 1-5

## Example Data Structure

### Google Reviews Example:
```
Comment                                    | Overall Score | Club's Name
-------------------------------------------|---------------|-------------
"The gym is always crowded and equipment   | 2             | London Central
 is often broken"
"Great facilities but parking is terrible" | 2             | Manchester
"Staff are friendly but showers are cold"  | 2             | Birmingham
```

### Trustpilot Reviews Example:
```
Review Content                             | Review Stars | Location Name
-------------------------------------------|--------------|-------------
"Love the classes but music is too loud"   | 2            | London Central
"Clean gym with good equipment variety"    | 4            | Manchester
"Terrible customer service and dirty rooms"| 1            | Birmingham
```
