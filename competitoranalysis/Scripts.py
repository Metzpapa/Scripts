import os
import glob
import time
import csv
import requests
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- CONFIGURATION V2 ---
API_KEY = os.getenv('GOOGLE_API_KEY')  # Load from .env file
if not API_KEY:
    print("ERROR: GOOGLE_API_KEY not found in .env file")
    print("Please add it to your .env file: GOOGLE_API_KEY='your-key-here'")
    exit()

OUTPUT_FILE = 'str_cleaning_market_analysis_v2.csv'
SEARCH_URL = 'https://maps.googleapis.com/maps/api/place/textsearch/json'

# V2: Even more specific search queries
SEARCH_QUERIES = [
    'Airbnb turnover cleaning service in {}',
    'Vacation rental cleaning services in {}',
    'STR cleaning company in {}',
    'Short term rental cleaners in {}'
]

# V2: NEGATIVE KEYWORD FILTER
# We will exclude any business whose name contains any of these words (case-insensitive)
NEGATIVE_KEYWORDS = [
    'real estate', 'realty', 'property management', 'hvac', 'solar', 
    'government', 'county', 'department', 'realtor', 'community', 
    'apartments', 'heating', 're/max', 'solutions', 'realty', 'group',
    'administration', 'license', 'residential services'
]


def parse_location_from_filename(filepath):
    """Extracts a clean location name from the CSV filename."""
    basename = os.path.basename(filepath)
    location_no_ext = os.path.splitext(basename)[0]
    
    # Handle special cases first
    if "WashingtonD.C." in location_no_ext:
        return "Washington D.C."
    if ',' in location_no_ext:
        return location_no_ext.replace(',', ', ')
        
    parts = location_no_ext.split('_')
    if len(parts) > 1:
        city = ' '.join(parts[:-1])
        state = parts[-1]
        return f"{city}, {state}"
    else:
        # Fallback for filenames like "Rhodelsland_Rhodelsland.csv"
        return parts[0]


def get_total_listings(filepath):
    """Counts the number of listings in an Inside Airbnb CSV file."""
    try:
        df = pd.read_csv(filepath)
        return len(df)
    except Exception as e:
        print(f"  [ERROR] Could not read {filepath}: {e}")
        return 0

def find_specialist_cleaners(location):
    """
    Searches Google Places, then applies a negative keyword filter.
    Returns a list of unique, relevant competitors.
    """
    print(f"  -> Searching for competitors in {location}...")
    unique_competitors = {} # Using a dict with place_id for de-duplication

    for query_template in SEARCH_QUERIES:
        query = query_template.format(location)
        params = {'query': query, 'key': API_KEY}
        
        while True:
            try:
                response = requests.get(SEARCH_URL, params=params)
                response.raise_for_status()
                results = response.json()
                
                for place in results.get('results', []):
                    place_id = place.get('place_id')
                    place_name = place.get('name', '').lower()
                    
                    # --- V2: THE FILTERING LOGIC ---
                    is_excluded = any(keyword in place_name for keyword in NEGATIVE_KEYWORDS)
                    
                    if place_id and place_id not in unique_competitors and not is_excluded:
                        unique_competitors[place_id] = place.get('name', 'N/A')
                    elif is_excluded:
                        print(f"      -> Filtering out: {place.get('name', 'N/A')} (contains negative keyword)")

                next_page_token = results.get('next_page_token')
                if next_page_token:
                    print("  -> Fetching next page of results...")
                    params['pagetoken'] = next_page_token
                    time.sleep(2)
                else:
                    break

            except requests.exceptions.RequestException as e:
                print(f"  [ERROR] API request failed for query '{query}': {e}")
                break
    
    # The dictionary values are the names
    filtered_list = list(unique_competitors.values())
    print(f"  -> Found {len(filtered_list)} unique and *filtered* specialist competitors for {location}.")
    return filtered_list


def main():
    """Main function to run the analysis."""
    if API_KEY == 'YOUR_API_KEY_HERE' or not API_KEY:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! ERROR: Please replace 'YOUR_API_KEY_HERE' with   !!!")
        print("!!! your actual Google Places API key in the script. !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return

    csv_files = [f for f in glob.glob('*.csv') if f != OUTPUT_FILE]
    
    if not csv_files:
        print("No Inside Airbnb CSV files found in this directory.")
        return

    all_market_data = []
    print(f"Found {len(csv_files)} location files to analyze.")

    for filepath in csv_files:
        location_name = parse_location_from_filename(filepath)
        print(f"\n--- Processing: {location_name} ---")

        total_listings = get_total_listings(filepath)
        if total_listings == 0:
            continue
        print(f"  -> Found {total_listings} total STR listings.")

        competitors = find_specialist_cleaners(location_name)
        competitor_count = len(competitors)
        
        if competitor_count > 0:
            opportunity_score = total_listings / competitor_count
        else:
            opportunity_score = total_listings

        competitors_list_str = " | ".join(sorted(competitors))

        all_market_data.append({
            'Location': location_name,
            'Opportunity_Score': round(opportunity_score, 2),
            'Total_STR_Listings': total_listings,
            'Specialist_Cleaner_Count': competitor_count,
            'Specialist_Cleaners_List': competitors_list_str
        })
        time.sleep(1)

    sorted_market_data = sorted(all_market_data, key=lambda x: x['Opportunity_Score'], reverse=True)
    
    print(f"\n--- Analysis complete. Writing results to {OUTPUT_FILE} ---")
    
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as outfile:
        if sorted_market_data:
            writer = csv.DictWriter(outfile, fieldnames=sorted_market_data[0].keys())
            writer.writeheader()
            writer.writerows(sorted_market_data)

    print("--- All done! Your much cleaner results are in " + OUTPUT_FILE + " ---")


if __name__ == "__main__":
    main()