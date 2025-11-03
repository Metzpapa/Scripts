import pandas as pd
import requests
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
API_KEY = os.getenv('GOOGLE_API_KEY')  # Load from .env file
if not API_KEY:
    print("ERROR: GOOGLE_API_KEY not found in .env file")
    print("Please add it to your .env file: GOOGLE_API_KEY='your-key-here'")
    exit()

INPUT_CSV_FILE = 'listings.csv'      # <-- The name of your data file
OUTPUT_CSV_FILE = 'listings_with_addresses.csv'
LAT_COLUMN = 'latitude'              # The name of your latitude column
LON_COLUMN = 'longitude'             # The name of your longitude column
# -------------------

def get_address_from_coords(lat, lon, api_key):
    """Fetches an address from Google Maps API for given coordinates."""
    if pd.isna(lat) or pd.isna(lon):
        return "Invalid Coordinates"
    
    base_url = "https://maps.googleapis.com/maps/api/geocode/json?"
    params = {
        'latlng': f'{lat},{lon}',
        'key': api_key
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raises an exception for bad status codes
        results = response.json().get('results', [])
        
        if results:
            return results[0]['formatted_address']
        else:
            return "Address Not Found"
    except requests.exceptions.RequestException as e:
        return f"API Request Error: {e}"

# Read the input CSV file
print(f"Reading data from {INPUT_CSV_FILE}...")
df = pd.read_csv(INPUT_CSV_FILE)

# Create a new column for the address
df['full_address'] = ''

print("Starting reverse geocoding process (this may take a while)...")
# Loop through each row in the DataFrame
for index, row in df.iterrows():
    lat = row[LAT_COLUMN]
    lon = row[LON_COLUMN]
    
    address = get_address_from_coords(lat, lon, API_KEY)
    df.at[index, 'full_address'] = address
    
    print(f"Row {index + 1}/{len(df)}: Found address -> {address}")
    
    # Be respectful to the API and avoid hitting rate limits
    time.sleep(0.1) 

# Save the updated DataFrame to a new CSV file
print(f"Process complete. Saving results to {OUTPUT_CSV_FILE}...")
df.to_csv(OUTPUT_CSV_FILE, index=False)

print("Done!")