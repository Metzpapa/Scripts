import pandas as pd

def create_high_signal_hit_list(
    listings_file='nashville_listings.csv',
    output_file='hit_list_high_signal.csv',
    min_bio_length=250  # Set a threshold for what we consider a "long" bio
):
    """
    Processes Airbnb listings to generate a targeted hit list of hosts
    who are not only professionals but also have long, descriptive bios,
    indicating a higher likelihood of finding contact information.
    """
    print("--- Phase 1: Loading and Preparing Data ---")
    try:
        df_listings = pd.read_csv(listings_file)
        print("Successfully loaded listings data.")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the listings CSV is in the correct directory.")
        return

    # --- Data Cleaning ---
    # Crucially, clean up the host_about column
    df_listings['host_about'] = df_listings['host_about'].astype(str).fillna('')
    # Calculate the length of the bio
    df_listings['host_about_length'] = df_listings['host_about'].str.len()

    print("--- Phase 2: Applying High-Signal Filters ---")
    
    # --- Rule 1: The High-Signal Professional Host Profile ---
    # This is our new, more restrictive filter.
    high_signal_filter = (
        (df_listings['host_total_listings_count'] >= 4) &
        (df_listings['host_total_listings_count'] <= 25) & # Slightly increased upper bound
        (df_listings['property_type'].str.contains('Entire', na=False)) &
        (df_listings['host_about_length'] >= min_bio_length) # THE NEW KEY FILTER
    )
    
    df_filtered = df_listings[high_signal_filter].copy()

    # We don't need A/B lists anymore; this filter is strong enough on its own.
    # We just need one entry per host.
    df_final_hosts = df_filtered.drop_duplicates(subset=['host_id'], keep='first')
    
    print(f"Identified {len(df_final_hosts)} unique hosts with long, professional bios.")

    print("--- Phase 3: Exporting Actionable CSV ---")
    output_columns = [
        'host_id',
        'host_name',
        'host_total_listings_count',
        'host_about_length', # So you can sort by it
        'host_about',        # The actual text for your analysis
        'host_url',
        'listing_url'
    ]
    
    hit_list_high_signal = df_final_hosts[output_columns]
    
    # Sort by bio length so the most promising leads are at the top
    hit_list_high_signal = hit_list_high_signal.sort_values(by='host_about_length', ascending=False)

    hit_list_high_signal.to_csv(output_file, index=False)
    
    print("-" * 30)
    print("SUCCESS!")
    print(f"Generated '{output_file}' with {len(hit_list_high_signal)} high-signal hosts.")
    print("The file is sorted by bio length. Start your search with the hosts at the top!")
    print("-" * 30)

# --- Run the script ---
if __name__ == '__main__':
    # You can adjust the min_bio_length here if you want to be more or less strict
    create_high_signal_hit_list(min_bio_length=250)