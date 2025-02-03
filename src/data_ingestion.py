import os
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

def load_api_key():
    """
    Step 1: Load the NASA API key from the .env file.
    """
    load_dotenv()  # Loads environment variables from .env file
    api_key = os.getenv("NASA_API_KEY")
    if not api_key:
        raise ValueError("NASA_API_KEY not found in .env file. Please set it before running the script.")
    return api_key

def fetch_feed_data(start_date: str, end_date: str, api_key: str) -> dict:
    """
    Fetch data from NASA's NeoWs Feed endpoint for the given date range.

    Args:
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        api_key (str): NASA API key.

    Returns:
        dict: JSON data returned by the API.
    """
    url = "https://api.nasa.gov/neo/rest/v1/feed"
    params = {
        "start_date": start_date,
        "end_date": end_date,
        "api_key": api_key
    }
    response = requests.get(url, params=params)
    response.raise_for_status()  # Will raise an HTTPError for bad responses
    return response.json()

def process_data(json_data: dict) -> list:
    """
    Process and flatten the JSON data from the API into a list of records.
    
    Args:
        json_data (dict): Raw JSON data from the NASA API.
    
    Returns:
        list: A list of flattened records (each record is a dict).
    """
    records = []
    # The API returns data under the "near_earth_objects" key,
    # which is a dictionary with dates as keys and lists of asteroid records as values.
    neo_data = json_data.get("near_earth_objects", {})
    for date, asteroids in neo_data.items():
        for asteroid in asteroids:
            record = {
                "id": asteroid.get("id"),
                "name": asteroid.get("name"),
                "absolute_magnitude_h": asteroid.get("absolute_magnitude_h"),
                "estimated_diameter_min_km": asteroid.get("estimated_diameter", {})\
                    .get("kilometers", {}).get("estimated_diameter_min"),
                "estimated_diameter_max_km": asteroid.get("estimated_diameter", {})\
                    .get("kilometers", {}).get("estimated_diameter_max"),
                "is_sentry_object": asteroid.get("is_sentry_object"),
                "is_potentially_hazardous_asteroid": asteroid.get("is_potentially_hazardous_asteroid"),
                "close_approach_date": None,
                "relative_velocity_kph": None,
                "miss_distance_km": None,
                "orbiting_body": None
            }
            # Process the first close approach data if available
            if asteroid.get("close_approach_data"):
                ca_data = asteroid["close_approach_data"][0]
                record["close_approach_date"] = ca_data.get("close_approach_date")
                record["relative_velocity_kph"] = ca_data.get("relative_velocity", {})\
                    .get("kilometers_per_hour")
                record["miss_distance_km"] = ca_data.get("miss_distance", {})\
                    .get("kilometers")
                record["orbiting_body"] = ca_data.get("orbiting_body")
            records.append(record)
    return records

def fetch_data_over_range(overall_start: datetime, overall_end: datetime, api_key: str) -> list:
    """
    Fetch data over an overall date range by splitting it into 7-day segments.

    Args:
        overall_start (datetime): Overall start datetime.
        overall_end (datetime): Overall end datetime.
        api_key (str): NASA API key.
    
    Returns:
        list: Combined list of records from all segments.
    """
    all_records = []
    current_start = overall_start

    # The API allows a maximum of 7 days per request.
    while current_start <= overall_end:
        current_end = min(current_start + timedelta(days=6), overall_end)  # inclusive 7-day window
        s_date = current_start.strftime("%Y-%m-%d")
        e_date = current_end.strftime("%Y-%m-%d")
        print(f"Fetching data from {s_date} to {e_date}...")
        try:
            json_data = fetch_feed_data(s_date, e_date, api_key)
            records = process_data(json_data)
            all_records.extend(records)
            print(f"Fetched {len(records)} records for {s_date} to {e_date}.")
        except requests.RequestException as e:
            print(f"Error fetching data from {s_date} to {e_date}: {e}")
        # Move to the next segment (avoid overlapping by moving one day ahead)
        current_start = current_end + timedelta(days=1)
    return all_records

def save_csv(records: list, filename: str):
    """
    Save the processed records to a CSV file.

    Args:
        records (list): List of dictionaries (flattened records).
        filename (str): File path where data will be saved.
    """
    df = pd.DataFrame(records)
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename} in CSV format.")

def main():
    # Load the API key from .env
    api_key = load_api_key()
    
    # Define the overall date range from January 1, 2004 to December 31, 2025.
    overall_start = datetime(2004, 1, 1)
    overall_end = datetime(2025, 1, 1)
    print(f"Fetching data from {overall_start.strftime('%Y-%m-%d')} to {overall_end.strftime('%Y-%m-%d')}...")

    # Fetch data over the specified range.
    records = fetch_data_over_range(overall_start, overall_end, api_key)
    print(f"Total records fetched: {len(records)}")
    
    # Ensure the data directory exists
    data_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Save the fetched data to a CSV file in the data directory.
    csv_filename = os.path.join(data_dir, f"neo_feed_{overall_start.strftime('%Y-%m-%d')}_to_{overall_end.strftime('%Y-%m-%d')}.csv")
    save_csv(records, csv_filename)

if __name__ == "__main__":
    main()

