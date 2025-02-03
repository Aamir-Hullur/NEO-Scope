import os
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client

def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    df.drop_duplicates(subset=['id', 'close_approach_date'], inplace=True)
    
    numeric_cols = [
        'absolute_magnitude_h', 
        'estimated_diameter_min_km', 
        'estimated_diameter_max_km', 
        'relative_velocity_kph', 
        'miss_distance_km'
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=['absolute_magnitude_h', 'estimated_diameter_min_km','estimated_diameter_max_km'], inplace=True)

    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df['close_approach_date'] = pd.to_datetime(df['close_approach_date'], errors='coerce')
    df['estimated_diameter_avg_km'] = (df['estimated_diameter_min_km'] + df['estimated_diameter_max_km']) / 2.0
    return df

def get_supabase_client() -> Client:
    load_dotenv()  
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Supabase URL or KEY not found in .env file.")
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return supabase

def save_to_supabase(df: pd.DataFrame, batch_size=1000):
    supabase = get_supabase_client()  

    df["close_approach_date"] = df["close_approach_date"].astype(str)

    df["record_id"] = df["id"].astype(str) + "_" + df["close_approach_date"]

    df = df.where(pd.notna(df), None)

    data = df.to_dict(orient="records")

    total_records = len(data)
    for i in range(0, total_records, batch_size):
        batch = data[i:i + batch_size]  
        response = supabase.table("neo_data").upsert(batch).execute()

        response_dict = response.model_dump()
        if response_dict.get("data") is None:
            print("Error inserting data:", response_dict)
        else:
            print("Data inserted successfully. Response data:")

    print(f"Successfully inserted {total_records} records in batches.")



def main():
    data_dir = os.path.join(os.getcwd(), "data")
    input_file = os.path.join(data_dir, "neo_feed_2004-01-01_to_2025-01-01.csv")

    df = load_data(input_file)

    df = clean_data(df)

    df = feature_engineering(df)

    save_to_supabase(df)

if __name__ == "__main__":
    main()