import os
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client
from typing import Dict, Tuple
from sklearn.preprocessing import StandardScaler

class NEOFeatureEngineer:
    def __init__(self):
        self.temporal_scaler = StandardScaler()
        self.physical_scaler = StandardScaler()
    
    def create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['close_approach_date'] = pd.to_datetime(df['close_approach_date'], errors='coerce')
        df['estimated_diameter_avg_km'] = (df['estimated_diameter_min_km'] + df['estimated_diameter_max_km']) / 2.0
        return df

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.sort_values(['id', 'close_approach_date'])
        
        df['year'] = df['close_approach_date'].dt.year
        df['month'] = df['close_approach_date'].dt.month
        df['day_of_year'] = df['close_approach_date'].dt.dayofyear
        df['days_since_first_obs'] = df.groupby('id')['close_approach_date'].transform(lambda x: (x - x.min()).dt.days)
        df['velocity_change'] = df.groupby('id')['relative_velocity_kph'].transform(lambda x: x.diff())
        df['miss_distance_change'] = df.groupby('id')['miss_distance_km'].transform(lambda x: x.diff())

        return df

    def create_physical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create physical characteristic features
        """
        df = df.copy()

        df['estimated_volume_km3'] = (4/3) * np.pi * (df['estimated_diameter_avg_km']/2)**3
        df['estimated_surface_area_km2'] = 4 * np.pi * (df['estimated_diameter_avg_km']/2)**2
        df['hazard_score'] = (
            df['is_potentially_hazardous_asteroid'].astype(int) * 0.7 +
            df['is_sentry_object'].astype(int) * 0.3
        )
        
        return df

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from CSV file
    """
    df = pd.read_csv(file_path)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataframe by removing duplicates and handling missing values
    """
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

def process_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Process data with enhanced error handling and float value cleaning
    """
    feature_engineer = NEOFeatureEngineer()
    
    df = clean_data(df)
    
    try:
        df = feature_engineer.create_basic_features(df)
        df = feature_engineer.create_temporal_features(df)
        df = feature_engineer.create_physical_features(df)
        df = df.replace([np.inf, -np.inf], np.nan)
        
        model_data = {
            'temporal_features': df[[
                'velocity_change',
                'miss_distance_change',
                'days_since_first_obs',
                'year', 'month', 'day_of_year'
            ]].replace([np.inf, -np.inf], np.nan),
            
            'physical_features': df[[
                'estimated_volume_km3',
                'estimated_surface_area_km2',
                'hazard_score',
                'estimated_diameter_avg_km'
            ]].replace([np.inf, -np.inf], np.nan)
        }
        
        return df, model_data
    except Exception as e:
        print(f"Error in process_data: {str(e)}")
        raise

def get_supabase_client() -> Client:
    """
    Get Supabase client using environment variables
    """
    load_dotenv()
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Supabase URL or KEY not found in .env file.")
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def handle_float_value(value):
    """
    Handle float values to ensure they are database-compliant
    """
    if pd.isna(value) or np.isinf(value):
        return None
    return float(value)

def save_to_supabase(df: pd.DataFrame, model_data: Dict, batch_size=1000):
    """
    Save processed data and model features to separate tables in Supabase
    with proper handling of float values
    """
    supabase = get_supabase_client()

    df_supabase = df.copy()
    df_supabase["close_approach_date"] = df_supabase["close_approach_date"].astype(str)
    df_supabase["record_id"] = df_supabase["id"].astype(str) + "_" + df_supabase["close_approach_date"]
    df_supabase = df_supabase.where(pd.notna(df_supabase), None)
    
    # Save main data in batches
    data = df_supabase.to_dict(orient="records")
    total_records = len(data)
    for i in range(0, total_records, batch_size):
        batch = data[i:i + batch_size]
        try:
            response = supabase.table("neo_data").upsert(batch).execute()
            print(f"Successfully inserted batch {i//batch_size + 1} of main data")
        except Exception as e:
            print(f"Error in batch {i//batch_size + 1} of main data:", str(e))
    
    temporal_features = []
    for idx, row in model_data['temporal_features'].iterrows():
        try:
            record = {
                "record_id": str(df.loc[idx, "id"]) + "_" + str(df.loc[idx, "close_approach_date"].date()),
                "velocity_change": handle_float_value(row["velocity_change"]),
                "miss_distance_change": handle_float_value(row["miss_distance_change"]),
                "days_since_first_obs": int(row["days_since_first_obs"]) if pd.notna(row["days_since_first_obs"]) else None,
                "year": int(row["year"]),
                "month": int(row["month"]),
                "day_of_year": int(row["day_of_year"])
            }
            temporal_features.append(record)
        except Exception as e:
            print(f"Error preparing temporal record for index {idx}: {str(e)}")
            continue

    # Save temporal features in batches
    for i in range(0, len(temporal_features), batch_size):
        batch = temporal_features[i:i + batch_size]
        try:
            response = supabase.table("neo_temporal_features").upsert(batch).execute()
            print(f"Successfully saved temporal features batch {i//batch_size + 1}")
        except Exception as e:
            print(f"Error saving temporal features batch {i//batch_size + 1}:", str(e))

    physical_features = []
    for idx, row in model_data['physical_features'].iterrows():
        try:
            record = {
                "record_id": str(df.loc[idx, "id"]) + "_" + str(df.loc[idx, "close_approach_date"].date()),
                "estimated_volume_km3": handle_float_value(row["estimated_volume_km3"]),
                "estimated_surface_area_km2": handle_float_value(row["estimated_surface_area_km2"]),
                "hazard_score": handle_float_value(row["hazard_score"])
            }
            physical_features.append(record)
        except Exception as e:
            print(f"Error preparing physical record for index {idx}: {str(e)}")
            continue

    # Save physical features in batches
    for i in range(0, len(physical_features), batch_size):
        batch = physical_features[i:i + batch_size]
        try:
            response = supabase.table("neo_physical_features").upsert(batch).execute()
            print(f"Successfully saved physical features batch {i//batch_size + 1}")
        except Exception as e:
            print(f"Error saving physical features batch {i//batch_size + 1}:", str(e))

def main():
    data_dir = os.path.join(os.getcwd(), "data")
    input_file = os.path.join(data_dir, "neo_feed_2004-01-01_to_2025-01-01.csv")

    df = load_data(input_file)
    
    processed_df, model_data = process_data(df)
    
    # Save to Supabase
    save_to_supabase(processed_df, model_data)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()