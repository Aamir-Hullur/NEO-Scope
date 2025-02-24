import pandas as pd
import numpy as np
import torch
from typing import Tuple, Dict
from supabase import create_client, Client
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
import os


def get_supabase_client() -> Client:
    """Get Supabase client"""
    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    return create_client(url, key)

def handle_missing_values(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Handle missing values in the DataFrame
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to check for missing values
    
    Returns:
        pd.DataFrame: DataFrame with handled missing values
    """
    total_rows = len(df)
    
    # Print missing value statistics
    print("\nMissing Value Analysis:")
    for col in columns:
        missing = df[col].isna().sum()
        if missing > 0:
            print(f"{col}: {missing} missing values ({(missing/total_rows)*100:.2f}%)")
    
    # Handle missing values based on column type
    for col in columns:
        if df[col].dtype in ['float64', 'int64']:
            # For numeric columns, fill with median
            df[col] = df[col].fillna(df[col].median())
        else:
            # For categorical/other columns, fill with mode
            df[col] = df[col].fillna(df[col].mode()[0])
    
    return df

def load_training_data(start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training data from Supabase with enhanced data validation
    
    Args:
        start_date (str): Start date for training data
        end_date (str): End date for training data
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (main_data, temporal_features)
    """
    supabase = get_supabase_client()
    
    # Load main data
    response = supabase.table("neo_data")\
        .select("*")\
        .gte("close_approach_date", start_date)\
        .lte("close_approach_date", end_date)\
        .execute()
    main_data = pd.DataFrame(response.data)
    
    # Load temporal features
    response = supabase.table("neo_temporal_features")\
        .select("*")\
        .execute()
    temporal_features = pd.DataFrame(response.data)
    
    # Print data info for debugging
    print(f"\nLoaded {len(main_data)} main records")
    print(f"Loaded {len(temporal_features)} temporal records")
    
    if temporal_features.empty or main_data.empty:
        raise ValueError(f"No data found between {start_date} and {end_date}")
    
    # Handle missing values in main_data
    main_data_columns = [
        'absolute_magnitude_h', 
        'estimated_diameter_min_km',
        'estimated_diameter_max_km',
        'relative_velocity_kph',
        'miss_distance_km'
    ]
    main_data = handle_missing_values(main_data, main_data_columns)
    
    # Handle missing values in temporal_features
    temporal_columns = [
        'velocity_change',
        'miss_distance_change',
        'days_since_first_obs',
        'year',
        'month',
        'day_of_year'
    ]
    temporal_features = handle_missing_values(temporal_features, temporal_columns)
    
    # Extract original asteroid ID from record_id in temporal features
    if 'record_id' in temporal_features.columns:
        temporal_features['id'] = temporal_features['record_id'].str.split('_').str[0]
        print(f"\nNumber of unique asteroids in temporal features: {temporal_features['id'].nunique()}")
    
    return main_data, temporal_features

class NEODataPreprocessor:
    def __init__(self):
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self._is_fitted = False

    def save(self, path: str):
        """Save preprocessor state"""
        if not self._is_fitted:
            raise ValueError("Preprocessor must be fitted before saving")
        
        scalers = {
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler
        }
        torch.save(scalers, path)

    def load(self, path: str):
        """Load preprocessor state"""
        scalers = torch.load(path)
        self.feature_scaler = scalers['feature_scaler']
        self.target_scaler = scalers['target_scaler']
        self._is_fitted = True

    def prepare_sequences(self, 
                         df: pd.DataFrame, 
                         config: Dict,
                         train: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare sequence data for training or prediction with improved scaling
        """
        print("\nDataFrame Info:")
        print(df.info())

        # Extract close_approach_date from record_id
        df['close_approach_date'] = df['record_id'].str.split('_').str[1]
        df['close_approach_date'] = pd.to_datetime(df['close_approach_date'])
        df = df.sort_values(['id', 'close_approach_date'])

        print(f"\nTotal unique asteroids: {df['id'].nunique()}")
        print(f"Date range: {df['close_approach_date'].min()} to {df['close_approach_date'].max()}")

        # Log transform numerical columns before scaling
        numerical_columns = ['velocity_change', 'miss_distance_change']
        for col in numerical_columns:
            # Add small constant to handle zeros and negative values
            df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]) + 1)

        sequences = []
        targets = []

        asteroids_processed = 0
        for neo_id in df['id'].unique():
            neo_data = df[df['id'] == neo_id][config['features']].values

            if len(neo_data) >= config['sequence_length'] + 1:
                asteroids_processed += 1
                for i in range(len(neo_data) - config['sequence_length']):
                    seq = neo_data[i:i+config['sequence_length']]
                    target = neo_data[i+config['sequence_length']][
                        [config['features'].index(f) for f in config['target_features']]
                    ]
                    sequences.append(seq)
                    targets.append(target)

            if asteroids_processed % 100 == 0 and asteroids_processed > 0:
                print(f"Processed {asteroids_processed} asteroids...")

        if not sequences:
            raise ValueError(
                f"No valid sequences could be created. Need at least {config['sequence_length'] + 1} "
                f"observations per asteroid."
            )

        sequences = np.array(sequences)
        targets = np.array(targets)

        print(f"\nCreated {len(sequences)} sequences from {asteroids_processed} asteroids")
        print(f"Average sequences per asteroid: {len(sequences)/asteroids_processed:.2f}")
        print(f"Sequence shape: {sequences.shape}")
        print(f"Target shape: {targets.shape}")

        # Scale features and targets separately
        if train:
            seq_shape = sequences.shape
            sequences = self.feature_scaler.fit_transform(
                sequences.reshape(-1, sequences.shape[-1])
            ).reshape(seq_shape)
            
            targets = self.target_scaler.fit_transform(targets)
            self._is_fitted = True
        else:
            if not self._is_fitted:
                raise ValueError("Preprocessor must be fitted before transform")
            seq_shape = sequences.shape
            sequences = self.feature_scaler.transform(
                sequences.reshape(-1, sequences.shape[-1])
            ).reshape(seq_shape)
            targets = self.target_scaler.transform(targets)

        return torch.FloatTensor(sequences), torch.FloatTensor(targets)