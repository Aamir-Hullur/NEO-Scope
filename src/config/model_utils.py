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

def load_training_data(start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load training data from Supabase"""
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
    
    return main_data, temporal_features

class NEODataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def prepare_sequences(self, 
                         df: pd.DataFrame, 
                         config: Dict,
                         train: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare sequence data for training or prediction
        """
        # Sort by NEO ID and date
        df = df.sort_values(['id', 'close_approach_date'])
        
        sequences = []
        targets = []
        
        # Create sequences for each NEO
        for neo_id in df['id'].unique():
            neo_data = df[df['id'] == neo_id][config['features']].values
            
            if len(neo_data) >= config['sequence_length'] + 1:
                for i in range(len(neo_data) - config['sequence_length']):
                    seq = neo_data[i:i+config['sequence_length']]
                    target = neo_data[i+config['sequence_length']][
                        [config['features'].index(f) for f in config['target_features']]
                    ]
                    sequences.append(seq)
                    targets.append(target)
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        # Scale features
        if train:
            # Reshape for scaling
            seq_shape = sequences.shape
            sequences = self.scaler.fit_transform(
                sequences.reshape(-1, sequences.shape[-1])
            ).reshape(seq_shape)
        else:
            # Use pre-fitted scaler for prediction data
            seq_shape = sequences.shape
            sequences = self.scaler.transform(
                sequences.reshape(-1, sequences.shape[-1])
            ).reshape(seq_shape)
        
        # Convert to PyTorch tensors
        sequences = torch.FloatTensor(sequences)
        targets = torch.FloatTensor(targets)
        
        return sequences, targets