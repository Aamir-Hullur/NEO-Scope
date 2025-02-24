import torch

TEMPORAL_MODEL_CONFIG = {
    'sequence_length': 5,
    'hidden_units': 256,  # Increased from 128
    'num_attention_heads': 8,  # Increased from 4
    'dropout_rate': 0.4,  # Increased from 0.3
    'learning_rate': 0.0003,  # Further decreased
    'batch_size': 64,  # Increased from 32
    'epochs': 150,  # Increased from 100
    'validation_split': 0.2,
    'patience': 15,  # Increased from 10
    'gradient_clip': 1.0,  # Added gradient clipping
    'features': [
        'velocity_change',
        'miss_distance_change',
        'days_since_first_obs'
    ],
    'target_features': [
        'velocity_change',
        'miss_distance_change'
    ]
}

# Date range for prediction
PREDICTION_HORIZON = '2025-12-31'
TRAINING_START_DATE = '2004-01-01'