TEMPORAL_MODEL_CONFIG = {
    'sequence_length': 5,
    'hidden_units': 64,
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 50,
    'validation_split': 0.2,
    'features': [
        'velocity_change',
        'miss_distance_change',
        'days_since_first_obs'
    ],
    'target_features': [
        'velocity_change',
        'miss_distance_change'
    ],
    # PyTorch specific parameters
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,  # for data loading
    'pin_memory': True  # for faster data transfer to GPU
}

# Date range for prediction
PREDICTION_HORIZON = '2025-12-31'
TRAINING_START_DATE = '2004-01-01'