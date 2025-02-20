import os
from models import TemporalNEOModel
from models import NEODataPreprocessor, load_training_data
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from models.config.model_config import TEMPORAL_MODEL_CONFIG, TRAINING_START_DATE, PREDICTION_HORIZON

def plot_training_history(history: dict, save_path: str):
    """Plot and save training history"""
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history['train_mae'], label='Train MAE')
    plt.plot(history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Create preprocessor
    preprocessor = NEODataPreprocessor()
    
    # Load training data
    print("Loading training data...")
    main_data, temporal_features = load_training_data(
        TRAINING_START_DATE,
        PREDICTION_HORIZON
    )
    
    # Prepare sequences for training
    print("Preparing sequences...")
    X, y = preprocessor.prepare_sequences(
        temporal_features,
        TEMPORAL_MODEL_CONFIG,
        train=True
    )
    
    # Initialize and train model
    print("Training model...")
    model = TemporalNEOModel(TEMPORAL_MODEL_CONFIG)
    history = model.train(X, y)
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("models", "saved", f"temporal_model_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the model and preprocessor
    model.save(save_dir)
    torch.save(preprocessor.scaler, os.path.join(save_dir, 'scaler.pth'))
    
    # Plot and save training history
    plot_training_history(
        history,
        os.path.join(save_dir, 'training_history.png')
    )
    
    print(f"Model saved to {save_dir}")

if __name__ == "__main__":
    main()