import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Tuple
import os
import json
from .base_model import BaseModel


class NEODataset(Dataset):
    """Custom Dataset for NEO sequences"""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    """LSTM model architecture"""
    def __init__(self, input_size: int, hidden_size: int, dropout_rate: float):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lstm2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            batch_first=True
        )
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size // 2, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 2)  # 2 target features
        
    def forward(self, x):
        # First LSTM layer
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout1(lstm1_out)
        
        # Second LSTM layer
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.dropout2(lstm2_out[:, -1, :])  # Take only last sequence output
        
        # Dense layers
        fc1_out = self.relu(self.fc1(lstm2_out))
        output = self.fc2(fc1_out)
        
        return output

class TemporalNEOModel(BaseModel):
    """
    Temporal Pattern Analysis model for NEO trajectory prediction using LSTM (PyTorch)
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the model with given configuration
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate']
        )
        self.history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}
        
    def _build_model(self) -> nn.Module:
        """Build the LSTM model architecture"""
        return LSTMModel(
            input_size=len(self.config['features']),
            hidden_size=self.config['hidden_units'],
            dropout_rate=self.config['dropout_rate']
        )
    
    def _calculate_mae(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        """Calculate Mean Absolute Error"""
        return torch.mean(torch.abs(y_pred - y_true)).item()
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Train the model"""
        # Create data loaders
        dataset = NEODataset(X, y)
        train_size = int((1 - self.config['validation_split']) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size']
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(self.config['epochs']):
            # Training phase
            self.model.train()
            train_loss = 0
            train_mae = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                train_mae += self._calculate_mae(outputs, batch_y)
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_mae = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    val_loss += self.criterion(outputs, batch_y).item()
                    val_mae += self._calculate_mae(outputs, batch_y)
            
            # Calculate average metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_train_mae = train_mae / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_val_mae = val_mae / len(val_loader)
            
            # Update history
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            self.history['train_mae'].append(avg_train_mae)
            self.history['val_mae'].append(avg_val_mae)
            
            # Print progress
            print(f'Epoch {epoch+1}/{self.config["epochs"]}:')
            print(f'Train Loss: {avg_train_loss:.4f}, Train MAE: {avg_train_mae:.4f}')
            print(f'Val Loss: {avg_val_loss:.4f}, Val MAE: {avg_val_mae:.4f}')
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 5:  # patience of 5 epochs
                    print("Early stopping triggered")
                    break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            predictions = self.model(X_tensor)
        return predictions.cpu().numpy()
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
            loss = self.criterion(predictions, y_tensor).item()
            mae = self._calculate_mae(predictions, y_tensor)
        
        return {
            'loss': loss,
            'mae': mae
        }
    
    def save(self, path: str) -> None:
        """Save model and configuration"""
        os.makedirs(path, exist_ok=True)
        
        # Save model state
        torch.save(self.model.state_dict(), os.path.join(path, 'model_state.pth'))
        
        # Save configuration
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(self.config, f)
        
        # Save training history
        with open(os.path.join(path, 'history.json'), 'w') as f:
            json.dump(self.history, f)
    
    def load(self, path: str) -> None:
        """Load model and configuration"""
        # Load configuration
        with open(os.path.join(path, 'config.json'), 'r') as f:
            self.config = json.load(f)
        
        # Rebuild model with loaded configuration
        self.model = self._build_model().to(self.device)
        
        # Load model state
        self.model.load_state_dict(
            torch.load(os.path.join(path, 'model_state.pth'))
        )
        
        # Load history if exists
        history_path = os.path.join(path, 'history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                self.history = json.load(f)