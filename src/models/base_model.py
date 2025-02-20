from abc import ABC, abstractmethod
import torch
import numpy as np
from typing import Dict, Any

class BaseModel(ABC):
    """Base class for all models"""
    
    @abstractmethod
    def train(self, X: torch.Tensor, y: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: torch.Tensor) -> np.ndarray:
        """Make predictions"""
        pass
    
    @abstractmethod
    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """Evaluate model performance"""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk"""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk"""
        pass