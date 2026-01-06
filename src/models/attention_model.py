"""
Attention-based Neural Network for tweet trading prediction.

ENHANCED VERSION:
- Added batch normalization for better training stability
- Increased weight_decay to 0.05 for stronger regularization
- Uses config.settings for hyperparameters
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.models.base_model import BaseTradingModel
from config.settings import (
    ATTENTION_HIDDEN_DIM, ATTENTION_NUM_HEADS, ATTENTION_NUM_LAYERS,
    ATTENTION_DROPOUT, ATTENTION_LEARNING_RATE, ATTENTION_WEIGHT_DECAY,
    ATTENTION_BATCH_SIZE, ATTENTION_EPOCHS, ATTENTION_EARLY_STOPPING_PATIENCE
)

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Calculate scaled dot-product attention."""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, V)
        return output, attention_weights

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, d_k)."""
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, x):
        batch_size = x.shape[0]

        # Add sequence dimension if not present
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        residual = x

        # Linear projections
        Q = self.split_heads(self.W_q(x), batch_size)
        K = self.split_heads(self.W_k(x), batch_size)
        V = self.split_heads(self.W_v(x), batch_size)

        # Attention
        attention_output, _ = self.scaled_dot_product_attention(Q, K, V)

        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, -1, self.d_model)

        # Output projection
        output = self.W_o(attention_output)
        output = self.dropout(output)

        # Residual connection and layer norm
        output = self.layer_norm(output + residual)

        # Remove sequence dimension
        if output.shape[1] == 1:
            output = output.squeeze(1)

        return output


class AttentionTradingModel(nn.Module):
    """
    Deep neural network with multi-head attention for trading prediction.

    ENHANCED VERSION:
    - Added BatchNorm1d for better training stability
    - Stronger regularization with dropout and weight decay
    """

    def __init__(self, input_dim: int, hidden_dim: int = 512, num_heads: int = 8,
                 num_layers: int = 3, dropout: float = 0.2, use_batch_norm: bool = True):
        super().__init__()

        self.use_batch_norm = use_batch_norm

        # Enhanced input projection with batch norm
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Multiple attention layers
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Enhanced feed-forward layers with batch norm
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2) if use_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim)
            )
            for _ in range(num_layers)
        ])

        # Enhanced prediction head with batch norm
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, x):
        # Input projection
        x = self.input_projection(x)

        # Attention layers with feed-forward
        for attention, ff in zip(self.attention_layers, self.ff_layers):
            # Attention block
            attn_out = attention(x)

            # Feed-forward block
            ff_out = ff(attn_out)

            # Residual connection
            x = attn_out + ff_out

        # Prediction
        output = self.prediction_head(x)

        return output.squeeze(-1)


class AttentionModel(BaseTradingModel):
    """
    Wrapper for Attention-based trading model.
    """

    def __init__(self, name: str = "AttentionModel",
                 hidden_dim: int = ATTENTION_HIDDEN_DIM,
                 num_heads: int = ATTENTION_NUM_HEADS,
                 num_layers: int = ATTENTION_NUM_LAYERS,
                 dropout: float = ATTENTION_DROPOUT,
                 learning_rate: float = ATTENTION_LEARNING_RATE,
                 weight_decay: float = ATTENTION_WEIGHT_DECAY,
                 batch_size: int = ATTENTION_BATCH_SIZE,
                 epochs: int = ATTENTION_EPOCHS,
                 device: str = None,
                 use_batch_norm: bool = True):
        super().__init__(name)

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay  # Enhanced: now configurable
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_batch_norm = use_batch_norm  # Enhanced: batch normalization flag

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")
        print(f"Batch normalization: {'enabled' if use_batch_norm else 'disabled'}")
        print(f"Weight decay (L2 regularization): {self.weight_decay}")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Train the attention model."""
        print(f"\nTraining {self.name}...")
        print(f"Architecture: {self.num_layers} layers, {self.num_heads} heads, {self.hidden_dim} hidden dim")

        input_dim = X_train.shape[1]

        # Initialize model (with batch normalization)
        self.model = AttentionTradingModel(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout,
            use_batch_norm=self.use_batch_norm  # Enhanced: batch norm enabled
        ).to(self.device)

        # Print model size
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Optimizer and loss (Enhanced: stronger weight decay for regularization)
        optimizer = optim.AdamW(self.model.parameters(),
                               lr=self.learning_rate,
                               weight_decay=self.weight_decay)  # Enhanced: 0.05 from config
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        criterion = nn.MSELoss()

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = ATTENTION_EARLY_STOPPING_PATIENCE  # From config

        train_losses = []
        val_losses = []

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    predictions = self.model(X_batch)
                    loss = criterion(predictions, y_batch)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        self.model.load_state_dict(self.best_model_state)
        self.is_fitted = True

        # Calculate final metrics
        train_metrics = self.evaluate(X_train, y_train)
        val_metrics = self.evaluate(X_val, y_val)

        print(f"\nFinal Training Correlation: {train_metrics['correlation']:.4f}")
        print(f"Final Validation Correlation: {val_metrics['correlation']:.4f}")
        print(f"Directional Accuracy (Val): {val_metrics['directional_accuracy']:.4f}")

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'best_val_loss': best_val_loss
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)

            # Predict in batches for efficiency
            predictions = []
            for i in range(0, len(X_tensor), self.batch_size):
                batch = X_tensor[i:i+self.batch_size]
                pred = self.model(batch)
                predictions.append(pred.cpu().numpy())

            return np.concatenate(predictions)

if __name__ == "__main__":
    print("Attention model module loaded")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
