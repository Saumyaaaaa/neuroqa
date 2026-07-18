"""Orchestration training infrastructure for the NeuroQA artifact detection model."""

from __future__ import annotations
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from neuroqa.models.transformer import EEGTransformer

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Manages model training loops, metric computing, and weight optimization checkpoints."""

    def __init__(
        self,
        model: EEGTransformer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str | None = None
    ):
        """Initializes the training runtime.

        Args:
            model: Target EEGTransformer instance.
            train_loader: DataLoader instance providing training instances.
            val_loader: DataLoader instance providing validation metrics.
            lr: Learning rate constant optimization factor. Defaults to 1e-3.
            weight_decay: L2 regularization weight decay factor. Defaults to 1e-4.
            device: Runtime hardware target configuration ('cuda' or 'cpu').
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Configure weight-balanced optimization criteria
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        logger.info("ModelTrainer initialized on operational device target: %s", self.device)

    def train_epoch(self) -> float:
        """Runs a complete operational pass through training samples.

        Returns:
            float: Average epoch cross-entropy loss value.
        """
        self.model.train()
        running_loss = 0.0
        
        for batch_x, batch_y in self.train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(batch_x)
            loss = self.criterion(logits, batch_y)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * batch_x.size(0)
            
        return running_loss / len(self.train_loader.dataset)

    def evaluate(self) -> tuple[float, float]:
        """Runs validation evaluation loops over cross-validation splits.

        Returns:
            tuple[float, float]: Consolidated tuple consisting of (average_loss, accuracy_score).
        """
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)
                
                running_loss += loss.item() * batch_x.size(0)
                predictions = torch.argmax(logits, dim=-1)
                
                correct_predictions += (predictions == batch_y).sum().item()
                total_samples += batch_y.size(0)
                
        val_loss = running_loss / len(self.val_loader.dataset)
        val_acc = correct_predictions / total_samples if total_samples > 0 else 0.0
        return val_loss, val_acc