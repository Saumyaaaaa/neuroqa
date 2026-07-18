"""Root integration pipeline script to verify the NeuroQA architecture end-to-end."""

import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from neuroqa.models.transformer import EEGTransformer
from neuroqa.training.trainer import ModelTrainer
from neuroqa.evaluation.attention_rollout import generate_artifact_report

# Configure clean terminal logging layout
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("NeuroQA.Pipeline")

def run_mock_pipeline():
    logger.info("Initializing comprehensive NeuroQA system pipeline verification...")

    # 1. Setup Mock Tensor Datasets to simulate data loading segments
    # Input shape: (N_segments, n_channels, window_samples)
    n_samples, n_channels, window_samples = 32, 10, 512
    
    mock_x = torch.randn(n_samples, n_channels, window_samples).float()
    mock_y = torch.randint(0, 2, (n_samples,)).long()
    
    train_dataset = TensorDataset(mock_x, mock_y)
    val_dataset = TensorDataset(mock_x, mock_y)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    logger.info("Mock dataloaders generated successfully (%d execution batches).", len(train_loader))

    # 2. Initialize Model and Trainer Ecosystem
    model = EEGTransformer(n_channels=n_channels, window_samples=window_samples)
    trainer = ModelTrainer(model=model, train_loader=train_loader, val_loader=val_loader, lr=1e-3)

    # 3. Simulate a single complete training and validation epoch execution pass
    logger.info("Starting structural optimization loop execution step...")
    epoch_loss = trainer.train_epoch()
    val_loss, val_acc = trainer.evaluate()
    
    logger.info("Epoch completed -> Train Loss: %.4f | Val Loss: %.4f | Val Accuracy: %.2f%%", 
                epoch_loss, val_loss, val_acc * 100)

    # 4. Generate custom diagnostic explainability report on a single array slice
    logger.info("Triggering evaluation explainability report generator...")
    sample_segment = np.random.randn(n_channels, window_samples).astype("float32")
    channels = [f"EEG_{i+1}" for i in range(n_channels)]
    
    report = generate_artifact_report(model, sample_segment, channel_names=channels, sfreq=256)
    
    print("\n" + "="*50)
    print("         SYSTEM EXECUTION REPORT SUCCESS        ")
    print("="*50)
    print(f"Classification Verdict : {report['prediction'].upper()}")
    print(f"Confidence Level       : {report['confidence']:.4%}")
    print("Top Localized Artifact Time Windows (Seconds):")
    for start, end, weight in report['top_time_windows']:
        print(f"  - [{start:.3f}s to {end:.3f}s] -> Spatial Importance: {weight:.2%}")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_mock_pipeline()