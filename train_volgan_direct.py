#!/usr/bin/env python3
"""
VolGAN Training Script - Direct Training with Processed B3 Data
Uses the massive B3 dataset we just processed
"""
import sys
from pathlib import Path
import argparse
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import VolGAN components
from volgan_model import VolGANTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('volgan_training_direct.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_and_prepare_data():
    """Load the processed B3 data and prepare it for VolGAN training"""
    logger.info("📥 Loading processed B3 data...")
    
    # Load training data
    training_file = Path('data/processed/volgan_training_data.csv')
    if not training_file.exists():
        raise FileNotFoundError(f"Training data not found: {training_file}")
    
    training_data = pd.read_csv(training_file)
    logger.info(f"✅ Loaded training data: {len(training_data):,} records")
    
    # Load test data
    test_file = Path('data/processed/volgan_test_data.csv')
    if test_file.exists():
        test_data = pd.read_csv(test_file)
        logger.info(f"✅ Loaded test data: {len(test_data):,} records")
    else:
        test_data = None
        logger.warning("⚠️  Test data not found, will use validation split from training data")
    
    # Prepare features for VolGAN
    logger.info("🔧 Preparing features for VolGAN...")
    
    # Use normalized features for training
    X_train = training_data[['k_normalized', 'T_normalized', 'iv_normalized']].values
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    
    logger.info(f"✅ Training features shape: {X_train_tensor.shape}")
    
    # Create a simple dataset that returns single tensors
    class SingleTensorDataset(torch.utils.data.Dataset):
        def __init__(self, tensor):
            self.tensor = tensor
        
        def __len__(self):
            return len(self.tensor)
        
        def __getitem__(self, idx):
            return (self.tensor[idx],)  # Return as tuple with one element
    
    train_dataset = SingleTensorDataset(X_train_tensor)
    
    return train_dataset, test_data

def train_volgan_model(train_dataset, args):
    """Train the VolGAN model"""
    logger.info("🚀 Starting VolGAN training...")
    
    # Initialize trainer
    trainer = VolGANTrainer(
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        generator_lr=args.lr,
        discriminator_lr=args.lr
    )
    
    logger.info(f"✅ Trainer initialized on device: {trainer.device}")
    logger.info(f"📊 Training data: {len(train_dataset):,} samples")
    
    # Create dataloader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0  # Set to 0 for compatibility
    )
    
    # Use the built-in train method
    logger.info(f"🎯 Starting training for {args.epochs} epochs...")
    
    trainer.train(
        train_loader=train_loader,
        epochs=args.epochs,
        save_interval=args.save_interval,
        model_dir='models'
    )
    
    logger.info("✅ Training completed!")
    
    return trainer

def generate_samples(trainer, num_samples=1000):
    """Generate sample volatility surfaces using the trained VolGAN"""
    logger.info(f"🎨 Generating {num_samples} sample volatility surfaces...")
    
    trainer.generator.eval()
    with torch.no_grad():
        # Generate random latent vectors
        z = torch.randn(num_samples, trainer.latent_dim).to(trainer.device)
        
        # Generate samples
        generated_samples = trainer.generator(z)
        
        # Convert back to original scale (if needed)
        # For now, return normalized samples
        return generated_samples.cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description='VolGAN Direct Training with B3 Data')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--latent-dim', type=int, default=100, help='Latent dimension')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--save-interval', type=int, default=10, help='Save model every N epochs')
    
    args = parser.parse_args()
    
    logger.info("🎯 VolGAN Training with Processed B3 Data")
    logger.info("=" * 60)
    logger.info(f"Training epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Latent dimension: {args.latent_dim}")
    logger.info(f"Learning rate: {args.lr}")
    
    try:
        # Load and prepare data
        train_dataset, test_data = load_and_prepare_data()
        
        # Create models directory
        Path("models").mkdir(exist_ok=True)
        
        # Train the model
        trainer = train_volgan_model(train_dataset, args)
        
        # Generate samples
        samples = generate_samples(trainer, num_samples=1000)
        
        # Save samples
        np.save("results/generated_samples.npy", samples)
        logger.info("💾 Generated samples saved")
        
        logger.info("✅ VolGAN training completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
