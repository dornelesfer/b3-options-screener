# src/volgan_model.py
"""
VolGAN Model Implementation
Based on Milena Vuletic's paper for volatility surface generation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)

class VolGANGenerator(nn.Module):
    """
    Generator network for VolGAN
    Generates implied volatility surfaces from noise
    """
    
    def __init__(self, latent_dim=100, hidden_dims=[256, 512, 256, 128]):
        super(VolGANGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # Build layers
        layers = []
        input_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)
            ])
            input_dim = hidden_dim
        
        # Output layer - generates (k, T, iv) coordinates
        layers.append(nn.Linear(input_dim, 3))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, z):
        """Forward pass"""
        return self.network(z)

class VolGANDiscriminator(nn.Module):
    """
    Discriminator network for VolGAN
    Distinguishes between real and generated IV surfaces
    """
    
    def __init__(self, input_dim=3, hidden_dims=[128, 256, 128, 64]):
        super(VolGANDiscriminator, self).__init__()
        
        self.hidden_dims = hidden_dims
        
        # Build layers
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3)
            ])
            current_dim = hidden_dim
        
        # Output layer - single value for real/fake classification
        layers.append(nn.Linear(current_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass"""
        return self.network(x)

class VolGANTrainer:
    """
    Trainer class for VolGAN model
    """
    
    def __init__(self, 
                 generator_lr=0.0002,
                 discriminator_lr=0.0002,
                 beta1=0.5,
                 beta2=0.999,
                 latent_dim=100,
                 batch_size=32,
                 device='auto'):
        
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self.generator = VolGANGenerator(latent_dim=latent_dim).to(self.device)
        self.discriminator = VolGANDiscriminator().to(self.device)
        
        # Initialize optimizers
        self.g_optimizer = optim.Adam(
            self.generator.parameters(), 
            lr=generator_lr, 
            betas=(beta1, beta2)
        )
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(), 
            lr=discriminator_lr, 
            betas=(beta1, beta2)
        )
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Training history
        self.g_losses = []
        self.d_losses = []
        self.g_losses_real = []
        self.g_losses_fake = []
        
    def prepare_data(self, data_path, test_split=0.2):
        """
        Prepare data for training
        
        Args:
            data_path: Path to the VolGAN training data CSV
            test_split: Fraction of data to use for testing
            
        Returns:
            train_loader, test_loader, data_stats
        """
        logger.info("Preparing training data...")
        
        # Load data
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} data points")
        
        # Extract features
        features = df[['k_normalized', 'T_normalized', 'iv_normalized']].values
        features = torch.tensor(features, dtype=torch.float32)
        
        # Split data
        n_test = int(len(features) * test_split)
        train_features = features[:-n_test]
        test_features = features[-n_test:]
        
        logger.info(f"Training samples: {len(train_features)}")
        logger.info(f"Test samples: {len(test_features)}")
        
        # Create data loaders
        train_dataset = TensorDataset(train_features)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        test_dataset = TensorDataset(test_features)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Calculate data statistics for denormalization
        data_stats = {
            'k_mean': df['k'].mean(),
            'k_std': df['k'].std(),
            'T_mean': df['T'].mean(),
            'T_std': df['T'].std(),
            'iv_mean': df['iv'].mean(),
            'iv_std': df['iv'].std()
        }
        
        return train_loader, test_loader, data_stats
    
    def train_step(self, real_data):
        """
        Single training step
        
        Args:
            real_data: Batch of real IV surface points
            
        Returns:
            g_loss, d_loss
        """
        batch_size = real_data.size(0)
        real_data = real_data.to(self.device)
        
        # Labels
        real_labels = torch.ones(batch_size, 1).to(self.device)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)
        
        # Train Discriminator
        self.d_optimizer.zero_grad()
        
        # Real data
        real_outputs = self.discriminator(real_data)
        d_loss_real = self.criterion(real_outputs, real_labels)
        
        # Fake data
        noise = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_data = self.generator(noise)
        fake_outputs = self.discriminator(fake_data.detach())
        d_loss_fake = self.criterion(fake_outputs, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.d_optimizer.step()
        
        # Train Generator
        self.g_optimizer.zero_grad()
        
        # Generate fake data
        noise = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_data = self.generator(noise)
        fake_outputs = self.discriminator(fake_data)
        
        g_loss = self.criterion(fake_outputs, real_labels)
        g_loss.backward()
        self.g_optimizer.step()
        
        return g_loss.item(), d_loss.item()
    
    def train(self, train_loader, epochs=100, save_interval=10, model_dir='models'):
        """
        Train the VolGAN model
        
        Args:
            train_loader: Training data loader
            epochs: Number of training epochs
            save_interval: Save model every N epochs
            model_dir: Directory to save models
        """
        logger.info(f"Starting training for {epochs} epochs...")
        
        model_dir = Path(model_dir)
        model_dir.mkdir(exist_ok=True)
        
        for epoch in range(epochs):
            epoch_g_losses = []
            epoch_d_losses = []
            
            for batch_idx, (real_data,) in enumerate(train_loader):
                g_loss, d_loss = self.train_step(real_data)
                epoch_g_losses.append(g_loss)
                epoch_d_losses.append(d_loss)
                
                if batch_idx % 100 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, "
                              f"G Loss: {g_loss:.4f}, D Loss: {d_loss:.4f}")
            
            # Record epoch losses
            avg_g_loss = np.mean(epoch_g_losses)
            avg_d_loss = np.mean(epoch_d_losses)
            self.g_losses.append(avg_g_loss)
            self.d_losses.append(avg_d_loss)
            
            logger.info(f"Epoch {epoch+1}/{epochs} - "
                       f"Avg G Loss: {avg_g_loss:.4f}, "
                       f"Avg D Loss: {avg_d_loss:.4f}")
            
            # Save model periodically
            if (epoch + 1) % save_interval == 0:
                self.save_model(model_dir / f"volgan_epoch_{epoch+1}.pth")
        
        # Save final model
        self.save_model(model_dir / "volgan_final.pth")
        logger.info("Training completed!")
    
    def save_model(self, path):
        """Save model state"""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'g_losses': self.g_losses,
            'd_losses': self.d_losses,
            'latent_dim': self.latent_dim,
            'batch_size': self.batch_size
        }, path)
        logger.info(f"Model saved to: {path}")
    
    def load_model(self, path):
        """Load model state"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        self.g_losses = checkpoint['g_losses']
        self.d_losses = checkpoint['d_losses']
        
        logger.info(f"Model loaded from: {path}")
    
    def generate_surfaces(self, n_surfaces=10, n_points=100, data_stats=None):
        """
        Generate synthetic IV surfaces
        
        Args:
            n_surfaces: Number of surfaces to generate
            n_points: Points per surface
            data_stats: Statistics for denormalization
            
        Returns:
            List of generated surfaces
        """
        self.generator.eval()
        
        surfaces = []
        
        with torch.no_grad():
            for i in range(n_surfaces):
                # Generate noise
                noise = torch.randn(n_points, self.latent_dim).to(self.device)
                
                # Generate surface
                generated = self.generator(noise)
                
                # Denormalize if stats provided
                if data_stats:
                    generated_np = generated.cpu().numpy()
                    
                    k_denorm = generated_np[:, 0] * data_stats['k_std'] + data_stats['k_mean']
                    T_denorm = generated_np[:, 1] * data_stats['T_std'] + data_stats['T_mean']
                    iv_denorm = generated_np[:, 2] * data_stats['iv_std'] + data_stats['iv_mean']
                    
                    surface = pd.DataFrame({
                        'k': k_denorm,
                        'T': T_denorm,
                        'iv': iv_denorm
                    })
                else:
                    surface = pd.DataFrame(generated.cpu().numpy(), 
                                        columns=['k', 'T', 'iv'])
                
                surfaces.append(surface)
        
        return surfaces
    
    def plot_training_history(self, save_path=None):
        """Plot training loss history"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.g_losses, label='Generator Loss')
        plt.plot(self.d_losses, label='Discriminator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.g_losses, label='Generator Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Generator Loss Detail')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to: {save_path}")
        
        plt.show()
    
    def plot_generated_surface(self, surface, save_path=None):
        """Plot a single generated IV surface"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(surface['k'], surface['T'], surface['iv'], 
                           c=surface['iv'], cmap='viridis', s=50)
        
        ax.set_xlabel('Log Moneyness (k)')
        ax.set_ylabel('Time to Maturity (T)')
        ax.set_zlabel('Implied Volatility (σ)')
        ax.set_title('Generated IV Surface')
        
        plt.colorbar(scatter, ax=ax, label='IV')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Surface plot saved to: {save_path}")
        
        plt.show()

def main():
    """Example usage"""
    print("🚀 VolGAN Model Implementation")
    print("=" * 40)
    
    # Initialize trainer
    trainer = VolGANTrainer(
        latent_dim=100,
        batch_size=32,
        generator_lr=0.0002,
        discriminator_lr=0.0002
    )
    
    print(f"Device: {trainer.device}")
    print(f"Generator parameters: {sum(p.numel() for p in trainer.generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in trainer.discriminator.parameters()):,}")
    
    # Check if training data exists
    data_path = Path('data/processed/volgan_training_data.csv')
    if data_path.exists():
        print(f"\n✅ Training data found: {data_path}")
        print("Ready to train VolGAN model!")
    else:
        print(f"\n❌ Training data not found: {data_path}")
        print("Please run the data processing pipeline first:")
        print("python run_volgan_pipeline.py --full")

if __name__ == "__main__":
    main()
