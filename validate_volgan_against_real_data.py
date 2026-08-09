#!/usr/bin/env python3
"""
VolGAN Validation Against Real Brazilian Options Data
Comprehensive testing of generated vs. actual volatility surfaces
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import VolGAN components
from volgan_model import VolGANTrainer

# Setup plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_trained_model():
    """Load the trained VolGAN model"""
    print("📥 Loading trained VolGAN model...")
    
    trainer = VolGANTrainer(
        latent_dim=100,
        batch_size=128,
        generator_lr=0.0002,
        discriminator_lr=0.0002
    )
    
    model_path = "models/volgan_final.pth"
    if Path(model_path).exists():
        trainer.load_model(model_path)
        print(f"✅ Model loaded from: {model_path}")
        return trainer
    else:
        raise FileNotFoundError(f"Model not found: {model_path}")

def load_real_data():
    """Load real Brazilian options data for comparison"""
    print("📥 Loading real Brazilian options data...")
    
    # Load test data (unseen during training)
    test_file = Path('data/processed/volgan_test_data.csv')
    if test_file.exists():
        test_data = pd.read_csv(test_file)
        print(f"✅ Loaded test data: {len(test_data):,} records")
        
        # Sample a subset for comparison
        sample_size = min(10000, len(test_data))
        real_sample = test_data.sample(n=sample_size, random_state=42)
        
        # Extract features
        real_features = real_sample[['k', 'T', 'iv']].values
        print(f"✅ Sampled {len(real_features)} real data points for comparison")
        
        return real_features, real_sample
    else:
        raise FileNotFoundError(f"Test data not found: {test_file}")

def generate_volgan_samples(trainer, num_samples=10000):
    """Generate samples using the trained VolGAN"""
    print(f"🎨 Generating {num_samples} VolGAN samples...")
    
    trainer.generator.eval()
    with torch.no_grad():
        # Generate random latent vectors
        z = torch.randn(num_samples, trainer.latent_dim).to(trainer.device)
        
        # Generate samples
        generated_samples = trainer.generator(z)
        
        # Convert to numpy
        generated_np = generated_samples.cpu().numpy()
        
        print(f"✅ Generated {len(generated_np)} samples")
        return generated_np

def denormalize_samples(samples, stats):
    """Denormalize generated samples to original scale"""
    if stats is None:
        return samples
    
    denorm_samples = samples.copy()
    denorm_samples[:, 0] = samples[:, 0] * stats['k_std'] + stats['k_mean']  # k
    denorm_samples[:, 1] = samples[:, 1] * stats['T_std'] + stats['T_mean']  # T
    denorm_samples[:, 2] = samples[:, 2] * stats['iv_std'] + stats['iv_mean']  # iv
    
    return denorm_samples

def calculate_statistical_metrics(real_data, generated_data):
    """Calculate comprehensive statistical comparison metrics"""
    print("📊 Calculating statistical comparison metrics...")
    
    metrics = {}
    
    # Basic statistics
    for i, feature in enumerate(['k', 'T', 'iv']):
        real_feature = real_data[:, i]
        gen_feature = generated_data[:, i]
        
        metrics[f'{feature}_mean_diff'] = np.mean(gen_feature) - np.mean(real_feature)
        metrics[f'{feature}_std_diff'] = np.std(gen_feature) - np.std(real_feature)
        metrics[f'{feature}_min_diff'] = np.min(gen_feature) - np.min(real_feature)
        metrics[f'{feature}_max_diff'] = np.max(gen_feature) - np.max(real_feature)
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.ks_2samp(real_feature, gen_feature)
        metrics[f'{feature}_ks_stat'] = ks_stat
        metrics[f'{feature}_ks_pvalue'] = ks_pvalue
        
        # Wasserstein distance (Earth Mover's Distance)
        wasserstein_dist = stats.wasserstein_distance(real_feature, gen_feature)
        metrics[f'{feature}_wasserstein'] = wasserstein_dist
        
        # Mean squared error and mean absolute error
        mse = mean_squared_error(real_feature, gen_feature)
        mae = mean_absolute_error(real_feature, gen_feature)
        metrics[f'{feature}_mse'] = mse
        metrics[f'{feature}_mae'] = mae
    
    # Overall distribution similarity
    real_flat = real_data.flatten()
    gen_flat = generated_data.flatten()
    
    metrics['overall_ks_stat'] = stats.ks_2samp(real_flat, gen_flat)[0]
    metrics['overall_wasserstein'] = stats.wasserstein_distance(real_flat, gen_flat)
    
    return metrics

def plot_comparison_analysis(real_data, generated_data, metrics):
    """Create comprehensive comparison plots"""
    print("📊 Creating comparison analysis plots...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(24, 16))
    
    # Plot 1: Feature distributions comparison
    feature_names = ['Log Moneyness (k)', 'Time to Maturity (T)', 'Implied Volatility (σ)']
    
    for i in range(3):
        ax = fig.add_subplot(3, 4, i*4 + 1)
        
        # Histograms
        ax.hist(real_data[:, i], bins=50, alpha=0.7, label='Real Data', color='blue', density=True)
        ax.hist(generated_data[:, i], bins=50, alpha=0.7, label='Generated Data', color='red', density=True)
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel('Density')
        ax.set_title(f'{feature_names[i]} Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 2: Q-Q plots
    for i in range(3):
        ax = fig.add_subplot(3, 4, i*4 + 2)
        
        # Sort data for Q-Q plot
        real_sorted = np.sort(real_data[:, i])
        gen_sorted = np.sort(generated_data[:, i])
        
        # Create theoretical quantiles
        n_real = len(real_sorted)
        n_gen = len(gen_sorted)
        theoretical_real = np.quantile(real_sorted, np.linspace(0, 1, n_real))
        theoretical_gen = np.quantile(gen_sorted, np.linspace(0, 1, n_gen))
        
        ax.scatter(theoretical_real, real_sorted, alpha=0.6, label='Real Data', color='blue')
        ax.scatter(theoretical_gen, gen_sorted, alpha=0.6, label='Generated Data', color='red')
        ax.plot([real_sorted.min(), real_sorted.max()], [real_sorted.min(), real_sorted.max()], 'k--', alpha=0.5)
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Sample Quantiles')
        ax.set_title(f'{feature_names[i]} Q-Q Plot')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 3: 2D scatter comparisons
    # k vs T
    ax1 = fig.add_subplot(3, 4, 3)
    ax1.scatter(real_data[:, 0], real_data[:, 1], alpha=0.6, label='Real Data', color='blue', s=20)
    ax1.scatter(generated_data[:, 0], generated_data[:, 1], alpha=0.6, label='Generated Data', color='red', s=20)
    ax1.set_xlabel('Log Moneyness (k)')
    ax1.set_ylabel('Time to Maturity (T)')
    ax1.set_title('k vs T Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # k vs IV
    ax2 = fig.add_subplot(3, 4, 4)
    ax2.scatter(real_data[:, 0], real_data[:, 2], alpha=0.6, label='Real Data', color='blue', s=20)
    ax2.scatter(generated_data[:, 0], generated_data[:, 2], alpha=0.6, label='Generated Data', color='red', s=20)
    ax2.set_xlabel('Log Moneyness (k)')
    ax2.set_ylabel('Implied Volatility (σ)')
    ax2.set_title('k vs IV Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 4: 3D comparison
    ax3d = fig.add_subplot(3, 4, 7, projection='3d')
    ax3d.scatter(real_data[:, 0], real_data[:, 1], real_data[:, 2], 
                 alpha=0.6, label='Real Data', color='blue', s=20)
    ax3d.scatter(generated_data[:, 0], generated_data[:, 1], generated_data[:, 2], 
                 alpha=0.6, label='Generated Data', color='red', s=20)
    ax3d.set_xlabel('Log Moneyness (k)')
    ax3d.set_ylabel('Time to Maturity (T)')
    ax3d.set_zlabel('Implied Volatility (σ)')
    ax3d.set_title('3D Volatility Surface Comparison')
    ax3d.legend()
    
    # Plot 5: Statistical metrics summary
    ax_metrics = fig.add_subplot(3, 4, 8)
    
    # Create metrics summary table
    metric_names = ['Mean Diff (k)', 'Std Diff (k)', 'KS Stat (k)', 'Wasserstein (k)',
                   'Mean Diff (T)', 'Std Diff (T)', 'KS Stat (T)', 'Wasserstein (T)',
                   'Mean Diff (IV)', 'Std Diff (IV)', 'KS Stat (IV)', 'Wasserstein (IV)']
    
    metric_values = [metrics['k_mean_diff'], metrics['k_std_diff'], metrics['k_ks_stat'], metrics['k_wasserstein'],
                    metrics['T_mean_diff'], metrics['T_std_diff'], metrics['T_ks_stat'], metrics['T_wasserstein'],
                    metrics['iv_mean_diff'], metrics['iv_std_diff'], metrics['iv_ks_stat'], metrics['iv_wasserstein']]
    
    # Create table
    table_data = [[name, f"{value:.4f}"] for name, value in zip(metric_names, metric_values)]
    table = ax_metrics.table(cellText=table_data, colLabels=['Metric', 'Value'], 
                           cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    ax_metrics.set_title('Statistical Comparison Metrics')
    ax_metrics.axis('off')
    
    # Plot 6: Correlation matrices comparison
    # Real data correlation
    ax_corr1 = fig.add_subplot(3, 4, 11)
    real_corr = np.corrcoef(real_data.T)
    im1 = ax_corr1.imshow(real_corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax_corr1.set_title('Real Data Correlation Matrix')
    ax_corr1.set_xticks([0, 1, 2])
    ax_corr1.set_yticks([0, 1, 2])
    ax_corr1.set_xticklabels(['k', 'T', 'IV'])
    ax_corr1.set_yticklabels(['k', 'T', 'IV'])
    plt.colorbar(im1, ax=ax_corr1, shrink=0.5)
    
    # Generated data correlation
    ax_corr2 = fig.add_subplot(3, 4, 12)
    gen_corr = np.corrcoef(generated_data.T)
    im2 = ax_corr2.imshow(gen_corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax_corr2.set_title('Generated Data Correlation Matrix')
    ax_corr2.set_xticks([0, 1, 2])
    ax_corr2.set_yticks([0, 1, 2])
    ax_corr2.set_xticklabels(['k', 'T', 'IV'])
    ax_corr2.set_yticklabels(['k', 'T', 'IV'])
    plt.colorbar(im2, ax=ax_corr2, shrink=0.5)
    
    plt.tight_layout()
    plt.suptitle('VolGAN vs Real Brazilian Options Data: Comprehensive Comparison', fontsize=16, y=0.98)
    
    return fig

def print_validation_summary(metrics):
    """Print comprehensive validation summary"""
    print("\n" + "="*80)
    print("🎯 VOLGAN VALIDATION SUMMARY")
    print("="*80)
    
    print("\n📊 STATISTICAL COMPARISON METRICS:")
    print("-" * 50)
    
    features = ['k', 'T', 'iv']
    feature_names = ['Log Moneyness', 'Time to Maturity', 'Implied Volatility']
    
    for i, (feature, name) in enumerate(zip(features, feature_names)):
        print(f"\n🔍 {name} ({feature}):")
        print(f"   Mean Difference: {metrics[f'{feature}_mean_diff']:+.4f}")
        print(f"   Std Difference: {metrics[f'{feature}_std_diff']:+.4f}")
        print(f"   Min Difference: {metrics[f'{feature}_min_diff']:+.4f}")
        print(f"   Max Difference: {metrics[f'{feature}_max_diff']:+.4f}")
        print(f"   KS Statistic: {metrics[f'{feature}_ks_stat']:.4f}")
        print(f"   KS P-value: {metrics[f'{feature}_ks_pvalue']:.4f}")
        print(f"   Wasserstein Distance: {metrics[f'{feature}_wasserstein']:.4f}")
        print(f"   Mean Squared Error: {metrics[f'{feature}_mse']:.4f}")
        print(f"   Mean Absolute Error: {metrics[f'{feature}_mae']:.4f}")
    
    print(f"\n🌐 OVERALL DISTRIBUTION SIMILARITY:")
    print(f"   Overall KS Statistic: {metrics['overall_ks_stat']:.4f}")
    print(f"   Overall Wasserstein Distance: {metrics['overall_wasserstein']:.4f}")
    
    print("\n✅ VALIDATION COMPLETE!")
    print("="*80)

def main():
    """Main validation function"""
    print("🎯 VolGAN Validation Against Real Brazilian Options Data")
    print("=" * 70)
    
    try:
        # Load components
        trainer = load_trained_model()
        real_data, real_df = load_real_data()
        
        # Generate VolGAN samples
        generated_data = generate_volgan_samples(trainer, num_samples=len(real_data))
        
        # Load training stats for denormalization
        training_file = Path('data/processed/volgan_training_data.csv')
        if training_file.exists():
            training_data = pd.read_csv(training_file)
            stats = {
                'k_mean': training_data['k'].mean(),
                'k_std': training_data['k'].std(),
                'T_mean': training_data['T'].mean(),
                'T_std': training_data['T'].std(),
                'iv_mean': training_data['iv'].mean(),
                'iv_std': training_data['iv'].std()
            }
            print("✅ Training statistics loaded for denormalization")
        else:
            stats = None
            print("⚠️  Training data not found, using normalized values")
        
        # Denormalize generated data if stats available
        if stats:
            generated_denorm = denormalize_samples(generated_data, stats)
            print("✅ Generated data denormalized to original scale")
        else:
            generated_denorm = generated_data
            print("⚠️  Using normalized generated data")
        
        # Calculate statistical metrics
        metrics = calculate_statistical_metrics(real_data, generated_denorm)
        
        # Create comparison plots
        fig = plot_comparison_analysis(real_data, generated_denorm, metrics)
        
        # Save the comprehensive comparison plot
        fig.savefig("results/volgan_vs_real_validation.png", dpi=300, bbox_inches='tight')
        print("💾 Saved: volgan_vs_real_validation.png")
        
        # Print validation summary
        print_validation_summary(metrics)
        
        # Show plots
        plt.show()
        
        print("\n✅ Validation complete! Check 'results/volgan_vs_real_validation.png' for detailed comparison.")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        raise

if __name__ == "__main__":
    main()

