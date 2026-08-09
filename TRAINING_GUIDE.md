# 🚀 VolGAN-BR Training Guide

This guide will walk you through training the VolGAN model using the same date range as Milena Vuletic's paper, with smart data management to avoid redownloading.

## 📅 Date Ranges (Vuletic's Paper)

- **Training Period**: January 3, 2000 to June 16, 2018
  - **Start**: 3rd January 2000
  - **End**: 16th June 2018
  - **Total Days**: ~6,750 days
  - **Rationale**: 18+ years of data provides extensive training samples for robust model training

- **Test Period**: June 17, 2019 to most recent available
  - **Start**: 17th June 2019
  - **End**: Current date (automatically updated)
  - **Total Days**: Varies based on current date
  - **Rationale**: Out-of-sample testing on recent market data

## 🎯 Quick Start

### 1. Check Project Status
```bash
python check_status.py
```
This shows what's available and what needs to be done.

### 2. Run Complete Training Pipeline
```bash
python train_volgan.py
```
This automatically:
- Downloads missing B3 data (if any)
- Processes the data into VolGAN-ready format
- Trains the VolGAN model
- Saves results and visualizations

### 3. Custom Training Parameters
```bash
python train_volgan.py --epochs 200 --batch-size 64 --latent-dim 128
```

## 🔧 Training Options

### Basic Training
```bash
# Default settings (100 epochs, batch size 32)
python train_volgan.py
```

### Advanced Training
```bash
# Custom parameters
python train_volgan.py \
    --epochs 200 \
    --batch-size 64 \
    --latent-dim 128 \
    --lr 0.0001 \
    --save-interval 20
```

### Data-Only Operations
```bash
# Download data only
python train_volgan.py --skip-training

# Force redownload
python train_volgan.py --force-download --skip-training

# Force reprocessing
python train_volgan.py --force-process --skip-training
```

## 📊 Data Management

### Smart Download
- **First run**: Downloads all required data (2021-2023)
- **Subsequent runs**: Only downloads missing dates
- **No redownloading**: Data is preserved between runs

### Data Processing
- **Automatic**: Processes raw data into VolGAN format
- **Cached**: Processed data is saved and reused
- **Efficient**: Only reprocesses when needed

### Data Quality
- **Volume filter**: Minimum 10 trades per option
- **Maturity range**: 1 day to 2 years
- **IV bounds**: 0% to 500% implied volatility
- **Validation**: Put-call parity and consistency checks

## 🏗️ Model Architecture

### Generator Network
- **Input**: 100-dimensional noise vector
- **Hidden layers**: [256, 512, 256, 128] neurons
- **Output**: 3-dimensional (k, T, iv) coordinates
- **Activation**: ReLU with BatchNorm and Dropout

### Discriminator Network
- **Input**: 3-dimensional (k, T, iv) coordinates
- **Hidden layers**: [128, 256, 128, 64] neurons
- **Output**: Real/fake classification
- **Activation**: LeakyReLU with Dropout

### Training Parameters
- **Learning rate**: 0.0002 (Adam optimizer)
- **Batch size**: 32 (configurable)
- **Epochs**: 100 (configurable)
- **Save interval**: Every 10 epochs

## 📁 Output Structure

### Models Directory (`models/`)
```
models/
├── volgan_epoch_10.pth      # Checkpoint at epoch 10
├── volgan_epoch_20.pth      # Checkpoint at epoch 20
├── ...
└── volgan_final.pth         # Final trained model
```

### Results Directory (`results/`)
```
results/
├── data_statistics.json     # Data normalization stats
├── training_results.json     # Training summary
├── training_history.png      # Loss curves
├── sample_surface_3d.png    # 3D surface visualization
├── sample_surface_1.csv     # Generated surface data
├── sample_surface_2.csv     # Generated surface data
└── ...
```

### Logs
- **Training log**: `volgan_training.log`
- **Pipeline log**: `volgan_pipeline.log`

## 🎨 Generated Surfaces

### Surface Characteristics
- **Points per surface**: 100 (configurable)
- **Number of surfaces**: 10 (configurable)
- **Coordinates**: (k, T, iv) in denormalized units

### Surface Visualization
- **3D scatter plots** with color-coded IV values
- **Interactive matplotlib** plots
- **Saved as PNG** for documentation

## 🔍 Monitoring Training

### Loss Curves
- **Generator loss**: Should decrease over time
- **Discriminator loss**: Should stabilize around 0.5
- **Convergence**: Look for stable loss patterns

### Quality Indicators
- **Realistic IV ranges**: 10% to 100% typically
- **Smooth surfaces**: No extreme outliers
- **Consistent patterns**: Similar to training data

## 🚨 Troubleshooting

### Common Issues

1. **Out of Memory**
   ```bash
   # Reduce batch size
   python train_volgan.py --batch-size 16
   ```

2. **Training Not Converging**
   ```bash
   # Increase epochs, adjust learning rate
   python train_volgan.py --epochs 200 --lr 0.0001
   ```

3. **Data Issues**
   ```bash
   # Force reprocessing
   python train_volgan.py --force-process
   ```

### Performance Tips

1. **GPU Training**: Automatically detected if available
2. **Batch Size**: Larger batches for faster training
3. **Save Frequency**: Lower save_interval for more checkpoints

## 📈 Evaluation

### Training Metrics
- **Loss curves**: Generator vs Discriminator
- **Convergence**: Stable loss patterns
- **Overfitting**: Monitor validation performance

### Generated Quality
- **IV ranges**: Compare to training data
- **Surface smoothness**: Visual inspection
- **Statistical properties**: Mean, variance, distributions

## 🔄 Resuming Training

### From Checkpoint
```python
from src.volgan_model import VolGANTrainer

trainer = VolGANTrainer()
trainer.load_model('models/volgan_epoch_50.pth')

# Continue training
trainer.train(train_loader, epochs=100, start_epoch=50)
```

### Custom Training Loop
```python
# Load trained model
trainer.load_model('models/volgan_final.pth')

# Generate new surfaces
surfaces = trainer.generate_surfaces(n_surfaces=20, n_points=200)
```

## 📚 Next Steps

### After Training
1. **Analyze results**: Check generated surface quality
2. **Fine-tune**: Adjust hyperparameters if needed
3. **Generate**: Create new IV surfaces for analysis
4. **Validate**: Compare with real market data

### Research Applications
- **Risk modeling**: Generate stress scenarios
- **Portfolio optimization**: IV surface dynamics
- **Market simulation**: Synthetic volatility paths
- **Academic research**: VolGAN methodology validation

---

**Note**: This implementation follows Milena Vuletic's VolGAN paper methodology, adapted for Brazilian B3 options data. Always validate results against real market data before using in production systems.
