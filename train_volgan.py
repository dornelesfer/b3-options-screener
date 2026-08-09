#!/usr/bin/env python3
"""
VolGAN Training Script
Uses the same date range as Milena Vuletic's paper
Includes smart data management to avoid redownloading
"""

import sys
from pathlib import Path
import argparse
import logging
from datetime import datetime, timedelta
import json
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('volgan_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_vuletic_date_ranges():
    """
    Get the date ranges used in Milena Vuletic's VolGAN paper
    Training: 3rd January 2000 to 16th June 2018
    Test: 17th June 2019 to most recent available
    """
    from config import VULETIC_DATE_RANGE
    
    return (VULETIC_DATE_RANGE['training_start'], VULETIC_DATE_RANGE['training_end'],
            VULETIC_DATE_RANGE['test_start'], VULETIC_DATE_RANGE['test_end'])

def check_data_availability(start_date, end_date, data_dir='data'):
    """
    Check what data is already available and what needs to be downloaded
    
    Returns:
        dict with data status and missing dates
    """
    data_dir = Path(data_dir)
    raw_dir = data_dir / 'raw'
    processed_dir = data_dir / 'processed'
    
    # Check raw data files
    available_files = []
    if raw_dir.exists():
        for file_path in raw_dir.glob('*.csv'):
            try:
                # Extract date from filename
                date_str = file_path.stem.split('_')[-1]
                file_date = datetime.strptime(date_str, '%Y%m%d')
                if start_date <= file_date <= end_date:
                    available_files.append(file_date)
            except:
                continue
    
    # Check processed data
    has_processed_data = (processed_dir / 'volgan_training_data.csv').exists()
    
    # Find missing dates
    all_dates = []
    current_date = start_date
    while current_date <= end_date:
        all_dates.append(current_date)
        current_date += timedelta(days=1)
    
    missing_dates = [d for d in all_dates if d not in available_files]
    
    return {
        'available_dates': sorted(available_files),
        'missing_dates': sorted(missing_dates),
        'has_processed_data': has_processed_data,
        'total_available': len(available_files),
        'total_missing': len(missing_dates),
        'date_range_days': len(all_dates)
    }

def download_missing_data(start_date, end_date, force_download=False):
    """
    Download missing data for the specified date range
    """
    if force_download:
        logger.info("Force download requested - downloading all data")
        from b3_downloader import B3Downloader
        
        downloader = B3Downloader()
        files = downloader.download_historical_data(
            start_date=start_date,
            end_date=end_date,
            output_dir='data/raw'
        )
        
        if files:
            logger.info(f"Downloaded {len(files)} new files")
        else:
            logger.warning("No new files were downloaded")
        
        return len(files) > 0
    else:
        # Check what's missing
        status = check_data_availability(start_date, end_date)
        
        if status['total_missing'] == 0:
            logger.info("✅ All required data is already available")
            return True
        
        if status['total_missing'] > 0:
            logger.info(f"📥 Downloading {status['total_missing']} missing dates...")
            from b3_downloader import B3Downloader
            
            downloader = B3Downloader()
            files = downloader.download_historical_data(
                start_date=start_date,
                end_date=end_date,
                output_dir='data/raw'
            )
            
            if files:
                logger.info(f"Downloaded {len(files)} new files")
                return True
            else:
                logger.warning("No new files were downloaded")
                return False

def process_data_if_needed(start_date, end_date, force_process=False):
    """
    Process data if needed or if force_process is True
    """
    processed_file = Path('data/processed/volgan_training_data.csv')
    
    if processed_file.exists() and not force_process:
        logger.info("✅ Processed data already exists")
        return True
    
    logger.info("🔄 Processing data...")
    from enhanced_processor import EnhancedProcessor
    
    processor = EnhancedProcessor()
    results = processor.run_full_pipeline(
        date_range=(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')),
        min_volume=10
    )
    
    if results and len(results['volgan_dataset']) > 0:
        logger.info(f"✅ Data processing completed: {len(results['volgan_dataset'])} points")
        return True
    else:
        logger.error("❌ Data processing failed")
        return False

def train_volgan_model(epochs=100, batch_size=32, latent_dim=100, 
                       learning_rate=0.0002, save_interval=10, 
                       model_dir='models', results_dir='results'):
    """
    Train the VolGAN model
    """
    logger.info("🚀 Starting VolGAN training...")
    
    # Create directories
    model_dir = Path(model_dir)
    results_dir = Path(results_dir)
    model_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    
    # Initialize trainer
    from volgan_model import VolGANTrainer
    
    trainer = VolGANTrainer(
        latent_dim=latent_dim,
        batch_size=batch_size,
        generator_lr=learning_rate,
        discriminator_lr=learning_rate
    )
    
    # Prepare data
    data_path = Path('data/processed/volgan_training_data.csv')
    if not data_path.exists():
        logger.error(f"Training data not found: {data_path}")
        return False
    
    train_loader, test_loader, data_stats = trainer.prepare_data(data_path)
    
    # Save data statistics
    stats_file = results_dir / 'data_statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(data_stats, f, indent=2, default=str)
    logger.info(f"Data statistics saved to: {stats_file}")
    
    # Train model
    try:
        trainer.train(
            train_loader=train_loader,
            epochs=epochs,
            save_interval=save_interval,
            model_dir=model_dir
        )
        
        # Save training results
        results = {
            'training_completed': True,
            'epochs_trained': epochs,
            'final_g_loss': trainer.g_losses[-1] if trainer.g_losses else None,
            'final_d_loss': trainer.d_losses[-1] if trainer.d_losses else None,
            'total_training_time': datetime.now().isoformat(),
            'model_architecture': {
                'latent_dim': latent_dim,
                'batch_size': batch_size,
                'learning_rate': learning_rate
            }
        }
        
        results_file = results_dir / 'training_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate sample surfaces
        logger.info("🎨 Generating sample surfaces...")
        sample_surfaces = trainer.generate_surfaces(
            n_surfaces=5, 
            n_points=100, 
            data_stats=data_stats
        )
        
        # Save sample surfaces
        for i, surface in enumerate(sample_surfaces):
            surface_file = results_dir / f'sample_surface_{i+1}.csv'
            surface.to_csv(surface_file, index=False)
            logger.info(f"Sample surface {i+1} saved to: {surface_file}")
        
        # Plot training history
        history_plot = results_dir / 'training_history.png'
        trainer.plot_training_history(save_path=history_plot)
        
        # Plot sample surface
        if sample_surfaces:
            surface_plot = results_dir / 'sample_surface_3d.png'
            trainer.plot_generated_surface(sample_surfaces[0], save_path=surface_plot)
        
        logger.info("✅ VolGAN training completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return False

def main():
    """Main training execution"""
    parser = argparse.ArgumentParser(description='VolGAN Training Script')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--latent-dim', type=int, default=100, help='Latent dimension')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--save-interval', type=int, default=10, help='Save model every N epochs')
    parser.add_argument('--force-download', action='store_true', help='Force data download')
    parser.add_argument('--force-process', action='store_true', help='Force data processing')
    parser.add_argument('--skip-training', action='store_true', help='Skip training, only prepare data')
    
    args = parser.parse_args()
    
    # Get Vuletic's date ranges
    training_start, training_end, test_start, test_end = get_vuletic_date_ranges()
    
    print("🎯 VolGAN Training with Milena Vuletic's Exact Specifications")
    print("=" * 70)
    print(f"Training period: {training_start.strftime('%Y-%m-%d')} to {training_end.strftime('%Y-%m-%d')}")
    print(f"Training days: {(training_end - training_start).days + 1}")
    print(f"Test period: {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")
    print(f"Test days: {(test_end - test_start).days + 1}")
    print(f"Training epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Latent dimension: {args.latent_dim}")
    print(f"Learning rate: {args.lr}")
    print()
    
    # Step 1: Check training data availability
    print("📊 Checking training data availability...")
    training_status = check_data_availability(training_start, training_end)
    
    print(f"Training data available: {training_status['total_available']}")
    print(f"Training data missing: {training_status['total_missing']}")
    print(f"Has processed training data: {training_status['has_processed_data']}")
    print()
    
    # Step 2: Download missing training data if needed
    if training_status['total_missing'] > 0 or args.force_download:
        print("📥 Downloading missing training data...")
        success = download_missing_data(training_start, training_end, args.force_download)
        if not success:
            print("❌ Training data download failed. Exiting.")
            return
    else:
        print("✅ All required training data is available")
    
    # Step 3: Process training data if needed
    if not training_status['has_processed_data'] or args.force_process:
        print("🔄 Processing training data...")
        success = process_data_if_needed(training_start, training_end, args.force_process)
        if not success:
            print("❌ Training data processing failed. Exiting.")
            return
    else:
        print("✅ Processed training data is available")
    
    # Step 4: Download test data (for evaluation)
    print("\n📊 Checking test data availability...")
    test_status = check_data_availability(test_start, test_end)
    
    print(f"Test data available: {test_status['total_available']}")
    print(f"Test data missing: {test_status['total_missing']}")
    
    if test_status['total_missing'] > 0:
        print("📥 Downloading missing test data...")
        success = download_missing_data(test_start, test_end, args.force_download)
        if success:
            print("✅ Test data downloaded successfully")
        else:
            print("⚠️  Test data download failed - training can continue")
    
    # Step 5: Train model (unless skipped)
    if not args.skip_training:
        print("\n🚀 Starting VolGAN training...")
        success = train_volgan_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            latent_dim=args.latent_dim,
            learning_rate=args.lr,
            save_interval=args.save_interval
        )
        
        if success:
            print("\n🎉 Training completed successfully!")
            print("📁 Check the following directories for results:")
            print("   - models/: Trained model checkpoints")
            print("   - results/: Training results and visualizations")
            print("\n📊 Next steps:")
            print("   1. Evaluate model on test data")
            print("   2. Generate new IV surfaces")
            print("   3. Compare with real market data")
        else:
            print("\n❌ Training failed. Check logs for details.")
    else:
        print("\n⏭️  Skipping training as requested")
        print("📁 Data is ready for training. Run training manually:")
        print("   python train_volgan.py --epochs 100")

if __name__ == "__main__":
    main()
