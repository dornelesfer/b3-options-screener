#!/usr/bin/env python3
"""
Main script for VolGAN-BR project
Downloads B3 options data and processes it for VolGAN training
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime, timedelta
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('volgan_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(description='VolGAN-BR Data Pipeline')
    parser.add_argument('--download', action='store_true', 
                       help='Download data from B3')
    parser.add_argument('--process', action='store_true',
                       help='Process downloaded data')
    parser.add_argument('--full', action='store_true',
                       help='Run full pipeline (download + process)')
    parser.add_argument('--start-date', type=str, default=None,
                       help='Start date for data (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date for data (YYYY-MM-DD)')
    parser.add_argument('--days', type=int, default=30,
                       help='Number of days to process (default: 30)')
    parser.add_argument('--min-volume', type=int, default=10,
                       help='Minimum volume filter (default: 10)')
    
    args = parser.parse_args()
    
    # Set default dates if not provided
    if not args.start_date:
        args.start_date = (datetime.now() - timedelta(days=args.days)).strftime('%Y-%m-%d')
    if not args.end_date:
        args.end_date = datetime.now().strftime('%Y-%m-%d')
    
    print("🚀 VolGAN-BR Pipeline")
    print("=" * 40)
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Min volume: {args.min_volume}")
    print()
    
    try:
        if args.download or args.full:
            logger.info("Starting data download...")
            from b3_downloader import B3Downloader
            
            downloader = B3Downloader()
            files = downloader.download_historical_data(
                start_date=args.start_date,
                end_date=args.end_date,
                output_dir='data/raw'
            )
            
            if files:
                logger.info(f"Downloaded {len(files)} files")
            else:
                logger.warning("No files were downloaded")
        
        if args.process or args.full:
            logger.info("Starting data processing...")
            from enhanced_processor import EnhancedProcessor
            
            processor = EnhancedProcessor()
            results = processor.run_full_pipeline(
                date_range=(args.start_date, args.end_date),
                min_volume=args.min_volume
            )
            
            if results:
                print(f"\n📊 Pipeline Results:")
                print(f"   Raw data: {len(results['raw_data'])} records")
                print(f"   Processed data: {len(results['processed_data'])} records")
                print(f"   Daily surfaces: {len(results['surfaces'])} days")
                print(f"   VolGAN dataset: {len(results['volgan_dataset'])} points")
                
                # Show some statistics
                if len(results['volgan_dataset']) > 0:
                    volgan_df = results['volgan_dataset']
                    print(f"\n📈 VolGAN Dataset Statistics:")
                    print(f"   Date range: {volgan_df['date'].min()} to {volgan_df['date'].max()}")
                    print(f"   Moneyness range: {volgan_df['k'].min():.3f} to {volgan_df['k'].max():.3f}")
                    print(f"   Time range: {volgan_df['T'].min():.3f} to {volgan_df['T'].max():.3f} years")
                    print(f"   IV range: {volgan_df['iv'].min():.3f} to {volgan_df['iv'].max():.3f}")
                    
                    # Save summary
                    summary_file = Path('data/processed/pipeline_summary.txt')
                    summary_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(summary_file, 'w') as f:
                        f.write("VolGAN-BR Pipeline Summary\n")
                        f.write("=" * 30 + "\n")
                        f.write(f"Date: {datetime.now()}\n")
                        f.write(f"Date range: {args.start_date} to {args.end_date}\n")
                        f.write(f"Min volume: {args.min_volume}\n\n")
                        f.write(f"Raw data: {len(results['raw_data'])} records\n")
                        f.write(f"Processed data: {len(results['processed_data'])} records\n")
                        f.write(f"Daily surfaces: {len(results['surfaces'])} days\n")
                        f.write(f"VolGAN dataset: {len(results['volgan_dataset'])} points\n")
                    
                    print(f"\n💾 Summary saved to: {summary_file}")
            else:
                logger.error("Pipeline failed - no results")
        
        if not (args.download or args.process or args.full):
            print("No action specified. Use --help for options.")
            print("\nExample usage:")
            print("  python run_volgan_pipeline.py --full                    # Run complete pipeline")
            print("  python run_volgan_pipeline.py --download               # Download data only")
            print("  python run_volgan_pipeline.py --process                # Process data only")
            print("  python run_volgan_pipeline.py --full --days 7          # Last 7 days")
            print("  python run_volgan_pipeline.py --full --min-volume 5    # Lower volume threshold")
    
    except ImportError as e:
        logger.error(f"Import error: {e}")
        print("❌ Failed to import required modules. Make sure all dependencies are installed.")
        print("   Run: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"❌ Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
