# src/enhanced_processor.py
"""
Enhanced Data Processor for VolGAN-BR
Integrates downloaded B3 data with the VolGAN pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional

# Import our modules
from data_b3 import read_cotahist_options
from iv_utils import bs_implied_vol, estimate_forward_discount
from surface import build_surface_day

logger = logging.getLogger(__name__)

class EnhancedProcessor:
    """
    Enhanced processor for B3 options data with VolGAN integration
    """
    
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        self.surfaces_dir = self.data_dir / 'surfaces'
        
        # Create directories
        for dir_path in [self.raw_dir, self.processed_dir, self.surfaces_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def load_downloaded_data(self, date_range: Optional[Tuple[str, str]] = None) -> pd.DataFrame:
        """
        Load all downloaded data files
        
        Args:
            date_range: Optional tuple of (start_date, end_date) in 'YYYY-MM-DD' format
            
        Returns:
            Combined DataFrame with all options data
        """
        # Find all CSV files
        files = list(self.raw_dir.glob('*.csv'))
        
        if not files:
            logger.warning("No data files found in raw directory")
            return pd.DataFrame()
        
        # Filter by date range if specified
        if date_range:
            start_date, end_date = date_range
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            filtered_files = []
            for file_path in files:
                # Extract date from filename
                try:
                    date_str = file_path.stem.split('_')[-1]
                    file_date = datetime.strptime(date_str, '%Y%m%d')
                    if start_dt <= file_date <= end_dt:
                        filtered_files.append(file_path)
                except:
                    continue
            
            files = filtered_files
        
        logger.info(f"Loading {len(files)} data files...")
        
        # Load and combine all files
        all_data = []
        for file_path in files:
            try:
                df = pd.read_csv(file_path)
                all_data.append(df)
                logger.info(f"Loaded {file_path.name}: {len(df)} records")
            except Exception as e:
                logger.error(f"Failed to load {file_path.name}: {e}")
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Convert date column
        combined_df['date'] = pd.to_datetime(combined_df['date'], format='%Y%m%d', errors='coerce')
        
        # Remove invalid dates
        combined_df = combined_df.dropna(subset=['date'])
        
        logger.info(f"Combined data: {len(combined_df)} total records")
        logger.info(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
        
        return combined_df
    
    def estimate_strikes_and_maturities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate strike prices and maturities from option symbols
        This is a heuristic approach - in production you'd use the BVBG-086 table
        
        Args:
            df: DataFrame with options data
            
        Returns:
            DataFrame with estimated strikes and maturities
        """
        logger.info("Estimating strikes and maturities from option symbols...")
        
        df = df.copy()
        
        # Extract strike from symbol (heuristic)
        def extract_strike(symbol):
            try:
                # Look for numbers in the symbol that could be strike prices
                import re
                numbers = re.findall(r'\d+', symbol)
                if numbers:
                    # Assume the first number is the strike (adjust logic as needed)
                    return float(numbers[0])
                return np.nan
            except:
                return np.nan
        
        # Extract maturity from symbol (heuristic)
        def extract_maturity(symbol):
            try:
                # Look for maturity indicators in the symbol
                import re
                maturity_patterns = {
                    'A': 30, 'B': 60, 'C': 90, 'D': 120, 'E': 150, 'F': 180,
                    'G': 210, 'H': 240, 'I': 270, 'J': 300, 'K': 330, 'L': 360
                }
                
                for letter, days in maturity_patterns.items():
                    if letter in symbol:
                        return days
                
                return np.nan
            except:
                return np.nan
        
        # Apply extraction
        df['strike'] = df['option_symbol'].apply(extract_strike)
        maturity_days = df['option_symbol'].apply(extract_maturity)
        
        # Convert maturity days to actual dates
        df['maturity'] = df.apply(
            lambda row: row['date'] + timedelta(days=int(maturity_days[row.name])) 
            if not pd.isna(maturity_days[row.name]) else None, 
            axis=1
        )
        
        # Filter out options without valid strikes or maturities
        valid_df = df.dropna(subset=['strike', 'maturity']).copy()
        
        logger.info(f"Estimated strikes/maturities for {len(valid_df)} out of {len(df)} options")
        
        return valid_df
    
    def calculate_implied_volatilities(self, df: pd.DataFrame, risk_free_rate: float = 0.05) -> pd.DataFrame:
        """
        Calculate implied volatilities for all options
        
        Args:
            df: DataFrame with options data
            risk_free_rate: Risk-free rate (default: 5%)
            
        Returns:
            DataFrame with implied volatilities
        """
        logger.info("Calculating implied volatilities...")
        
        df = df.copy()
        
        # Calculate time to maturity in years
        df['T'] = (df['maturity'] - df['date']).dt.days / 365.0
        
        # Filter out options with very short or long maturities
        df = df[(df['T'] >= 1/365) & (df['T'] <= 2.0)]
        
        # Calculate IV for each option
        ivs = []
        for _, row in df.iterrows():
            try:
                iv = bs_implied_vol(
                    price=row['price'],
                    S=row['strike'],  # Using strike as spot price approximation
                    K=row['strike'],
                    T=row['T'],
                    r=risk_free_rate,
                    q=0.0,  # No dividend yield
                    opt_type=row['type']
                )
                ivs.append(iv)
            except:
                ivs.append(np.nan)
        
        df['iv'] = ivs
        
        # Filter out invalid IVs
        valid_df = df.dropna(subset=['iv']).copy()
        valid_df = valid_df[valid_df['iv'] > 0]
        
        logger.info(f"Calculated IVs for {len(valid_df)} out of {len(df)} options")
        logger.info(f"IV range: {valid_df['iv'].min():.3f} - {valid_df['iv'].max():.3f}")
        
        return valid_df
    
    def build_daily_surfaces(self, df: pd.DataFrame, min_volume: int = 10) -> Dict[str, pd.DataFrame]:
        """
        Build daily IV surfaces for VolGAN training
        
        Args:
            df: DataFrame with options data and IVs
            min_volume: Minimum volume filter
            
        Returns:
            Dictionary mapping dates to IV surfaces
        """
        logger.info("Building daily IV surfaces...")
        
        # Filter by volume
        volume_filtered = df[df['volume'] >= min_volume].copy()
        
        if len(volume_filtered) == 0:
            logger.warning("No options meet volume criteria")
            return {}
        
        # Group by date
        daily_surfaces = {}
        
        for date, day_data in volume_filtered.groupby('date'):
            try:
                # Estimate forward price for this day (simplified)
                forward_price = day_data['strike'].median()  # Simple approximation
                
                # Build surface for this day
                surface = build_surface_day(
                    day_data, 
                    forward_price, 
                    0.05,  # risk_free_rate
                    iv_func=bs_implied_vol
                )
                
                if len(surface) > 0:
                    daily_surfaces[date.strftime('%Y-%m-%d')] = surface
                    
                    # Save surface
                    surface_file = self.surfaces_dir / f"surface_{date.strftime('%Y%m%d')}.csv"
                    surface.to_csv(surface_file, index=False)
                    
            except Exception as e:
                logger.error(f"Failed to build surface for {date}: {e}")
                continue
        
        logger.info(f"Built {len(daily_surfaces)} daily surfaces")
        
        return daily_surfaces
    
    def create_volgan_dataset(self, surfaces: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create dataset suitable for VolGAN training
        
        Args:
            surfaces: Dictionary of daily IV surfaces
            
        Returns:
            DataFrame with standardized coordinates for VolGAN
        """
        logger.info("Creating VolGAN training dataset...")
        
        volgan_data = []
        
        for date, surface in surfaces.items():
            # Add date identifier
            surface_copy = surface.copy()
            surface_copy['date'] = date
            
            # Standardize coordinates
            surface_copy['k_normalized'] = (surface_copy['k'] - surface_copy['k'].mean()) / surface_copy['k'].std()
            surface_copy['T_normalized'] = (surface_copy['T'] - surface_copy['T'].mean()) / surface_copy['T'].std()
            surface_copy['iv_normalized'] = (surface_copy['iv'] - surface_copy['iv'].mean()) / surface_copy['iv'].std()
            
            volgan_data.append(surface_copy)
        
        if volgan_data:
            volgan_df = pd.concat(volgan_data, ignore_index=True)
            
            # Save VolGAN dataset
            volgan_file = self.processed_dir / 'volgan_training_data.csv'
            volgan_df.to_csv(volgan_file, index=False)
            
            logger.info(f"Created VolGAN dataset with {len(volgan_df)} points")
            logger.info(f"Saved to: {volgan_file}")
            
            return volgan_df
        else:
            logger.warning("No data available for VolGAN dataset")
            return pd.DataFrame()
    
    def run_full_pipeline(self, date_range: Optional[Tuple[str, str]] = None, 
                         min_volume: int = 10) -> Dict[str, pd.DataFrame]:
        """
        Run the complete data processing pipeline
        
        Args:
            date_range: Optional date range filter
            min_volume: Minimum volume filter
            
        Returns:
            Dictionary with processed data and surfaces
        """
        logger.info("🚀 Starting VolGAN-BR data processing pipeline")
        logger.info("=" * 50)
        
        # Step 1: Load downloaded data
        logger.info("Step 1: Loading downloaded data...")
        raw_data = self.load_downloaded_data(date_range)
        
        if len(raw_data) == 0:
            logger.error("No data to process")
            return {}
        
        # Step 2: Estimate strikes and maturities
        logger.info("Step 2: Estimating strikes and maturities...")
        enhanced_data = self.estimate_strikes_and_maturities(raw_data)
        
        # Step 3: Calculate implied volatilities
        logger.info("Step 3: Calculating implied volatilities...")
        iv_data = self.calculate_implied_volatilities(enhanced_data)
        
        # Step 4: Build daily surfaces
        logger.info("Step 4: Building daily IV surfaces...")
        surfaces = self.build_daily_surfaces(iv_data, min_volume)
        
        # Step 5: Create VolGAN dataset
        logger.info("Step 5: Creating VolGAN training dataset...")
        volgan_dataset = self.create_volgan_dataset(surfaces)
        
        # Save processed data
        processed_file = self.processed_dir / 'processed_options_data.csv'
        iv_data.to_csv(processed_file, index=False)
        logger.info(f"Saved processed data to: {processed_file}")
        
        logger.info("✅ Pipeline completed successfully!")
        
        return {
            'raw_data': raw_data,
            'processed_data': iv_data,
            'surfaces': surfaces,
            'volgan_dataset': volgan_dataset
        }

def main():
    """Example usage of the enhanced processor"""
    processor = EnhancedProcessor()
    
    # Run pipeline for last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    date_range = (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    
    results = processor.run_full_pipeline(date_range=date_range, min_volume=5)
    
    if results:
        print(f"\n📊 Pipeline Results:")
        print(f"   Raw data: {len(results['raw_data'])} records")
        print(f"   Processed data: {len(results['processed_data'])} records")
        print(f"   Daily surfaces: {len(results['surfaces'])} days")
        print(f"   VolGAN dataset: {len(results['volgan_dataset'])} points")
    else:
        print("\n❌ Pipeline failed - no data available")

if __name__ == "__main__":
    main()
