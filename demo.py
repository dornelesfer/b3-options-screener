#!/usr/bin/env python3
"""
Demo script for VolGAN-BR project
Shows the complete workflow with sample data
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data_b3 import read_cotahist_options
from iv_utils import bs_implied_vol, estimate_forward_discount
from surface import build_surface_day

def create_sample_data():
    """Create sample options data for demonstration"""
    dates = pd.date_range('2024-01-01', periods=5)
    data = []
    
    for date in dates:
        # Create sample options for different strikes and maturities
        for strike in [90, 95, 100, 105, 110]:
            for maturity in [30, 60, 90]:  # days
                # Call option
                data.append({
                    'date': date,
                    'option_symbol': f'IBOV{strike}A{maturity}',
                    'type': 'C',
                    'strike': strike,
                    'maturity': date + pd.Timedelta(days=maturity),
                    'price': max(0, 100 - strike + 5)  # Simple pricing
                })
                # Put option
                data.append({
                    'date': date,
                    'option_symbol': f'IBOV{strike}M{maturity}',
                    'type': 'P',
                    'strike': strike,
                    'maturity': date + pd.Timedelta(days=maturity),
                    'price': max(0, strike - 100 + 5)  # Simple pricing
                })
    
    return pd.DataFrame(data)

def main():
    print("🚀 VolGAN-BR Project Demo")
    print("=" * 40)
    
    # Create sample data
    print("1. Creating sample options data...")
    df = create_sample_data()
    print(f"   Created {len(df)} option records")
    
    # Test data processing
    print("\n2. Testing data processing...")
    sample_day = df[df['date'] == df['date'].iloc[0]].copy()
    print(f"   Sample day has {len(sample_day)} options")
    
    # Test forward estimation
    print("\n3. Testing forward estimation...")
    for maturity in sample_day['maturity'].unique():
        day_maturity = sample_day[sample_day['maturity'] == maturity]
        calls = day_maturity[day_maturity['type'] == 'C']
        puts = day_maturity[day_maturity['type'] == 'P']
        
        if len(calls) > 0 and len(puts) > 0:
            # Create put-call parity dataframe
            strikes = sorted(set(calls['strike'].tolist() + puts['strike'].tolist()))
            pc_data = []
            for k in strikes:
                c_price = calls[calls['strike'] == k]['price'].iloc[0] if len(calls[calls['strike'] == k]) > 0 else np.nan
                p_price = puts[puts['strike'] == k]['price'].iloc[0] if len(puts[puts['strike'] == k]) > 0 else np.nan
                if not pd.isna(c_price) and not pd.isna(p_price):
                    pc_data.append({'K': k, 'C': c_price, 'P': p_price, 'T': (maturity - sample_day['date'].iloc[0]).days/365})
            
            if len(pc_data) > 0:
                pc_df = pd.DataFrame(pc_data)
                F, D = estimate_forward_discount(pc_df)
                print(f"   Maturity {maturity.strftime('%Y-%m-%d')}: F={F:.2f}, D={D:.4f}")
    
    # Test IV surface construction
    print("\n4. Testing IV surface construction...")
    # Use a simple forward price for demonstration
    forward_price = 100.0
    risk_free_rate = 0.05
    
    surface = build_surface_day(sample_day, forward_price, risk_free_rate, iv_func=bs_implied_vol)
    print(f"   Built surface with {len(surface)} points")
    if len(surface) > 0:
        print(f"   IV range: {surface['iv'].min():.3f} - {surface['iv'].max():.3f}")
        print(f"   Moneyness range: {surface['k'].min():.3f} - {surface['k'].max():.3f}")
        print(f"   Time range: {surface['T'].min():.3f} - {surface['T'].max():.3f} years")
    
    print("\n✅ Demo completed successfully!")
    print("\nNext steps:")
    print("1. Place real COTAHIST files in data/raw/")
    print("2. Adjust column positions in src/data_b3.py")
    print("3. Add BVBG-086 instrument table for strikes/maturities")
    print("4. Run with real data to build IV surfaces")

if __name__ == "__main__":
    main()
