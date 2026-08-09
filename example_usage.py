#!/usr/bin/env python3
"""
Example usage of VolGAN-BR project with real data
This shows the complete workflow for processing B3 options data
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data_b3 import read_cotahist_options, attach_instrument_table
from iv_utils import bs_implied_vol, estimate_forward_discount
from surface import build_surface_day

def process_real_data_example():
    """
    Example workflow for processing real B3 options data
    """
    print("📊 VolGAN-BR Real Data Processing Example")
    print("=" * 50)
    
    # Step 1: Check for data files
    raw_files = list(Path('data/raw').glob('*.txt')) + list(Path('data/raw').glob('*.zip'))
    
    if not raw_files:
        print("❌ No data files found in data/raw/")
        print("   Please place COTAHIST files (.txt or .zip) in the data/raw/ directory")
        print("   You can download them from: https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/market-data/cotacoes/")
        return
    
    print(f"✅ Found {len(raw_files)} data file(s):")
    for f in raw_files:
        print(f"   - {f.name}")
    
    # Step 2: Process each file
    all_options = []
    for file_path in raw_files:
        print(f"\n📁 Processing {file_path.name}...")
        try:
            df = read_cotahist_options(file_path)
            print(f"   Parsed {len(df)} option records")
            all_options.append(df)
        except Exception as e:
            print(f"   ❌ Error processing {file_path.name}: {e}")
    
    if not all_options:
        print("\n❌ No data could be processed")
        return
    
    # Combine all data
    df_all = pd.concat(all_options, ignore_index=True)
    print(f"\n📈 Total options processed: {len(df_all)}")
    
    # Step 3: Show data structure
    print(f"\n📋 Data structure:")
    print(f"   Date range: {df_all['date'].min()} to {df_all['date'].max()}")
    print(f"   Option types: {df_all['type'].value_counts().to_dict()}")
    print(f"   Unique symbols: {df_all['option_symbol'].nunique()}")
    
    # Step 4: Note about next steps
    print(f"\n⚠️  Important notes:")
    print(f"   1. Column positions in src/data_b3.py may need adjustment")
    print(f"   2. Strike prices and maturities are currently NaN (need BVBG-086 table)")
    print(f"   3. Use attach_instrument_table() to add metadata")
    
    # Step 5: Example of what to do next
    print(f"\n🔧 Next steps:")
    print(f"   1. Download BVBG-086 instrument table from B3")
    print(f"   2. Adjust column positions in src/data_b3.py if needed")
    print(f"   3. Use attach_instrument_table() to add strikes/maturities")
    print(f"   4. Run forward estimation and IV calculation")
    print(f"   5. Build IV surfaces for VolGAN training")
    
    return df_all

if __name__ == "__main__":
    process_real_data_example()
