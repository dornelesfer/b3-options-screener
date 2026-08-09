#!/usr/bin/env python3
"""
Status check script for VolGAN-BR project
Shows what's available and what needs to be done
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def check_project_status():
    """Check the current status of the VolGAN-BR project"""
    print("🔍 VolGAN-BR Project Status Check")
    print("=" * 50)
    
    # Check configuration
    print("\n📋 Configuration:")
    try:
        from config import print_config
        print_config()
    except ImportError:
        print("   ❌ Configuration file not found")
    
    # Check data availability
    print("\n📊 Data Status:")
    data_dir = Path('data')
    
    if data_dir.exists():
        # Raw data
        raw_dir = data_dir / 'raw'
        if raw_dir.exists():
            raw_files = list(raw_dir.glob('*.csv'))
            print(f"   Raw data files: {len(raw_files)}")
            if raw_files:
                dates = []
                for f in raw_files:
                    try:
                        date_str = f.stem.split('_')[-1]
                        date = datetime.strptime(date_str, '%Y%m%d')
                        dates.append(date)
                    except:
                        continue
                
                if dates:
                    print(f"   Date range: {min(dates).strftime('%Y-%m-%d')} to {max(dates).strftime('%Y-%m-%d')}")
        else:
            print("   Raw data directory: ❌ Not found")
        
        # Processed data
        processed_dir = data_dir / 'processed'
        if processed_dir.exists():
            processed_files = list(processed_dir.glob('*.csv'))
            print(f"   Processed data files: {len(processed_files)}")
            
            volgan_data = processed_dir / 'volgan_training_data.csv'
            if volgan_data.exists():
                print("   VolGAN training data: ✅ Available")
            else:
                print("   VolGAN training data: ❌ Not found")
        else:
            print("   Processed data directory: ❌ Not found")
        
        # Surfaces
        surfaces_dir = data_dir / 'surfaces'
        if surfaces_dir.exists():
            surface_files = list(surfaces_dir.glob('*.csv'))
            print(f"   IV surface files: {len(surface_files)}")
        else:
            print("   IV surfaces directory: ❌ Not found")
    else:
        print("   Data directory: ❌ Not found")
    
    # Check models
    print("\n🤖 Model Status:")
    models_dir = Path('models')
    if models_dir.exists():
        model_files = list(models_dir.glob('*.pth'))
        print(f"   Trained models: {len(model_files)}")
        if model_files:
            for f in model_files:
                print(f"     - {f.name}")
    else:
        print("   Models directory: ❌ Not found")
    
    # Check results
    print("\n📈 Results Status:")
    results_dir = Path('results')
    if results_dir.exists():
        result_files = list(results_dir.glob('*'))
        print(f"   Result files: {len(result_files)}")
        if result_files:
            for f in result_files:
                print(f"     - {f.name}")
    else:
        print("   Results directory: ❌ Not found")
    
    # Check source code
    print("\n💻 Source Code Status:")
    src_dir = Path('src')
    if src_dir.exists():
        src_files = list(src_dir.glob('*.py'))
        print(f"   Source files: {len(src_files)}")
        
        required_files = [
            'data_b3.py',
            'iv_utils.py', 
            'surface.py',
            'b3_downloader.py',
            'enhanced_processor.py',
            'volgan_model.py'
        ]
        
        for req_file in required_files:
            if (src_dir / req_file).exists():
                print(f"     ✅ {req_file}")
            else:
                print(f"     ❌ {req_file}")
    else:
        print("   Source directory: ❌ Not found")
    
    # Check dependencies
    print("\n📦 Dependencies Status:")
    try:
        import torch
        print(f"   PyTorch: ✅ {torch.__version__}")
    except ImportError:
        print("   PyTorch: ❌ Not installed")
    
    try:
        import pandas
        print(f"   Pandas: ✅ {pandas.__version__}")
    except ImportError:
        print("   Pandas: ❌ Not installed")
    
    try:
        import numpy
        print(f"   NumPy: ✅ {numpy.__version__}")
    except ImportError:
        print("   NumPy: ❌ Not installed")
    
    try:
        import matplotlib
        print(f"   Matplotlib: ✅ {matplotlib.__version__}")
    except ImportError:
        print("   Matplotlib: ❌ Not installed")
    
    # Recommendations
    print("\n💡 Recommendations:")
    
    if not (data_dir / 'raw').exists() or len(list((data_dir / 'raw').glob('*.csv'))) == 0:
        print("   1. 📥 Download B3 data: python run_volgan_pipeline.py --download")
    
    if not (data_dir / 'processed' / 'volgan_training_data.csv').exists():
        print("   2. 🔄 Process data: python run_volgan_pipeline.py --process")
    
    if not (models_dir).exists() or len(list(models_dir.glob('*.pth'))) == 0:
        print("   3. 🚀 Train VolGAN: python train_volgan.py")
    
    print("\n🎯 Ready to proceed with:")
    if (data_dir / 'processed' / 'volgan_training_data.csv').exists():
        print("   ✅ VolGAN training")
    else:
        print("   ⏳ Data preparation first")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    check_project_status()
