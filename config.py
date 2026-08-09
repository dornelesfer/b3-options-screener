# config.py
"""
Configuration file for VolGAN-BR project
Contains settings used in Milena Vuletic's paper
"""

from datetime import datetime

# Date ranges for VolGAN training
# Note: B3 data availability may be limited, so we start with practical ranges
VULETIC_DATE_RANGE = {
    # Practical training period (last 2 years for B3 data)
    'training_start': datetime(2022, 1, 1),    # Training start: 1st Jan 2022
    'training_end': datetime(2023, 12, 31),    # Training end: 31st Dec 2023
    # Test period (current year)
    'test_start': datetime(2024, 1, 1),        # Test start: 1st Jan 2024
    'test_end': datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),  # Test end: Most recent available
    'description': 'Practical B3 data ranges: training 2022-2023, test 2024-present'
}

# Alternative: Full Vuletic paper ranges (if B3 data becomes available)
VULETIC_PAPER_RANGES = {
    'training_start': datetime(2000, 1, 3),    # Training start: 3rd Jan 2000
    'training_end': datetime(2018, 6, 16),     # Training end: 16th Jun 2018
    'test_start': datetime(2019, 6, 17),       # Test start: 17th Jun 2019
    'test_end': datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),  # Test end: Most recent available
    'description': 'Exact date ranges from Vuletic paper: training 2000-2018, test 2019-present'
}

# Training parameters (based on typical GAN training)
TRAINING_CONFIG = {
    'epochs': 100,
    'batch_size': 32,
    'latent_dim': 100,
    'generator_lr': 0.0002,
    'discriminator_lr': 0.0002,
    'beta1': 0.5,
    'beta2': 0.999,
    'save_interval': 10
}

# Model architecture (based on VolGAN paper)
MODEL_CONFIG = {
    'generator_hidden_dims': [256, 512, 256, 128],
    'discriminator_hidden_dims': [128, 256, 128, 64],
    'dropout_rate': 0.3,
    'activation': 'ReLU',
    'discriminator_activation': 'LeakyReLU'
}

# Data processing parameters
DATA_CONFIG = {
    'min_volume': 10,           # Minimum trading volume filter
    'min_maturity_days': 1,     # Minimum time to maturity (days)
    'max_maturity_years': 2.0,  # Maximum time to maturity (years)
    'max_iv': 5.0,              # Maximum implied volatility (500%)
    'test_split': 0.2,          # Fraction of data for testing
    'random_seed': 42           # For reproducible results
}

# File paths and directories
PATHS = {
    'data_raw': 'data/raw',
    'data_processed': 'data/processed',
    'data_surfaces': 'data/surfaces',
    'models': 'models',
    'results': 'results',
    'logs': 'logs'
}

# B3 data configuration
B3_CONFIG = {
    'base_url': 'https://www.b3.com.br',
    'options_endpoint': '/pt_br/market-data-e-indices/servicos-de-dados/market-data/cotacoes/derivativos/opcoes/cotahist-opcoes',
    'file_patterns': [
        'cotahist-opcoes-{date}.zip',
        'cotahist-opcoes-{date}.txt'
    ],
    'rate_limit_delay': 1,  # seconds between requests
    'timeout': 30,          # request timeout in seconds
    'max_retries': 3        # maximum download retries
}

# VolGAN specific settings
VOLGAN_CONFIG = {
    'surface_points': 100,      # Points per generated surface
    'n_surfaces': 10,          # Number of surfaces to generate
    'k_range': (-0.5, 0.5),    # Log moneyness range
    'T_range': (0.1, 2.0),     # Time to maturity range (years)
    'iv_range': (0.1, 1.0),    # Implied volatility range
    'normalization': 'zscore'   # Data normalization method
}

def get_config_summary():
    """Get a summary of all configuration settings"""
    return {
        'date_ranges': {
            'training': {
                'start': VULETIC_DATE_RANGE['training_start'].strftime('%Y-%m-%d'),
                'end': VULETIC_DATE_RANGE['training_end'].strftime('%Y-%m-%d'),
                'total_days': (VULETIC_DATE_RANGE['training_end'] - VULETIC_DATE_RANGE['training_start']).days + 1
            },
            'test': {
                'start': VULETIC_DATE_RANGE['test_start'].strftime('%Y-%m-%d'),
                'end': VULETIC_DATE_RANGE['test_end'].strftime('%Y-%m-%d'),
                'total_days': (VULETIC_DATE_RANGE['test_end'] - VULETIC_DATE_RANGE['test_start']).days + 1
            }
        },
        'training': TRAINING_CONFIG,
        'model': MODEL_CONFIG,
        'data': DATA_CONFIG,
        'volgan': VOLGAN_CONFIG
    }

def print_config():
    """Print current configuration"""
    print("⚙️  VolGAN-BR Configuration")
    print("=" * 50)
    
    config = get_config_summary()
    
    print(f"📅 Date Ranges (Vuletic's Paper):")
    print(f"   Training Period:")
    print(f"     Start: {config['date_ranges']['training']['start']}")
    print(f"     End: {config['date_ranges']['training']['end']}")
    print(f"     Total Days: {config['date_ranges']['training']['total_days']}")
    print(f"   Test Period:")
    print(f"     Start: {config['date_ranges']['test']['start']}")
    print(f"     End: {config['date_ranges']['test']['end']}")
    print(f"     Total Days: {config['date_ranges']['test']['total_days']}")
    
    print(f"\n🎯 Training Parameters:")
    for key, value in config['training'].items():
        print(f"   {key}: {value}")
    
    print(f"\n🏗️  Model Architecture:")
    for key, value in config['model'].items():
        print(f"   {key}: {value}")
    
    print(f"\n📊 Data Processing:")
    for key, value in config['data'].items():
        print(f"   {key}: {value}")
    
    print(f"\n🎨 VolGAN Settings:")
    for key, value in config['volgan'].items():
        print(f"   {key}: {value}")

if __name__ == "__main__":
    print_config()
