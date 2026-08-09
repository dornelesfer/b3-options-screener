# VolGAN-BR: Brazilian Market Volatility Surface Generator

This project is an adaptation of **Milena Vuletic's VolGAN paper** to the Brazilian B3 market, providing a complete pipeline for downloading, processing, and generating implied volatility surfaces from Brazilian options data.

## 🚀 Features

- **Automatic B3 Data Download**: Downloads COTAHIST options data directly from B3 website
- **Complete Data Processing Pipeline**: From raw data to VolGAN-ready surfaces
- **Implied Volatility Calculation**: Black-Scholes based IV computation
- **Surface Generation**: Daily IV surfaces in standardized coordinates (k, T)
- **VolGAN Integration Ready**: Preprocessed datasets for PyTorch model training

## 📁 Project Structure

```
volgan_b3_starter/
├── src/
│   ├── data_b3.py           # B3 COTAHIST data parser
│   ├── iv_utils.py          # IV calculation utilities
│   ├── surface.py            # IV surface construction
│   ├── b3_downloader.py     # Automatic B3 data downloader
│   └── enhanced_processor.py # Complete data processing pipeline
├── data/
│   ├── raw/                 # Downloaded COTAHIST files
│   ├── processed/           # Processed options data
│   └── surfaces/            # Daily IV surfaces
├── notebooks/
│   └── 00_quickstart.ipynb # Quick start guide
├── run_volgan_pipeline.py   # Main pipeline script
├── demo.py                  # Demo with sample data
└── requirements.txt         # Dependencies
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd volgan_b3_starter
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python demo.py
   ```

## 📊 Usage

### 1. Quick Demo (Sample Data)
```bash
python demo.py
```
This runs the complete pipeline with synthetic data to verify everything works.

### 2. Download Real B3 Data
```bash
python run_volgan_pipeline.py --download --days 30
```
Downloads the last 30 days of options data from B3.

### 3. Process Downloaded Data
```bash
python run_volgan_pipeline.py --process --min-volume 10
```
Processes downloaded data and generates IV surfaces.

### 4. Full Pipeline (Download + Process)
```bash
python run_volgan_pipeline.py --full --days 30 --min-volume 10
```
Runs the complete pipeline: download → process → generate surfaces.

### 5. Custom Date Range
```bash
python run_volgan_pipeline.py --full --start-date 2024-01-01 --end-date 2024-01-31
```

## 🔧 Pipeline Details

### Data Download (`b3_downloader.py`)
- **Source**: B3 website (COTAHIST files)
- **Format**: ZIP/TXT files with fixed-width records
- **Content**: Options data (symbol, type, price, volume, date)
- **Output**: CSV files in `data/raw/`

### Data Processing (`enhanced_processor.py`)
1. **Load Data**: Combines all downloaded files
2. **Estimate Strikes/Maturities**: Heuristic extraction from symbols
3. **Calculate IVs**: Black-Scholes implied volatility
4. **Build Surfaces**: Daily IV surfaces in (k, T) coordinates
5. **Create VolGAN Dataset**: Standardized coordinates for training

### IV Surface Construction
- **k = ln(K/F)**: Log moneyness (strike/forward ratio)
- **T**: Time to maturity in years
- **σ**: Implied volatility
- **Filtering**: Volume, liquidity, and OTM extremes

## 📈 Output Files

### Processed Data
- `data/processed/processed_options_data.csv`: Clean options data with IVs
- `data/processed/volgan_training_data.csv`: VolGAN-ready dataset

### Daily Surfaces
- `data/surfaces/surface_YYYYMMDD.csv`: Daily IV surfaces
- Each surface contains: `k`, `T`, `iv` columns

### VolGAN Dataset
- **Columns**: `date`, `k`, `T`, `iv`, `k_normalized`, `T_normalized`, `iv_normalized`
- **Normalization**: Z-score standardization for training
- **Format**: Ready for PyTorch DataLoader

## 🎯 VolGAN Integration

The processed data is ready for VolGAN training:

```python
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# Load VolGAN dataset
df = pd.read_csv('data/processed/volgan_training_data.csv')

# Convert to tensors
k_tensor = torch.tensor(df['k_normalized'].values, dtype=torch.float32)
T_tensor = torch.tensor(df['T_normalized'].values, dtype=torch.float32)
iv_tensor = torch.tensor(df['iv_normalized'].values, dtype=torch.float32)

# Create dataset
dataset = TensorDataset(k_tensor, T_tensor, iv_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## ⚙️ Configuration

### Volume Filtering
- **Default**: `min_volume = 10`
- **Purpose**: Filter out illiquid options
- **Adjustment**: Lower for more data, higher for quality

### Date Ranges
- **Default**: Last 30 days
- **Custom**: Specify start/end dates
- **Format**: YYYY-MM-DD

### Risk-Free Rate
- **Default**: 5% (0.05)
- **Adjustment**: Modify in `enhanced_processor.py`

## 🔍 Data Quality

### Filters Applied
- **Volume**: Minimum trading volume
- **Maturity**: 1 day to 2 years
- **IV Range**: 0 < σ < 5.0 (500%)
- **Price Validity**: Non-negative option prices

### Validation
- **Put-Call Parity**: Forward estimation
- **IV Bounds**: Reasonable volatility ranges
- **Data Consistency**: Date and symbol validation

## 🚨 Troubleshooting

### Common Issues

1. **No Data Downloaded**
   - Check internet connection
   - Verify B3 website accessibility
   - Try different date ranges

2. **Import Errors**
   - Install dependencies: `pip install -r requirements.txt`
   - Check Python path and module structure

3. **Processing Failures**
   - Verify data format in `data/raw/`
   - Check log files for specific errors
   - Adjust volume and date filters

### Logging
- **File**: `volgan_pipeline.log`
- **Level**: INFO
- **Content**: Download progress, processing steps, errors

## 📚 References

- **Original Paper**: Vuletić & Cont (2024/2025) - VolGAN
- **B3 Data**: [B3 Market Data](https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/market-data/cotacoes/)
- **COTAHIST Format**: B3 official documentation
- **Black-Scholes**: Standard options pricing model

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature-name`
3. **Commit changes**: `git commit -am 'Add feature'`
4. **Push branch**: `git push origin feature-name`
5. **Submit pull request**

## 📄 License

This project is adapted from academic research. Please respect the original authors' work and cite appropriately.

## 🆘 Support

- **Issues**: GitHub Issues
- **Documentation**: Check this README and inline code comments
- **Research**: Refer to original VolGAN paper for theoretical background

---

**Note**: This project is for research and educational purposes. Always verify data quality and model assumptions before using in production trading systems.
