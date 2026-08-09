# src/b3_downloader.py
"""
B3 Data Downloader for VolGAN-BR project
Downloads options data directly from B3 website
"""

import requests
import pandas as pd
import numpy as np
from pathlib import Path
import zipfile
import io
import time
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class B3Downloader:
    """
    Downloads options data from B3 website
    """
    
    def __init__(self):
        self.base_url = "https://www.b3.com.br"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
    
    def get_cotahist_urls(self, start_date=None, end_date=None):
        """
        Get available COTAHIST download URLs for options data
        
        Args:
            start_date: datetime object or str (YYYY-MM-DD) for start date
            end_date: datetime object or str (YYYY-MM-DD) for end date
            
        Returns:
            List of download URLs
        """
        # Convert string dates to datetime objects if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
            
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
            
        logger.info(f"Searching for COTAHIST files from {start_date.date()} to {end_date.date()}")
        
        # B3 options data URL structure
        urls = []
        current_date = start_date
        
        while current_date <= end_date:
            # Format: YYYYMMDD
            date_str = current_date.strftime('%Y%m%d')
            
            # Try different URL patterns for options data
            patterns = [
                f"https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/market-data/cotacoes/derivativos/opcoes/cotahist-opcoes/cotahist-opcoes-{date_str}.zip",
                f"https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/market-data/cotacoes/derivativos/opcoes/cotahist-opcoes/cotahist-opcoes-{date_str}.txt"
            ]
            
            for url in patterns:
                try:
                    response = self.session.head(url, timeout=10)
                    if response.status_code == 200:
                        urls.append((url, current_date))
                        logger.info(f"Found data for {current_date.date()}: {url}")
                        break
                except Exception as e:
                    logger.debug(f"URL not accessible: {url} - {e}")
            
            current_date += timedelta(days=1)
        
        return urls
    
    def download_cotahist_file(self, url, save_path=None):
        """
        Download a single COTAHIST file
        
        Args:
            url: URL to download
            save_path: Path to save the file (optional)
            
        Returns:
            DataFrame with options data or None if failed
        """
        try:
            logger.info(f"Downloading: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Determine file type and process
            if url.endswith('.zip'):
                return self._process_zip_file(response.content, save_path)
            elif url.endswith('.txt'):
                return self._process_txt_file(response.content, save_path)
            else:
                logger.error(f"Unknown file type: {url}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return None
    
    def _process_zip_file(self, content, save_path):
        """Process ZIP file content"""
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zip_file:
                # Find the text file inside
                txt_files = [f for f in zip_file.namelist() if f.endswith('.txt')]
                if not txt_files:
                    logger.error("No text files found in ZIP")
                    return None
                
                txt_content = zip_file.read(txt_files[0]).decode('latin1')
                return self._parse_cotahist_content(txt_content, save_path)
                
        except Exception as e:
            logger.error(f"Failed to process ZIP file: {e}")
            return None
    
    def _process_txt_file(self, content, save_path):
        """Process TXT file content"""
        try:
            txt_content = content.decode('latin1')
            return self._parse_cotahist_content(txt_content, save_path)
        except Exception as e:
            logger.error(f"Failed to process TXT file: {e}")
            return None
    
    def _parse_cotahist_content(self, content, save_path):
        """Parse COTAHIST content and extract options data"""
        lines = content.splitlines()
        options_data = []
        
        for line in lines:
            if len(line) < 120:
                continue
                
            # Check if this is an options line (type 70 = options)
            try:
                record_type = line[0:2]
                if record_type == '70':  # Options record type
                    data = self._parse_options_line(line)
                    if data:
                        options_data.append(data)
            except Exception as e:
                logger.debug(f"Failed to parse line: {e}")
                continue
        
        if options_data:
            df = pd.DataFrame(options_data)
            logger.info(f"Parsed {len(df)} options records")
            
            # Save if requested
            if save_path:
                self._save_data(df, save_path)
            
            return df
        else:
            logger.warning("No options data found in file")
            return None
    
    def _parse_options_line(self, line):
        """Parse a single options line from COTAHIST"""
        try:
            # COTAHIST options format (adjust positions as needed)
            date = line[2:10]  # YYYYMMDD
            symbol = line[12:24].strip()  # Option symbol
            price_raw = line[109:121].strip()  # Price * 100
            volume = line[152:170].strip()  # Volume
            
            # Parse price
            if price_raw.isdigit():
                price = float(price_raw) / 100
            else:
                price = np.nan
            
            # Parse volume
            if volume.isdigit():
                volume = int(volume)
            else:
                volume = 0
            
            # Determine option type (heuristic - adjust based on actual data)
            opt_type = None
            if symbol.endswith(('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L')):
                opt_type = 'C'  # Call
            elif symbol.endswith(('M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X')):
                opt_type = 'P'  # Put
            
            if opt_type and not pd.isna(price):
                return {
                    'date': date,
                    'option_symbol': symbol,
                    'type': opt_type,
                    'price': price,
                    'volume': volume,
                    'strike': np.nan,  # Will be filled later
                    'maturity': None   # Will be filled later
                }
            
        except Exception as e:
            logger.debug(f"Failed to parse options line: {e}")
            return None
        
        return None
    
    def _save_data(self, df, save_path):
        """Save data to file"""
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            if save_path.suffix == '.parquet':
                df.to_parquet(save_path, index=False)
            elif save_path.suffix == '.csv':
                df.to_csv(save_path, index=False)
            else:
                df.to_csv(save_path.with_suffix('.csv'), index=False)
            
            logger.info(f"Data saved to: {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save data: {e}")
    
    def download_historical_data(self, start_date, end_date, output_dir='data/raw'):
        """
        Download historical options data for a date range
        
        Args:
            start_date: Start date (datetime or string)
            end_date: End date (datetime or string)
            output_dir: Directory to save files
            
        Returns:
            List of downloaded files
        """
        # Convert dates if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get available URLs
        urls = self.get_cotahist_urls(start_date, end_date)
        
        if not urls:
            logger.warning("No data files found for the specified date range")
            return []
        
        downloaded_files = []
        
        for url, date in urls:
            # Create filename
            filename = f"cotahist_opcoes_{date.strftime('%Y%m%d')}.csv"
            save_path = output_dir / filename
            
            # Skip if already exists
            if save_path.exists():
                logger.info(f"File already exists: {filename}")
                downloaded_files.append(save_path)
                continue
            
            # Download and process
            df = self.download_cotahist_file(url, save_path)
            if df is not None:
                downloaded_files.append(save_path)
                logger.info(f"Successfully downloaded: {filename}")
            
            # Rate limiting
            time.sleep(1)
        
        return downloaded_files

def main():
    """Example usage"""
    downloader = B3Downloader()
    
    # Download last 7 days of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    print("🚀 B3 Options Data Downloader")
    print("=" * 40)
    print(f"Downloading data from {start_date.date()} to {end_date.date()}")
    
    files = downloader.download_historical_data(start_date, end_date)
    
    if files:
        print(f"\n✅ Successfully downloaded {len(files)} files:")
        for f in files:
            print(f"   - {f.name}")
    else:
        print("\n❌ No files were downloaded")
        print("   This might be due to:")
        print("   - No data available for the date range")
        print("   - Network issues")
        print("   - B3 website changes")

if __name__ == "__main__":
    main()
