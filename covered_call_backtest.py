#!/usr/bin/env python3
"""
Covered Call Strategy Backtesting on IBOV
Using real Brazilian options data to test covered call strategy performance
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import our modules
from data_b3 import read_cotahist_options
from iv_utils import bs_price, bs_implied_vol, estimate_forward_discount
from surface import build_surface_day

# Setup plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CoveredCallBacktester:
    """Backtesting framework for covered call strategy on IBOV"""
    
    def __init__(self, data_path="data/processed/volgan_complete_data.csv"):
        """Initialize the backtester with options data"""
        self.data_path = data_path
        self.options_data = None
        self.bova_data = None  # Changed from ibov_data to bova_data
        self.results = {}
        
    def load_data(self):
        """Load options and IBOV index data"""
        print("📥 Loading Brazilian options data...")
        
        # Load processed options data
        if Path(self.data_path).exists():
            self.options_data = pd.read_csv(self.data_path)
            print(f"✅ Loaded {len(self.options_data):,} options records")
        else:
            raise FileNotFoundError(f"Options data not found: {self.data_path}")
        
        # Create synthetic BOVA ETF data based on options data
        # In practice, you'd load real BOVA ETF data
        self._create_synthetic_bova_data()
        
        print("✅ Data loading complete")
    
    def _create_synthetic_bova_data(self):
        """Create synthetic BOVA ETF data for backtesting"""
        print("📊 Creating synthetic BOVA ETF data...")
        
        # Get unique dates from options data
        dates = pd.to_datetime(self.options_data['date']).unique()
        dates = np.sort(dates)
        
        # Create synthetic BOVA price series
        # BOVA is an ETF, so it should track IBOV more closely
        np.random.seed(42)  # For reproducibility
        n_days = len(dates)
        
        # Generate realistic BOVA movements (ETF characteristics - lower volatility than index)
        daily_returns = np.random.normal(0.0003, 0.015, n_days)  # ~0.03% daily return, 1.5% volatility
        daily_returns[0] = 0  # Start with no change
        
        # Add some market trends and volatility clustering
        for i in range(1, n_days):
            # Add momentum effect (less pronounced for ETF)
            if daily_returns[i-1] > 0.01:
                daily_returns[i] += 0.0001
            elif daily_returns[i-1] < -0.01:
                daily_returns[i] -= 0.0001
            
            # Add volatility clustering
            if abs(daily_returns[i-1]) > 0.025:
                daily_returns[i] *= 1.1
        
        # Calculate cumulative prices (BOVA typically trades around 100-200 range)
        prices = 150 * np.cumprod(1 + daily_returns)  # Start at 150 (typical BOVA price)
        
        # Create BOVA dataframe
        self.bova_data = pd.DataFrame({
            'date': dates,
            'close': prices,
            'daily_return': daily_returns
        })
        
        print(f"✅ Created synthetic BOVA data: {len(self.bova_data)} days")
        print(f"   Start: R$ {self.bova_data['close'].iloc[0]:.2f}")
        print(f"   End: R$ {self.bova_data['close'].iloc[-1]:.2f}")
        print(f"   Total return: {(self.bova_data['close'].iloc[-1] / self.bova_data['close'].iloc[0] - 1) * 100:.2f}%")
    
    def find_atm_strike(self, current_price, available_strikes, tolerance=0.20):
        """Find the strike closest to at-the-money for BOVA"""
        if len(available_strikes) == 0:
            return None
        
        # Apply contract multiplier (100x for B3 options)
        # COTAHIST strike prices are per-share, we need total contract values
        contract_multiplier = 100
        total_strikes = np.array(available_strikes) * contract_multiplier
        
        # For BOVA, strikes should be closer to the current price (100-500 range)
        # Calculate moneyness for each total strike
        moneyness = np.abs(total_strikes - current_price) / current_price
        
        # Find strikes within tolerance
        valid_indices = np.where(moneyness <= tolerance)[0]
        
        if len(valid_indices) == 0:
            # If no strikes within tolerance, use closest one
            closest_idx = np.argmin(moneyness)
            return available_strikes[closest_idx]
        
        # Among valid strikes, pick the one closest to current price
        valid_strikes = available_strikes[valid_indices]
        valid_total_strikes = total_strikes[valid_indices]
        closest_idx = np.argmin(np.abs(valid_total_strikes - current_price))
        return valid_strikes[closest_idx]
    
    def get_options_for_date(self, target_date, min_days_to_expiry=7, max_days_to_expiry=30):
        """Get available options for a specific date"""
        # Convert target_date to match the format in our data
        target_date_str = pd.to_datetime(target_date).strftime('%Y-%m-%d')
        
        date_data = self.options_data[
            (self.options_data['date'] == target_date_str) &
            (self.options_data['T'] >= min_days_to_expiry/365) &
            (self.options_data['T'] <= max_days_to_expiry/365) &
            (self.options_data['iv'] > 0) &  # Valid implied volatility
            (self.options_data['iv'] < 5.0) &  # Reasonable IV range
            (self.options_data['volume'] > 100)  # Minimum volume
        ].copy()
        
        if len(date_data) == 0:
            return None
        
        # Group by strike and get the option with shortest time to expiry
        best_options = date_data.loc[date_data.groupby('strike_price')['T'].idxmin()]
        
        return best_options
    
    def calculate_option_price(self, S, K, T, r, sigma, option_type='C'):
        """Calculate option price using Black-Scholes"""
        try:
            # Use per-share strike for Black-Scholes calculation
            # The contract multiplier is applied to the final price, not the strike
            return bs_price(S, K, T, r, 0.0, sigma, option_type)  # q=0.0 for no dividend yield
        except:
            return 0.0
    
    def run_backtest(self, 
                    initial_capital=100000,
                    risk_free_rate=0.10,  # 10% annual rate (Brazilian context)
                    min_days_to_expiry=7,
                    max_days_to_expiry=30,
                    roll_frequency_days=7):
        """Run the covered call backtest"""
        print("🚀 Starting Covered Call Backtest...")
        print(f"   Initial Capital: R$ {initial_capital:,.2f}")
        print(f"   Risk-free Rate: {risk_free_rate*100:.1f}%")
        print(f"   Roll Frequency: Every {roll_frequency_days} days")
        
        # Initialize tracking variables
        portfolio_value = initial_capital
        shares_held = 0
        options_held = None
        cash = initial_capital
        
        # Track performance
        results = []
        buy_hold_value = initial_capital
        
        # Get unique dates
        dates = pd.to_datetime(self.bova_data['date']).unique()
        dates = np.sort(dates)
        
        # Find last Friday (or most recent trading day)
        from datetime import datetime, timedelta
        today = datetime.now()
        last_friday = today - timedelta(days=(today.weekday() + 3) % 7)
        if last_friday.weekday() != 4:  # If not Friday, go back to previous Friday
            last_friday = last_friday - timedelta(days=7)
        
        # Filter dates to go up to last Friday
        last_friday_pd = pd.to_datetime(last_friday)
        dates = dates[dates <= last_friday_pd]
        
        # Start from day 30 to allow for options data
        start_idx = 30
        current_date = dates[start_idx]
        
        print(f"📅 Backtest period: {pd.to_datetime(current_date).date()} to {pd.to_datetime(dates[-1]).date()}")
        print(f"📅 Last Friday: {last_friday.date()}")
        
        for i in range(start_idx, len(dates)):
            current_date = dates[i]
            current_price = self.bova_data[self.bova_data['date'] == current_date]['close'].iloc[0]
            
            # Update buy-and-hold value
            buy_hold_value = initial_capital * (current_price / self.bova_data['close'].iloc[start_idx])
            
            # Check if we need to roll options
            roll_options = False
            if options_held is not None:
                days_to_expiry = (pd.to_datetime(options_held['expiry_date']) - current_date).days
                if days_to_expiry <= min_days_to_expiry:
                    roll_options = True
            elif i == start_idx or (i - start_idx) % roll_frequency_days == 0:
                roll_options = True
            
            if roll_options:
                # Close existing position
                if options_held is not None:
                    # Sell the option (we were short)
                    option_price = self.calculate_option_price(
                        current_price, options_held['strike_price'], 
                        float(options_held['T']), risk_free_rate, float(options_held['iv'])
                    )
                    cash += option_price * 100  # 100 shares per contract
                
                # Find new ATM call option
                available_options = self.get_options_for_date(
                    current_date, min_days_to_expiry, max_days_to_expiry
                )
                
                if available_options is not None and len(available_options) > 0:
                    print(f"   Found {len(available_options)} options for {pd.to_datetime(current_date).date()}")
                    print(f"   BOVA Price: R$ {current_price:.2f}")
                    # Find ATM strike
                    strikes = available_options['strike_price'].unique()
                    print(f"   Available strikes: {strikes[:5]}...")  # Show first 5 strikes
                    atm_strike = self.find_atm_strike(current_price, strikes)
                    
                    if atm_strike is not None:
                        # Get the option with this strike
                        option_data = available_options[available_options['strike_price'] == atm_strike].iloc[0]
                        
                        # Calculate option price
                        option_price = self.calculate_option_price(
                            current_price, atm_strike, float(option_data['T']), 
                            risk_free_rate, float(option_data['iv'])
                        )
                        
                        if option_price > 0:
                            # Update position
                            shares_held = 100  # 100 shares per contract
                            options_held = option_data.copy()
                            options_held['strike_price'] = atm_strike
                            options_held['expiry_date'] = pd.to_datetime(current_date) + timedelta(days=int(option_data['T'] * 365))
                            
                            # Update cash (we receive premium for selling call)
                            cash += option_price * 100
                            
                            print(f"📅 {pd.to_datetime(current_date).date()}: Sold ATM call K={atm_strike*100:.0f}, "
                                  f"T={option_data['T']*365:.0f}d, Premium=R${option_price*100:.2f}")
                        else:
                            options_held = None
                    else:
                        options_held = None
                else:
                    options_held = None
            
            # Calculate current portfolio value
            if options_held is not None:
                # Current option value (we're short, so we owe this)
                current_option_price = self.calculate_option_price(
                    current_price, options_held['strike_price'], 
                    float(options_held['T']), risk_free_rate, float(options_held['iv'])
                )
                option_liability = current_option_price * 100
            else:
                option_liability = 0
            
            # Portfolio value = shares * current_price + cash - option_liability
            portfolio_value = shares_held * current_price + cash - option_liability
            
            # Store results
            results.append({
                'date': current_date,
                'bova_price': current_price,
                'portfolio_value': portfolio_value,
                'buy_hold_value': buy_hold_value,
                'shares_held': shares_held,
                'cash': cash,
                'option_liability': option_liability,
                'has_option': options_held is not None,
                'strike': options_held['strike_price'] * 100 if options_held is not None else None,  # Show total contract value
                'days_to_expiry': (pd.to_datetime(options_held['expiry_date']) - current_date).days if options_held is not None else None
            })
        
        self.results = pd.DataFrame(results)
        print("✅ Backtest completed!")
        
        return self.results
    
    def calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics"""
        if self.results is None or len(self.results) == 0:
            raise ValueError("No backtest results available. Run backtest first.")
        
        print("📊 Calculating Performance Metrics...")
        
        # Basic returns
        initial_value = self.results['portfolio_value'].iloc[0]
        final_value = self.results['portfolio_value'].iloc[-1]
        total_return = (final_value / initial_value - 1) * 100
        
        initial_bh = self.results['buy_hold_value'].iloc[0]
        final_bh = self.results['buy_hold_value'].iloc[-1]
        buy_hold_return = (final_bh / initial_bh - 1) * 100
        
        # Daily returns
        portfolio_returns = self.results['portfolio_value'].pct_change().dropna()
        buy_hold_returns = self.results['buy_hold_value'].pct_change().dropna()
        
        # Risk metrics
        portfolio_vol = portfolio_returns.std() * np.sqrt(252) * 100
        buy_hold_vol = buy_hold_returns.std() * np.sqrt(252) * 100
        
        # Sharpe ratios (assuming 10% risk-free rate)
        risk_free_rate = 0.10
        portfolio_sharpe = (portfolio_returns.mean() * 252 - risk_free_rate) / (portfolio_returns.std() * np.sqrt(252))
        buy_hold_sharpe = (buy_hold_returns.mean() * 252 - risk_free_rate) / (buy_hold_returns.std() * np.sqrt(252))
        
        # Maximum drawdown
        portfolio_cummax = self.results['portfolio_value'].cummax()
        portfolio_drawdown = ((self.results['portfolio_value'] - portfolio_cummax) / portfolio_cummax * 100).min()
        
        bh_cummax = self.results['buy_hold_value'].cummax()
        bh_drawdown = ((self.results['buy_hold_value'] - bh_cummax) / bh_cummax * 100).min()
        
        # Win rate
        portfolio_win_rate = (portfolio_returns > 0).mean() * 100
        bh_win_rate = (buy_hold_returns > 0).mean() * 100
        
        metrics = {
            'Covered Call Strategy': {
                'Total Return (%)': total_return,
                'Annualized Volatility (%)': portfolio_vol,
                'Sharpe Ratio': portfolio_sharpe,
                'Max Drawdown (%)': portfolio_drawdown,
                'Win Rate (%)': portfolio_win_rate,
                'Final Value (R$)': final_value
            },
            'Buy & Hold BOVA': {
                'Total Return (%)': buy_hold_return,
                'Annualized Volatility (%)': buy_hold_vol,
                'Sharpe Ratio': buy_hold_sharpe,
                'Max Drawdown (%)': bh_drawdown,
                'Win Rate (%)': bh_win_rate,
                'Final Value (R$)': final_bh
            }
        }
        
        # Calculate outperformance
        outperformance = total_return - buy_hold_return
        volatility_reduction = buy_hold_vol - portfolio_vol
        
        metrics['Outperformance'] = {
            'Return Difference (%)': outperformance,
            'Volatility Reduction (%)': volatility_reduction,
            'Risk-Adjusted Outperformance': portfolio_sharpe - buy_hold_sharpe
        }
        
        return metrics
    
    def plot_results(self):
        """Create comprehensive performance visualization"""
        if self.results is None or len(self.results) == 0:
            raise ValueError("No backtest results available. Run backtest first.")
        
        print("📊 Creating performance visualization...")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # Plot 1: Portfolio Value Comparison
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.plot(self.results['date'], self.results['portfolio_value'], 
                label='Covered Call Strategy', linewidth=2, color='blue')
        ax1.plot(self.results['date'], self.results['buy_hold_value'], 
                label='Buy & Hold IBOV', linewidth=2, color='red', alpha=0.7)
        ax1.set_title('Portfolio Value Comparison')
        ax1.set_ylabel('Portfolio Value (R$)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Cumulative Returns
        ax2 = fig.add_subplot(2, 3, 2)
        portfolio_cumret = (self.results['portfolio_value'] / self.results['portfolio_value'].iloc[0] - 1) * 100
        bh_cumret = (self.results['buy_hold_value'] / self.results['buy_hold_value'].iloc[0] - 1) * 100
        
        ax2.plot(self.results['date'], portfolio_cumret, 
                label='Covered Call Strategy', linewidth=2, color='blue')
        ax2.plot(self.results['date'], bh_cumret, 
                label='Buy & Hold IBOV', linewidth=2, color='red', alpha=0.7)
        ax2.set_title('Cumulative Returns (%)')
        ax2.set_ylabel('Cumulative Return (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Drawdown Analysis
        ax3 = fig.add_subplot(2, 3, 3)
        portfolio_cummax = self.results['portfolio_value'].cummax()
        portfolio_dd = (self.results['portfolio_value'] - portfolio_cummax) / portfolio_cummax * 100
        
        bh_cummax = self.results['buy_hold_value'].cummax()
        bh_dd = (self.results['buy_hold_value'] - bh_cummax) / bh_cummax * 100
        
        ax3.fill_between(self.results['date'], portfolio_dd, 0, 
                        alpha=0.7, label='Covered Call Strategy', color='blue')
        ax3.fill_between(self.results['date'], bh_dd, 0, 
                        alpha=0.7, label='Buy & Hold IBOV', color='red')
        ax3.set_title('Drawdown Analysis')
        ax3.set_ylabel('Drawdown (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Rolling Volatility
        ax4 = fig.add_subplot(2, 3, 4)
        window = 30  # 30-day rolling window
        
        portfolio_returns = self.results['portfolio_value'].pct_change().dropna()
        bh_returns = self.results['buy_hold_value'].pct_change().dropna()
        
        portfolio_vol = portfolio_returns.rolling(window).std() * np.sqrt(252) * 100
        bh_vol = bh_returns.rolling(window).std() * np.sqrt(252) * 100
        
        # Ensure same length for plotting
        min_len = min(len(portfolio_vol), len(bh_vol), len(self.results['date']) - window)
        portfolio_vol = portfolio_vol.iloc[:min_len]
        bh_vol = bh_vol.iloc[:min_len]
        plot_dates = self.results['date'][window:window+min_len]
        
        ax4.plot(plot_dates, portfolio_vol, 
                label='Covered Call Strategy', linewidth=2, color='blue')
        ax4.plot(plot_dates, bh_vol, 
                label='Buy & Hold IBOV', linewidth=2, color='red', alpha=0.7)
        ax4.set_title(f'{window}-Day Rolling Volatility')
        ax4.set_ylabel('Annualized Volatility (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
        
        # Plot 5: Option Position Tracking
        ax5 = fig.add_subplot(2, 3, 5)
        has_option = self.results['has_option'].astype(int)
        ax5.fill_between(self.results['date'], has_option, 0, 
                        alpha=0.7, label='Option Position Active', color='green')
        ax5.set_title('Option Position Status')
        ax5.set_ylabel('Position Active (0/1)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.tick_params(axis='x', rotation=45)
        
        # Plot 6: Strike Prices Over Time
        ax6 = fig.add_subplot(2, 3, 6)
        strikes = self.results['strike'].dropna()
        strike_dates = self.results.loc[strikes.index, 'date']
        
        ax6.scatter(strike_dates, strikes, alpha=0.7, s=50, color='purple')
        ax6.plot(self.results['date'], self.results['bova_price'], 
                label='BOVA Price', linewidth=2, color='red', alpha=0.7)
        ax6.set_title('Strike Prices vs BOVA Price')
        ax6.set_ylabel('Price (R$)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.suptitle('Covered Call Strategy vs Buy & Hold BOVA - Performance Analysis', 
                    fontsize=16, y=0.98)
        
        return fig
    
    def print_summary(self):
        """Print comprehensive performance summary"""
        if self.results is None or len(self.results) == 0:
            raise ValueError("No backtest results available. Run backtest first.")
        
        metrics = self.calculate_performance_metrics()
        
        print("\n" + "="*80)
        print("🎯 COVERED CALL STRATEGY BACKTEST RESULTS - BOVA ETF")
        print("="*80)
        
        print(f"\n📅 Backtest Period: {pd.to_datetime(self.results['date'].iloc[0]).date()} to {pd.to_datetime(self.results['date'].iloc[-1]).date()}")
        print(f"📊 Total Days: {len(self.results)}")
        
        print(f"\n📈 PERFORMANCE COMPARISON:")
        print("-" * 50)
        
        for strategy, metrics_dict in metrics.items():
            if strategy == 'Outperformance':
                continue
            print(f"\n🔍 {strategy}:")
            for metric, value in metrics_dict.items():
                if 'Value' in metric:
                    print(f"   {metric}: R$ {value:,.2f}")
                elif 'Ratio' in metric:
                    print(f"   {metric}: {value:.3f}")
                else:
                    print(f"   {metric}: {value:.2f}")
        
        print(f"\n🏆 OUTPERFORMANCE ANALYSIS:")
        print("-" * 50)
        for metric, value in metrics['Outperformance'].items():
            print(f"   {metric}: {value:.2f}")
        
        print("\n✅ Backtest Summary Complete!")
        print("="*80)

def main():
    """Main backtesting function"""
    print("🚀 Covered Call Strategy Backtesting on BOVA ETF")
    print("=" * 60)
    
    try:
        # Initialize backtester
        backtester = CoveredCallBacktester()
        
        # Load data
        backtester.load_data()
        
        # Run backtest
        results = backtester.run_backtest(
            initial_capital=100000,
            risk_free_rate=0.10,
            min_days_to_expiry=7,
            max_days_to_expiry=30,
            roll_frequency_days=7
        )
        
        # Calculate and print performance metrics
        backtester.print_summary()
        
        # Create visualization
        fig = backtester.plot_results()
        
        # Save results
        results.to_csv("results/covered_call_backtest_results.csv", index=False)
        fig.savefig("results/covered_call_backtest_analysis.png", dpi=300, bbox_inches='tight')
        
        print("\n💾 Results saved:")
        print("   - results/covered_call_backtest_results.csv")
        print("   - results/covered_call_backtest_analysis.png")
        
        # Show plots
        plt.show()
        
        print("\n✅ Covered Call Backtest Complete!")
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        raise

if __name__ == "__main__":
    main()
