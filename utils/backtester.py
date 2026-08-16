import numpy as np
import pandas as pd

class Backtester:
    """
    Lớp backtest cho chiến lược danh mục đầu tư
    """
    
    def __init__(self, daily_returns, initial_capital=10000, transaction_cost=0.001):
        """
        Args:
            daily_returns: DataFrame lợi nhuận hàng ngày
            initial_capital: Vốn ban đầu
            transaction_cost: Chi phí giao dịch (mặc định 0.1%)
        """
        self.daily_returns = daily_returns
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
    def run_backtest(self, weights):
        """
        Chạy backtest cho một chiến lược
        
        Returns:
            DataFrame với giá trị danh mục, lợi nhuận, v.v.
        """
        portfolio_returns = self.daily_returns @ weights
        portfolio_values = self.initial_capital * (1 + portfolio_returns).cumprod()
        
        # Tính các chỉ số
        metrics = self.calculate_metrics(portfolio_returns)
        
        return {
            'returns': portfolio_returns,
            'values': portfolio_values,
            'metrics': metrics
        }
    
    def calculate_metrics(self, returns):
        """Tính các chỉ số đánh giá nâng cao"""
        # Lợi nhuận hàng năm
        annual_return = returns.mean() * 252
        
        # Volatility hàng năm
        annual_vol = returns.std() * np.sqrt(252)
        
        # Sharpe Ratio
        risk_free_rate = 0.02
        sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
        
        # Sortino Ratio (chỉ xét rủi ro giảm)
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 1
        sortino = (annual_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
        
        # Maximum Drawdown
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdown = (rolling_max - cum_returns) / rolling_max
        max_drawdown = drawdown.max()
        
        # Calmar Ratio
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win Rate
        win_rate = (returns > 0).mean()
        
        return {
            'Annual Return (%)': annual_return * 100,
            'Volatility (%)': annual_vol * 100,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Max Drawdown (%)': max_drawdown * 100,
            'Calmar Ratio': calmar,
            'Win Rate (%)': win_rate * 100
        }