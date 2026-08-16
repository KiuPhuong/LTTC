#!/usr/bin/env python
# main.py - Chạy toàn bộ pipeline

import yfinance as yf
import pandas as pd
import numpy as np
from utils.portfolio_optimizer import PortfolioOptimizer
from utils.backtester import Backtester

def main():
    print("="*80)
    print("PORTFOLIO OPTIMIZATION WITH TRANSFORMER PREDICTIONS")
    print("="*80)
    
    # 1. Load data
    print("\n[1/5] Loading data...")
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META']
    prices_df = pd.DataFrame()
    for ticker in tickers:
        df = yf.download(ticker, start='2015-01-01', end='2024-01-01')
        prices_df[ticker] = df['Close']
    
    # 2. Calculate returns
    print("\n[2/5] Calculating returns...")
    daily_returns = prices_df.pct_change().dropna()
    expected_returns = daily_returns.mean() * 252
    cov_matrix = daily_returns.cov() * 252
    
    # 3. Get Transformer predictions (demo)
    print("\n[3/5] Loading Transformer predictions...")
    predicted_growth = np.array([0.25, 0.20, 0.15, 0.30, 0.35, 0.18])
    
    # 4. Optimize portfolio
    print("\n[4/5] Optimizing portfolio...")
    optimizer = PortfolioOptimizer(expected_returns.values, cov_matrix.values)
    strategies = optimizer.get_all_strategies(predicted_growth)
    
    # 5. Backtest
    print("\n[5/5] Running backtest...")
    backtester = Backtester(daily_returns)
    results = {}
    for name, weights in strategies.items():
        result = backtester.run_backtest(weights)
        results[name] = result
    
    # Print summary
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    for name, result in results.items():
        print(f"\n📌 {name}")
        print("-"*50)
        for key, value in result['metrics'].items():
            print(f"{key}: {value:.2f}")
    
    print("\n✅ Done!")

if __name__ == "__main__":
    main()