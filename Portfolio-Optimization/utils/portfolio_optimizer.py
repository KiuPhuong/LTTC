import numpy as np
import pandas as pd
from scipy.optimize import minimize

class PortfolioOptimizer:
    """
    Lớp tối ưu danh mục đầu tư với các chiến lược khác nhau
    """
    
    def __init__(self, expected_returns, cov_matrix, risk_free_rate=0.02):
        """
        Args:
            expected_returns: Lợi nhuận kỳ vọng của từng tài sản
            cov_matrix: Ma trận hiệp phương sai
            risk_free_rate: Lãi suất phi rủi ro (mặc định 2%)
        """
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(expected_returns)
    
    def portfolio_performance(self, weights):
        """Tính lợi nhuận, rủi ro và Sharpe Ratio"""
        ret = np.sum(weights * self.expected_returns)
        vol = np.sqrt(weights.T @ self.cov_matrix @ weights)
        sharpe = (ret - self.risk_free_rate) / vol
        return ret, vol, sharpe
    
    def equal_weight(self):
        """Chiến lược phân bổ đều"""
        return np.ones(self.n_assets) / self.n_assets
    
    def minimum_variance(self):
        """Chiến lược phương sai tối thiểu"""
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        def objective(w):
            return w.T @ self.cov_matrix @ w
        
        initial = np.ones(self.n_assets) / self.n_assets
        result = minimize(objective, initial, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        return result.x
    
    def maximum_sharpe(self):
        """Chiến lược Sharpe Ratio tối đa"""
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        def objective(w):
            ret = np.sum(w * self.expected_returns)
            vol = np.sqrt(w.T @ self.cov_matrix @ w)
            return -(ret - self.risk_free_rate) / vol
        
        initial = np.ones(self.n_assets) / self.n_assets
        result = minimize(objective, initial, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        return result.x
    
    def risk_parity(self, max_iter=1000):
        """Chiến lược cân bằng rủi ro"""
        weights = np.ones(self.n_assets) / self.n_assets
        for _ in range(max_iter):
            portfolio_vol = np.sqrt(weights.T @ self.cov_matrix @ weights)
            marginal_risk = self.cov_matrix @ weights / portfolio_vol
            weights = (1 / marginal_risk) / np.sum(1 / marginal_risk)
        return weights
    
    def ml_allocation(self, predicted_growth):
        """
        Chiến lược phân bổ dựa trên ML (dự đoán từ Transformer)
        
        Args:
            predicted_growth: Tăng trưởng dự đoán từ mô hình Transformer
        """
        # Điều chỉnh trọng số theo tăng trưởng dự đoán
        ml_weights = predicted_growth / predicted_growth.sum()
        
        # Kết hợp với Risk Parity để ổn định
        rp_weights = self.risk_parity()
        hybrid = (ml_weights * rp_weights) / np.sum(ml_weights * rp_weights)
        
        return hybrid
    
    def get_all_strategies(self, predicted_growth=None):
        """Lấy tất cả chiến lược"""
        strategies = {
            'Equal Weight': self.equal_weight(),
            'Minimum Variance': self.minimum_variance(),
            'Maximum Sharpe': self.maximum_sharpe(),
            'Risk Parity': self.risk_parity()
        }
        
        if predicted_growth is not None:
            strategies['ML Allocation'] = self.ml_allocation(predicted_growth)
        
        return strategies