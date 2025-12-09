"""
Performance Metrics - Calculate trading strategy performance metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, Union

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.05) -> float:
    """Calculate Sharpe Ratio"""
    if returns.empty or returns.std() == 0:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / 252)
    return np.sqrt(252) * excess_returns.mean() / returns.std()

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.05) -> float:
    """Calculate Sortino Ratio"""
    if returns.empty:
        return 0.0
        
    excess_returns = returns - (risk_free_rate / 252)
    downside_returns = returns[returns < 0]
    
    if downside_returns.empty or downside_returns.std() == 0:
        return 0.0
        
    return np.sqrt(252) * excess_returns.mean() / downside_returns.std()

def calculate_max_drawdown(prices: pd.Series) -> float:
    """Calculate Maximum Drawdown"""
    if prices.empty:
        return 0.0
        
    peak = prices.expanding(min_periods=1).max()
    drawdown = (prices - peak) / peak
    return drawdown.min()

def calculate_win_rate(trades: pd.DataFrame) -> float:
    """Calculate Win Rate"""
    if trades.empty:
        return 0.0
        
    winning_trades = len(trades[trades['pnl'] > 0])
    return winning_trades / len(trades)

def calculate_profit_factor(trades: pd.DataFrame) -> float:
    """Calculate Profit Factor"""
    if trades.empty:
        return 0.0
        
    gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())
    
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
        
    return gross_profit / gross_loss

def calculate_metrics(daily_returns: pd.Series, trades: pd.DataFrame, 
                     initial_capital: float, final_capital: float) -> Dict[str, float]:
    """Calculate Comprehensive Performance Metrics"""
    
    total_return = (final_capital - initial_capital) / initial_capital
    annualized_return = total_return * (252 / len(daily_returns)) if len(daily_returns) > 0 else 0.0
    
    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'benchmark_return': 0.0, # Placeholder
        'sharpe_ratio': calculate_sharpe_ratio(daily_returns),
        'sortino_ratio': calculate_sortino_ratio(daily_returns),
        'max_drawdown': calculate_max_drawdown((1 + daily_returns).cumprod()),
        'win_rate': calculate_win_rate(trades),
        'profit_factor': calculate_profit_factor(trades),
        'total_trades': len(trades),
        'average_win': trades[trades['pnl'] > 0]['pnl'].mean() if not trades.empty else 0.0,
        'average_loss': trades[trades['pnl'] < 0]['pnl'].mean() if not trades.empty else 0.0,
        'largest_win': trades['pnl'].max() if not trades.empty else 0.0,
        'largest_loss': trades['pnl'].min() if not trades.empty else 0.0,
    }
    
    return metrics
