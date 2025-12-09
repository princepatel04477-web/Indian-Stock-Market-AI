"""
Backtest Engine - Core logic for running backtests
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, Callable
from datetime import datetime

from .simulator import TradeSimulator
from .metrics import calculate_metrics
from ..utils.logger import setup_logger

logger = setup_logger('backtest_engine')

class BacktestEngine:
    def __init__(self, initial_capital: float = 100000.0):
        self.simulator = TradeSimulator(initial_capital)
        self.results = {}
        
    def run(self, data: pd.DataFrame, strategy_fn: Callable, symbol: str) -> Dict:
        """
        Run backtest on a single symbol
        
        Args:
            data: DataFrame with timestamp, open, high, low, close
            strategy_fn: Function that takes (current_row, lookback_data) and returns signal
                        signal: 1 (buy), -1 (sell), 0 (hold)
            symbol: Ticker symbol
        """
        logger.info(f"Starting backtest for {symbol}")
        
        # Sort data
        df = data.sort_values('timestamp').reset_index(drop=True)
        
        # Iterate
        # In a real engine, we might mask future data more strictly
        # Here we iterate row-by-row
        
        position = 0
        
        for i in range(50, len(df)): # Start after some warm-up
            current_row = df.iloc[i]
            lookback = df.iloc[:i+1] # Includes current candle for closing price
            timestamp = current_row['timestamp']
            price = current_row['close']
            
            # Update simulator with latest price
            self.simulator.update_market(timestamp, {symbol: price})
            
            # Get Signal
            # strategy_fn should ideally rely only on i-1 data to avoid lookahead bias
            # passing lookback data ending at i, but it's up to strategy to index correctly
            signal = strategy_fn(lookback)
            
            # Execute logic
            current_pos = self.simulator.portfolio.get_position(symbol)
            
            if signal == 1 and current_pos == 0:
                # Buy Entry
                # Calculate quantity based on risk management - simplified here to fixed % or max cash
                cash = self.simulator.portfolio.cash
                qty = int((cash * 0.95) / price) # 95% of cash
                self.simulator.execute_order(timestamp, symbol, 'BUY', qty, price)
                
            elif signal == -1 and current_pos > 0:
                # Sell Exit
                self.simulator.execute_order(timestamp, symbol, 'SELL', current_pos, price)
                
        # Close all positions at end
        final_row = df.iloc[-1]
        final_pos = self.simulator.portfolio.get_position(symbol)
        if final_pos > 0:
            self.simulator.execute_order(final_row['timestamp'], symbol, 'SELL', final_pos, final_row['close'])
            
        # Compile results
        equity_df = pd.DataFrame(self.simulator.portfolio.equity_curve)
        if not equity_df.empty:
            equity_df = equity_df.set_index('timestamp')
            daily_returns = equity_df['equity'].pct_change().dropna()
        else:
            daily_returns = pd.Series()
            
        trades_df = pd.DataFrame([vars(t) for t in self.simulator.trades])
        
        metrics = calculate_metrics(
            daily_returns, 
            trades_df, 
            self.simulator.portfolio.initial_capital,
            self.simulator.portfolio.total_value
        )
        
        results = {
            'metrics': metrics,
            'trades': trades_df,
            'equity_curve': equity_df,
            'final_value': self.simulator.portfolio.total_value
        }
        
        self.results = results
        logger.info(f"Backtest completed. Final Value: {results['final_value']:.2f}")
        return results

    def get_report(self):
        """Generate summary report"""
        if not self.results:
            return "No results available"
            
        metrics = self.results['metrics']
        return f"""
        Backtest Report
        ===============
        Total Return: {metrics['total_return']:.2%}
        Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
        Max Drawdown: {metrics['max_drawdown']:.2%}
        Win Rate:     {metrics['win_rate']:.2%}
        Total Trades: {metrics['total_trades']}
        """
