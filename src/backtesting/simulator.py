"""
Trade Simulator - Execute trades and track portfolio state
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Trade:
    timestamp: datetime
    symbol: str
    side: str # 'BUY' or 'SELL'
    quantity: int
    price: float
    commission: float
    pnl: float = 0.0
    exit_price: float = 0.0
    exit_time: datetime = None
    duration: int = 0

class Portfolio:
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Dict] = {} # symbol -> {quantity, avg_price}
        self.equity_curve = []
        
    @property
    def total_value(self) -> float:
        position_value = sum(pos['quantity'] * pos['current_price'] for pos in self.positions.values())
        return self.cash + position_value

    def update_price(self, symbol: str, price: float):
        if symbol in self.positions:
            self.positions[symbol]['current_price'] = price
            
    def get_position(self, symbol: str) -> int:
        return self.positions.get(symbol, {}).get('quantity', 0)

class TradeSimulator:
    def __init__(self, initial_capital: float = 100000.0, 
                 commission_pct: float = 0.001, 
                 slippage_pct: float = 0.001):
        self.portfolio = Portfolio(initial_capital)
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.trades: List[Trade] = []
        self.history: List[Dict] = []
        
    def calculate_cost(self, price: float, quantity: int) -> tuple[float, float]:
        """Calculate execution price and commission"""
        slippage = price * self.slippage_pct
        exec_price = price + slippage if quantity > 0 else price - slippage
        value = abs(quantity * exec_price)
        commission = value * self.commission_pct
        return exec_price, commission
        
    def execute_order(self, timestamp: datetime, symbol: str, 
                     side: str, quantity: int, price: float) -> Optional[Trade]:
        """
        Execute an order
        side: 'BUY' or 'SELL'
        quantity: positive integer
        """
        if quantity <= 0:
            return None
            
        real_qty = quantity if side == 'BUY' else -quantity
        exec_price, commission = self.calculate_cost(price, real_qty)
        cost = (abs(real_qty) * exec_price) + commission
        
        # Check if enough cash for BUY
        if side == 'BUY' and self.portfolio.cash < cost:
            return None # Insufficient funds
            
        # Check if enough position for SELL (assuming no shorting for now, or handle appropriately)
        # Simple long-only logic for checking 'SELL' validity:
        current_pos = self.portfolio.get_position(symbol)
        if side == 'SELL' and current_pos < quantity:
            # Adjust to close existing position
            quantity = current_pos
            real_qty = -quantity
            if quantity == 0:
                return None
                
        # Update Portfolio
        self.portfolio.cash -= (real_qty * exec_price) + commission
        
        # Track Trade
        trade = None
        if side == 'SELL':
            # Closing position - Calculate PnL
            # Assuming FIFO or Average Cost - simplified to Average Cost here
            avg_entry = self.portfolio.positions[symbol]['avg_price']
            pnl = (exec_price - avg_entry) * quantity - commission
            
            # Find matching entry (simplified)
            # In a full system, you'd match against specific open trades
            trade = Trade(
                timestamp=timestamp,
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=avg_entry, # Visualizing as closing the trade
                exit_price=exec_price,
                exit_time=timestamp,
                commission=commission,
                pnl=pnl
            )
            self.trades.append(trade)
            
            # Update position
            new_qty = current_pos - quantity
            if new_qty == 0:
                del self.portfolio.positions[symbol]
            else:
                self.portfolio.positions[symbol]['quantity'] = new_qty
        else:
            # BUY - New Position or adding
            current_pos = self.portfolio.get_position(symbol)
            current_avg = self.portfolio.positions.get(symbol, {}).get('avg_price', 0)
            
            new_cost_basis = (current_pos * current_avg) + (quantity * exec_price)
            new_total_qty = current_pos + quantity
            new_avg = new_cost_basis / new_total_qty
            
            self.portfolio.positions[symbol] = {
                'quantity': new_total_qty,
                'avg_price': new_avg,
                'current_price': exec_price
            }
            
        return trade
        
    def update_market(self, timestamp: datetime, prices: Dict[str, float]):
        """Update current market prices and equity"""
        for symbol, price in prices.items():
            self.portfolio.update_price(symbol, price)
            
        self.portfolio.equity_curve.append({
            'timestamp': timestamp,
            'equity': self.portfolio.total_value,
            'cash': self.portfolio.cash
        })
