"""
Macro Data Collector - Economic indicators, FII/DII flows, sentiment
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import requests

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)


class MacroDataCollector(BaseCollector):
    """Collector for macro economic data and market sentiment"""
    
    def __init__(self):
        super().__init__("Macro")
        self.session = requests.Session()
        
    def fetch_ohlcv(self, symbol: str, start_date: datetime, 
                    end_date: datetime, interval: str = "1d") -> pd.DataFrame:
        """Not applicable for macro data"""
        return pd.DataFrame()
    
    def fetch_live_quote(self, symbol: str) -> Dict:
        """Not applicable for macro data"""
        return {}
    
    def fetch_india_vix(self, days: int = 30) -> pd.DataFrame:
        """Fetch India VIX historical data"""
        from .yfinance_collector import YFinanceCollector
        yf = YFinanceCollector()
        end = datetime.now()
        start = end - timedelta(days=days)
        return yf.fetch_ohlcv("INDIA VIX", start, end)
    
    def get_economic_calendar(self) -> List[Dict]:
        """Get upcoming economic events for India"""
        # Static calendar - update monthly
        events = [
            {"event": "RBI MPC Meeting", "dates": ["2024-02-08", "2024-04-05", "2024-06-07"]},
            {"event": "Union Budget", "dates": ["2024-02-01"]},
            {"event": "GDP Data", "dates": ["2024-02-29", "2024-05-31"]},
            {"event": "CPI Inflation", "dates": ["2024-02-12", "2024-03-12"]},
            {"event": "F&O Expiry", "dates": ["weekly_thursday"]},
        ]
        return events
    
    def get_market_holidays(self, year: int = 2024) -> List[datetime]:
        """Get NSE market holidays"""
        holidays_2024 = [
            "2024-01-26", "2024-03-08", "2024-03-25", "2024-03-29",
            "2024-04-11", "2024-04-14", "2024-04-17", "2024-04-21",
            "2024-05-01", "2024-05-23", "2024-06-17", "2024-07-17",
            "2024-08-15", "2024-10-02", "2024-11-01", "2024-11-15",
            "2024-12-25"
        ]
        return [datetime.strptime(d, "%Y-%m-%d") for d in holidays_2024]
    
    def is_expiry_day(self, date: datetime = None) -> bool:
        """Check if given date is F&O expiry"""
        date = date or datetime.now()
        return date.weekday() == 3  # Thursday
    
    def days_to_expiry(self, date: datetime = None) -> int:
        """Calculate days to next weekly expiry"""
        date = date or datetime.now()
        days_ahead = 3 - date.weekday()  # Thursday = 3
        if days_ahead < 0:
            days_ahead += 7
        return days_ahead
