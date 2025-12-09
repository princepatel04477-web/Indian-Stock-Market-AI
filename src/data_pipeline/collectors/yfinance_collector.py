"""
YFinance Data Collector - Uses Yahoo Finance for historical data
Best for getting longer historical data (2-5 years)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)


class YFinanceCollector(BaseCollector):
    """
    Collector using Yahoo Finance API via yfinance library
    Good for historical data, limited for real-time
    """
    
    # Symbol suffix for Indian stocks
    INDIA_SUFFIX = ".NS"  # NSE
    BSE_SUFFIX = ".BO"    # BSE
    
    # Index mapping
    INDEX_SYMBOLS = {
        "NIFTY 50": "^NSEI",
        "NIFTY BANK": "^NSEBANK",
        "NIFTY IT": "^CNXIT",
        "NIFTY NEXT 50": "^NSMIDCP",
        "INDIA VIX": "^INDIAVIX",
        "SENSEX": "^BSESN",
    }
    
    # Interval mapping
    INTERVAL_MAP = {
        "1m": "1m",
        "2m": "2m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "60m",
        "1d": "1d",
        "1w": "1wk",
        "1M": "1mo",
    }
    
    def __init__(self, exchange: str = "NSE"):
        super().__init__("YFinance")
        self.exchange = exchange
        self.suffix = self.INDIA_SUFFIX if exchange == "NSE" else self.BSE_SUFFIX
        
    def _format_symbol(self, symbol: str) -> str:
        """Format symbol for Yahoo Finance"""
        symbol = symbol.upper().strip()
        
        # Check if it's an index
        if symbol in self.INDEX_SYMBOLS:
            return self.INDEX_SYMBOLS[symbol]
        
        # Already has suffix
        if symbol.endswith(".NS") or symbol.endswith(".BO"):
            return symbol
            
        # Add suffix
        return f"{symbol}{self.suffix}"
    
    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from Yahoo Finance
        
        Args:
            symbol: Stock/Index symbol
            start_date: Start date
            end_date: End date
            interval: Time interval (1m, 5m, 15m, 1h, 1d, 1w, 1M)
            
        Returns:
            DataFrame with OHLCV data
        """
        self._rate_limit()
        
        yf_symbol = self._format_symbol(symbol)
        yf_interval = self.INTERVAL_MAP.get(interval, interval)
        
        try:
            ticker = yf.Ticker(yf_symbol)
            
            # Fetch data
            df = ticker.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
                interval=yf_interval,
                auto_adjust=True,  # Adjust for splits/dividends
                prepost=False
            )
            
            if df.empty:
                logger.warning(f"No data found for {symbol} ({yf_symbol})")
                return pd.DataFrame()
            
            # Reset index and rename
            df = df.reset_index()
            df = df.rename(columns={
                "Date": "timestamp",
                "Datetime": "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            })
            
            # Keep only required columns
            cols = ["timestamp", "open", "high", "low", "close", "volume"]
            df = df[[c for c in cols if c in df.columns]]
            
            # Add volume if missing (for indices)
            if "volume" not in df.columns:
                df["volume"] = 0
            
            df = self.validate_data(df)
            self._log_fetch(symbol, start_date, end_date, len(df))
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_live_quote(self, symbol: str) -> Dict[str, Any]:
        """Fetch current quote for a symbol"""
        self._rate_limit()
        
        yf_symbol = self._format_symbol(symbol)
        
        try:
            ticker = yf.Ticker(yf_symbol)
            info = ticker.info
            
            return {
                "symbol": symbol,
                "last_price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "change": info.get("regularMarketChangePercent"),
                "open": info.get("regularMarketOpen"),
                "high": info.get("regularMarketDayHigh"),
                "low": info.get("regularMarketDayLow"),
                "prev_close": info.get("previousClose") or info.get("regularMarketPreviousClose"),
                "volume": info.get("regularMarketVolume"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch quote for {symbol}: {e}")
            return {}
    
    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
        max_workers: int = 5
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols in parallel
        
        Args:
            symbols: List of symbols
            start_date: Start date
            end_date: End date
            interval: Time interval
            max_workers: Max parallel downloads
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.fetch_ohlcv, symbol, start_date, end_date, interval
                ): symbol 
                for symbol in symbols
            }
            
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    df = future.result()
                    if not df.empty:
                        results[symbol] = df
                except Exception as e:
                    logger.error(f"Error fetching {symbol}: {e}")
        
        logger.info(f"Fetched data for {len(results)}/{len(symbols)} symbols")
        return results
    
    def fetch_company_info(self, symbol: str) -> Dict[str, Any]:
        """Fetch company fundamental information"""
        yf_symbol = self._format_symbol(symbol)
        
        try:
            ticker = yf.Ticker(yf_symbol)
            info = ticker.info
            
            return {
                "symbol": symbol,
                "name": info.get("longName") or info.get("shortName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "pb_ratio": info.get("priceToBook"),
                "dividend_yield": info.get("dividendYield"),
                "beta": info.get("beta"),
                "52w_high": info.get("fiftyTwoWeekHigh"),
                "52w_low": info.get("fiftyTwoWeekLow"),
                "avg_volume": info.get("averageVolume"),
                "shares_outstanding": info.get("sharesOutstanding"),
                "float_shares": info.get("floatShares"),
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch company info for {symbol}: {e}")
            return {}
    
    def fetch_financials(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Fetch company financials"""
        yf_symbol = self._format_symbol(symbol)
        
        try:
            ticker = yf.Ticker(yf_symbol)
            
            return {
                "income_statement": ticker.income_stmt,
                "balance_sheet": ticker.balance_sheet,
                "cash_flow": ticker.cashflow,
                "quarterly_income": ticker.quarterly_income_stmt,
                "quarterly_balance": ticker.quarterly_balance_sheet,
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch financials for {symbol}: {e}")
            return {}
    
    def fetch_dividends_splits(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Fetch dividend and split history"""
        yf_symbol = self._format_symbol(symbol)
        
        try:
            ticker = yf.Ticker(yf_symbol)
            
            return {
                "dividends": ticker.dividends.reset_index() if not ticker.dividends.empty else pd.DataFrame(),
                "splits": ticker.splits.reset_index() if not ticker.splits.empty else pd.DataFrame(),
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch dividends/splits for {symbol}: {e}")
            return {}
    
    def fetch_intraday_data(
        self,
        symbol: str,
        period: str = "5d",
        interval: str = "5m"
    ) -> pd.DataFrame:
        """
        Fetch recent intraday data
        
        Note: Yahoo Finance limits intraday history:
        - 1m data: last 7 days
        - 2m-90m data: last 60 days
        
        Args:
            symbol: Stock symbol
            period: Data period (1d, 5d, 1mo, 3mo)
            interval: Bar interval (1m, 5m, 15m, 30m, 1h)
        """
        self._rate_limit()
        
        yf_symbol = self._format_symbol(symbol)
        yf_interval = self.INTERVAL_MAP.get(interval, interval)
        
        try:
            ticker = yf.Ticker(yf_symbol)
            
            df = ticker.history(
                period=period,
                interval=yf_interval,
                prepost=False
            )
            
            if df.empty:
                return pd.DataFrame()
            
            df = df.reset_index()
            df = df.rename(columns={
                "Datetime": "timestamp",
                "Date": "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            })
            
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            
            return self.validate_data(df)
            
        except Exception as e:
            logger.error(f"Failed to fetch intraday data for {symbol}: {e}")
            return pd.DataFrame()
    
    def download_bulk_history(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Use yfinance bulk download for efficiency
        
        Returns MultiIndex DataFrame with symbols
        """
        # Format symbols
        yf_symbols = [self._format_symbol(s) for s in symbols]
        
        try:
            df = yf.download(
                tickers=yf_symbols,
                start=start_date,
                end=end_date,
                interval=interval,
                group_by='ticker',
                auto_adjust=True,
                threads=True
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Bulk download failed: {e}")
            return pd.DataFrame()


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    collector = YFinanceCollector()
    
    # Test historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    df = collector.fetch_ohlcv("RELIANCE", start_date, end_date)
    print(f"RELIANCE Historical Data: {df.shape}")
    print(df.tail())
    
    # Test index data
    df_nifty = collector.fetch_ohlcv("NIFTY 50", start_date, end_date)
    print(f"NIFTY 50 Data: {df_nifty.shape}")
    
    # Test intraday
    df_intraday = collector.fetch_intraday_data("TCS", period="5d", interval="15m")
    print(f"TCS Intraday: {df_intraday.shape}")
