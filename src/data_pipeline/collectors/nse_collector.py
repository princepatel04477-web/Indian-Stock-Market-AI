"""
NSE Data Collector - Fetches data directly from NSE India website
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import json
import time
from pathlib import Path

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)


class NSEDataCollector(BaseCollector):
    """
    Collector for NSE India market data
    Handles indices, equities, and F&O data
    """
    
    BASE_URL = "https://www.nseindia.com"
    
    # API endpoints
    ENDPOINTS = {
        "index_quote": "/api/allIndices",
        "equity_quote": "/api/quote-equity",
        "option_chain_index": "/api/option-chain-indices",
        "option_chain_equity": "/api/option-chain-equities",
        "market_status": "/api/marketStatus",
        "fii_dii": "/api/fiidiiTradeReact",
        "historical_index": "/api/historical/indicesHistory",
        "historical_equity": "/api/historical/cm/equity",
        "stock_info": "/api/quote-equity",
        "chart_data": "/api/chart-databyindex",
    }
    
    # Headers to mimic browser
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.nseindia.com/",
        "X-Requested-With": "XMLHttpRequest",
        "Connection": "keep-alive",
    }
    
    def __init__(self):
        super().__init__("NSE")
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self._cookies_set = False
        
    def _set_cookies(self):
        """Initialize session cookies by visiting main page"""
        if not self._cookies_set:
            try:
                # Visit main page to get cookies
                response = self.session.get(
                    self.BASE_URL,
                    timeout=10
                )
                response.raise_for_status()
                self._cookies_set = True
                logger.debug("NSE cookies initialized")
            except Exception as e:
                logger.warning(f"Failed to set NSE cookies: {e}")
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make authenticated request to NSE API"""
        self._rate_limit()
        self._set_cookies()
        
        url = f"{self.BASE_URL}{endpoint}"
        
        for attempt in range(self.config.max_retries):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=15
                )
                
                if response.status_code == 401:
                    # Re-authenticate
                    self._cookies_set = False
                    self._set_cookies()
                    continue
                    
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"NSE request failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay_seconds * (attempt + 1))
                else:
                    raise
                    
        return {}
    
    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from NSE
        
        Note: NSE's free API has limited historical data
        For longer history, consider using yfinance or paid vendors
        """
        is_index = symbol.upper() in ["NIFTY 50", "NIFTY BANK", "NIFTY IT", 
                                        "NIFTY NEXT 50", "INDIA VIX"]
        
        if is_index:
            return self._fetch_index_history(symbol, start_date, end_date)
        else:
            return self._fetch_equity_history(symbol, start_date, end_date)
    
    def _fetch_index_history(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch historical data for indices"""
        
        # Format symbol for NSE API
        symbol_map = {
            "NIFTY 50": "NIFTY 50",
            "NIFTY BANK": "NIFTY BANK",
            "NIFTY IT": "NIFTY IT",
            "INDIA VIX": "INDIA VIX",
        }
        
        api_symbol = symbol_map.get(symbol.upper(), symbol)
        
        params = {
            "indexType": api_symbol,
            "from": start_date.strftime("%d-%m-%Y"),
            "to": end_date.strftime("%d-%m-%Y")
        }
        
        try:
            data = self._make_request(self.ENDPOINTS["historical_index"], params)
            
            if "data" not in data or "indexCloseOnlineRecords" not in data["data"]:
                logger.warning(f"No historical data found for {symbol}")
                return pd.DataFrame()
            
            records = data["data"]["indexCloseOnlineRecords"]
            
            df = pd.DataFrame(records)
            
            # Rename and format columns
            df = df.rename(columns={
                "EOD_TIMESTAMP": "timestamp",
                "EOD_OPEN_INDEX_VAL": "open",
                "EOD_HIGH_INDEX_VAL": "high",
                "EOD_LOW_INDEX_VAL": "low",
                "EOD_CLOSE_INDEX_VAL": "close",
            })
            
            # Add volume (indices don't have volume on NSE)
            df["volume"] = 0
            
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            
            return self.validate_data(df)
            
        except Exception as e:
            logger.error(f"Failed to fetch index history for {symbol}: {e}")
            return pd.DataFrame()
    
    def _fetch_equity_history(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch historical data for equities"""
        
        params = {
            "symbol": symbol.upper(),
            "from": start_date.strftime("%d-%m-%Y"),
            "to": end_date.strftime("%d-%m-%Y")
        }
        
        try:
            data = self._make_request(self.ENDPOINTS["historical_equity"], params)
            
            if "data" not in data:
                logger.warning(f"No historical data found for {symbol}")
                return pd.DataFrame()
            
            records = data["data"]
            df = pd.DataFrame(records)
            
            # Rename columns
            df = df.rename(columns={
                "CH_TIMESTAMP": "timestamp",
                "CH_OPENING_PRICE": "open",
                "CH_TRADE_HIGH_PRICE": "high",
                "CH_TRADE_LOW_PRICE": "low",
                "CH_CLOSING_PRICE": "close",
                "CH_TOT_TRADED_QTY": "volume",
                "VWAP": "vwap",
            })
            
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            
            return self.validate_data(df)
            
        except Exception as e:
            logger.error(f"Failed to fetch equity history for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_live_quote(self, symbol: str) -> Dict[str, Any]:
        """Fetch live quote for a symbol"""
        
        is_index = symbol.upper() in ["NIFTY 50", "NIFTY BANK", "NIFTY IT"]
        
        if is_index:
            return self._fetch_index_quote(symbol)
        else:
            return self._fetch_equity_quote(symbol)
    
    def _fetch_index_quote(self, symbol: str) -> Dict[str, Any]:
        """Fetch live quote for index"""
        try:
            data = self._make_request(self.ENDPOINTS["index_quote"])
            
            indices = data.get("data", [])
            
            for idx in indices:
                if idx.get("index", "").upper() == symbol.upper():
                    return {
                        "symbol": symbol,
                        "last_price": idx.get("last"),
                        "change": idx.get("percentChange"),
                        "open": idx.get("open"),
                        "high": idx.get("high"),
                        "low": idx.get("low"),
                        "prev_close": idx.get("previousClose"),
                        "timestamp": datetime.now()
                    }
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to fetch index quote for {symbol}: {e}")
            return {}
    
    def _fetch_equity_quote(self, symbol: str) -> Dict[str, Any]:
        """Fetch live quote for equity"""
        try:
            params = {"symbol": symbol.upper()}
            data = self._make_request(f"{self.ENDPOINTS['equity_quote']}?symbol={symbol.upper()}")
            
            if "priceInfo" in data:
                price_info = data["priceInfo"]
                return {
                    "symbol": symbol,
                    "last_price": price_info.get("lastPrice"),
                    "change": price_info.get("pChange"),
                    "open": price_info.get("open"),
                    "high": price_info.get("intraDayHighLow", {}).get("max"),
                    "low": price_info.get("intraDayHighLow", {}).get("min"),
                    "prev_close": price_info.get("previousClose"),
                    "vwap": price_info.get("vwap"),
                    "volume": data.get("securityWiseDP", {}).get("quantityTraded"),
                    "timestamp": datetime.now()
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to fetch equity quote for {symbol}: {e}")
            return {}
    
    def fetch_option_chain(self, symbol: str) -> pd.DataFrame:
        """
        Fetch option chain data for a symbol
        
        Returns DataFrame with strike, expiry, call/put data, IV, OI, etc.
        """
        is_index = symbol.upper() in ["NIFTY", "BANKNIFTY", "NIFTY 50", "NIFTY BANK"]
        
        endpoint = (
            self.ENDPOINTS["option_chain_index"]
            if is_index
            else self.ENDPOINTS["option_chain_equity"]
        )
        
        # Map symbol for API
        api_symbol = symbol.upper()
        if api_symbol == "NIFTY 50":
            api_symbol = "NIFTY"
        elif api_symbol == "NIFTY BANK":
            api_symbol = "BANKNIFTY"
        
        try:
            params = {"symbol": api_symbol}
            data = self._make_request(endpoint, params)
            
            if "records" not in data:
                logger.warning(f"No option chain data for {symbol}")
                return pd.DataFrame()
            
            records = data["records"]["data"]
            underlying = data["records"]["underlyingValue"]
            
            rows = []
            for record in records:
                strike = record.get("strikePrice")
                expiry = record.get("expiryDate")
                
                # Call data
                ce = record.get("CE", {})
                # Put data  
                pe = record.get("PE", {})
                
                rows.append({
                    "symbol": symbol,
                    "underlying_price": underlying,
                    "strike": strike,
                    "expiry": expiry,
                    # Call side
                    "ce_oi": ce.get("openInterest", 0),
                    "ce_chg_oi": ce.get("changeinOpenInterest", 0),
                    "ce_volume": ce.get("totalTradedVolume", 0),
                    "ce_iv": ce.get("impliedVolatility", 0),
                    "ce_ltp": ce.get("lastPrice", 0),
                    "ce_bid": ce.get("bidprice", 0),
                    "ce_ask": ce.get("askPrice", 0),
                    # Put side
                    "pe_oi": pe.get("openInterest", 0),
                    "pe_chg_oi": pe.get("changeinOpenInterest", 0),
                    "pe_volume": pe.get("totalTradedVolume", 0),
                    "pe_iv": pe.get("impliedVolatility", 0),
                    "pe_ltp": pe.get("lastPrice", 0),
                    "pe_bid": pe.get("bidprice", 0),
                    "pe_ask": pe.get("askPrice", 0),
                    # Derived
                    "pcr_oi": (pe.get("openInterest", 0) / ce.get("openInterest", 1)) 
                              if ce.get("openInterest", 0) > 0 else 0,
                    "timestamp": datetime.now()
                })
            
            df = pd.DataFrame(rows)
            
            # Parse expiry dates
            df["expiry"] = pd.to_datetime(df["expiry"], format="%d-%b-%Y")
            
            logger.info(f"Fetched option chain for {symbol}: {len(df)} strikes")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch option chain for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_fii_dii_data(self) -> pd.DataFrame:
        """Fetch FII/DII trading activity data"""
        try:
            data = self._make_request(self.ENDPOINTS["fii_dii"])
            
            if not data:
                return pd.DataFrame()
            
            # Parse the data structure
            rows = []
            for category in data:
                rows.append({
                    "category": category.get("category"),
                    "date": datetime.now().date(),
                    "buy_value": category.get("buyValue"),
                    "sell_value": category.get("sellValue"),
                    "net_value": category.get("netValue"),
                })
            
            df = pd.DataFrame(rows)
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch FII/DII data: {e}")
            return pd.DataFrame()
    
    def fetch_market_status(self) -> Dict[str, Any]:
        """Fetch current market status"""
        try:
            data = self._make_request(self.ENDPOINTS["market_status"])
            
            return {
                "status": data.get("marketStatus", {}).get("market"),
                "message": data.get("marketStatus", {}).get("marketStatusMessage"),
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch market status: {e}")
            return {"status": "unknown", "timestamp": datetime.now()}
    
    def get_all_fno_symbols(self) -> List[str]:
        """Get list of all F&O eligible symbols"""
        # This is a static list - NSE doesn't have a direct API for this
        # Update periodically from NSE circulars
        return self.market_config.fno_symbols
    
    def get_expiry_dates(self, symbol: str) -> List[datetime]:
        """Get upcoming expiry dates for a symbol"""
        try:
            df = self.fetch_option_chain(symbol)
            if df.empty:
                return []
            
            expiries = df["expiry"].unique()
            expiries = sorted([pd.Timestamp(e).to_pydatetime() for e in expiries])
            return expiries
            
        except Exception as e:
            logger.error(f"Failed to get expiry dates for {symbol}: {e}")
            return []


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    collector = NSEDataCollector()
    
    # Test index quote
    quote = collector.fetch_live_quote("NIFTY 50")
    print(f"NIFTY 50 Quote: {quote}")
    
    # Test option chain
    chain = collector.fetch_option_chain("NIFTY")
    print(f"Option Chain Shape: {chain.shape}")
    
    # Test FII/DII
    fii_dii = collector.fetch_fii_dii_data()
    print(f"FII/DII Data:\n{fii_dii}")
