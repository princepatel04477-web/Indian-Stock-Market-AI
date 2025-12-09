"""
DuckDB Storage - SQL-based analytics database
"""

import duckdb
import pandas as pd
from pathlib import Path
from typing import Optional, List
from datetime import datetime
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config.settings import database_config

logger = logging.getLogger(__name__)


class DuckDBStorage:
    """DuckDB storage for analytical queries"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or database_config.duckdb_path
        self.con = None
    
    def connect(self) -> duckdb.DuckDBPyConnection:
        """Get database connection"""
        if self.con is None:
            self.con = duckdb.connect(self.db_path)
            self._init_tables()
        return self.con
    
    def _init_tables(self):
        """Initialize database tables"""
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                symbol VARCHAR,
                timestamp TIMESTAMP,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT,
                PRIMARY KEY (symbol, timestamp)
            )
        """)
        
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS options_chain (
                symbol VARCHAR,
                underlying_price DOUBLE,
                strike DOUBLE,
                expiry DATE,
                ce_oi BIGINT,
                pe_oi BIGINT,
                ce_iv DOUBLE,
                pe_iv DOUBLE,
                timestamp TIMESTAMP,
                PRIMARY KEY (symbol, strike, expiry, timestamp)
            )
        """)
        
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS features (
                symbol VARCHAR,
                timestamp TIMESTAMP,
                feature_name VARCHAR,
                value DOUBLE,
                PRIMARY KEY (symbol, timestamp, feature_name)
            )
        """)
    
    def save_ohlcv(self, df: pd.DataFrame, symbol: str):
        """Save OHLCV data"""
        con = self.connect()
        df = df.copy()
        df['symbol'] = symbol
        
        con.execute("""
            INSERT OR REPLACE INTO ohlcv 
            SELECT symbol, timestamp, open, high, low, close, volume 
            FROM df
        """)
        logger.info(f"Saved {len(df)} OHLCV rows for {symbol}")
    
    def load_ohlcv(self, symbol: str, start_date: datetime = None,
                   end_date: datetime = None) -> pd.DataFrame:
        """Load OHLCV data"""
        con = self.connect()
        
        query = "SELECT * FROM ohlcv WHERE symbol = ?"
        params = [symbol]
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        
        query += " ORDER BY timestamp"
        
        return con.execute(query, params).df()
    
    def query(self, sql: str) -> pd.DataFrame:
        """Execute raw SQL query"""
        con = self.connect()
        return con.execute(sql).df()
    
    def list_symbols(self) -> List[str]:
        """List all symbols in database"""
        con = self.connect()
        result = con.execute("SELECT DISTINCT symbol FROM ohlcv").fetchall()
        return [r[0] for r in result]
    
    def close(self):
        """Close database connection"""
        if self.con:
            self.con.close()
            self.con = None
