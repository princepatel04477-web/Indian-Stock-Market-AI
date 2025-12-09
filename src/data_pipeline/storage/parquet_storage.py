"""
Parquet Storage - Fast columnar storage for time series data
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Optional, List
from datetime import datetime
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config.settings import PARQUET_DIR

logger = logging.getLogger(__name__)


class ParquetStorage:
    """Parquet file storage for market data"""
    
    def __init__(self, base_dir: Path = PARQUET_DIR):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_path(self, symbol: str, data_type: str, date: Optional[str] = None) -> Path:
        """Get file path for symbol and data type"""
        path = self.base_dir / data_type / symbol
        path.mkdir(parents=True, exist_ok=True)
        
        if date:
            return path / f"{date}.parquet"
        return path / "data.parquet"
    
    def save(self, df: pd.DataFrame, symbol: str, data_type: str = "ohlcv",
             partition_by_date: bool = True) -> Path:
        """Save DataFrame to parquet"""
        if df.empty:
            logger.warning(f"Empty DataFrame for {symbol}")
            return None
        
        if partition_by_date and 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp']).dt.date.astype(str)
            for date, group in df.groupby('date'):
                path = self._get_path(symbol, data_type, date)
                group.drop(columns=['date']).to_parquet(path, index=False)
            logger.info(f"Saved {len(df)} rows for {symbol} (partitioned)")
            return self.base_dir / data_type / symbol
        else:
            path = self._get_path(symbol, data_type)
            df.to_parquet(path, index=False)
            logger.info(f"Saved {len(df)} rows to {path}")
            return path
    
    def load(self, symbol: str, data_type: str = "ohlcv",
             start_date: Optional[datetime] = None,
             end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Load data from parquet"""
        path = self.base_dir / data_type / symbol
        
        if not path.exists():
            logger.warning(f"No data found for {symbol}")
            return pd.DataFrame()
        
        # Load all parquet files
        if path.is_dir():
            files = list(path.glob("*.parquet"))
            if not files:
                return pd.DataFrame()
            
            dfs = [pd.read_parquet(f) for f in files]
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = pd.read_parquet(path)
        
        # Filter by date range
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            if start_date:
                df = df[df['timestamp'] >= start_date]
            if end_date:
                df = df[df['timestamp'] <= end_date]
        
        return df.sort_values('timestamp').reset_index(drop=True) if 'timestamp' in df.columns else df
    
    def list_symbols(self, data_type: str = "ohlcv") -> List[str]:
        """List all available symbols"""
        path = self.base_dir / data_type
        if not path.exists():
            return []
        return [d.name for d in path.iterdir() if d.is_dir()]
    
    def delete(self, symbol: str, data_type: str = "ohlcv"):
        """Delete data for a symbol"""
        import shutil
        path = self.base_dir / data_type / symbol
        if path.exists():
            shutil.rmtree(path)
            logger.info(f"Deleted data for {symbol}")
