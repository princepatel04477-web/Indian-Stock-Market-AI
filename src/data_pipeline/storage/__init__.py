"""
Data Storage - Parquet and DuckDB storage backends
"""

from .parquet_storage import ParquetStorage
from .duckdb_storage import DuckDBStorage

__all__ = ['ParquetStorage', 'DuckDBStorage']
