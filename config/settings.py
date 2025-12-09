"""
Global Configuration Settings for Indian Stock AI Trading System
"""

import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

# Base Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
PARQUET_DIR = DATA_DIR / "parquet"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DIR, 
                  PARQUET_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


class TradingStrategy(Enum):
    """Supported trading strategies"""
    FNO = "fno"
    INTRADAY = "intraday"
    SWING = "swing"
    POSITIONAL = "positional"


class Timeframe(Enum):
    """Supported timeframes"""
    TICK = "tick"
    ONE_SECOND = "1s"
    ONE_MINUTE = "1m"
    FIVE_MINUTE = "5m"
    FIFTEEN_MINUTE = "15m"
    THIRTY_MINUTE = "30m"
    ONE_HOUR = "1h"
    FOUR_HOUR = "4h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1M"


class SignalType(Enum):
    """Trading signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class DataSourceConfig:
    """Configuration for data sources"""
    # NSE Official Data
    nse_base_url: str = "https://www.nseindia.com"
    nse_options_url: str = "https://www.nseindia.com/api/option-chain-indices"
    
    # Third-party APIs (configure your API keys in .env)
    alpha_vantage_key: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    quandl_key: str = os.getenv("QUANDL_API_KEY", "")
    
    # Data vendor configs (if using paid vendors)
    truedata_username: str = os.getenv("TRUEDATA_USERNAME", "")
    truedata_password: str = os.getenv("TRUEDATA_PASSWORD", "")
    
    # Rate limiting
    request_delay_seconds: float = 0.5
    max_retries: int = 3
    retry_delay_seconds: float = 2.0


@dataclass
class DatabaseConfig:
    """Database configuration"""
    # DuckDB (default for development)
    duckdb_path: str = str(DATA_DIR / "market_data.duckdb")
    
    # TimescaleDB (for production)
    timescale_host: str = os.getenv("TIMESCALE_HOST", "localhost")
    timescale_port: int = int(os.getenv("TIMESCALE_PORT", "5432"))
    timescale_db: str = os.getenv("TIMESCALE_DB", "indian_stocks")
    timescale_user: str = os.getenv("TIMESCALE_USER", "postgres")
    timescale_password: str = os.getenv("TIMESCALE_PASSWORD", "")
    
    # Redis (for caching)
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_db: int = int(os.getenv("REDIS_DB", "0"))


@dataclass
class ModelConfig:
    """Model configuration"""
    # LightGBM baseline
    lgbm_params: Dict = field(default_factory=lambda: {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_estimators": 1000,
        "early_stopping_rounds": 50
    })
    
    # XGBoost
    xgb_params: Dict = field(default_factory=lambda: {
        "objective": "multi:softprob",
        "num_class": 3,
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 1000,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "early_stopping_rounds": 50
    })
    
    # TFT (Temporal Fusion Transformer)
    tft_params: Dict = field(default_factory=lambda: {
        "hidden_size": 32,
        "attention_head_size": 4,
        "dropout": 0.1,
        "hidden_continuous_size": 16,
        "learning_rate": 0.001,
        "max_epochs": 50,
        "batch_size": 128,
        "gradient_clip_val": 0.1
    })
    
    # Paths
    quant_model_path: Path = MODELS_DIR / "quant"
    llm_model_path: Path = MODELS_DIR / "llm"


@dataclass
class LLMConfig:
    """LLM fine-tuning configuration"""
    # Base model (choose one that Ollama supports and you can fine-tune)
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    
    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    
    # Training
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    max_seq_length: int = 2048
    gradient_accumulation_steps: int = 4
    
    # Output
    output_dir: Path = MODELS_DIR / "llm" / "finetuned"
    merged_dir: Path = MODELS_DIR / "llm" / "merged"
    gguf_path: Path = MODELS_DIR / "llm" / "ollama" / "trader-llm.gguf"
    
    # Ollama
    ollama_model_name: str = "trader-llm"
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")


@dataclass
class FeatureConfig:
    """Feature engineering configuration"""
    # Technical indicator windows
    ema_periods: List[int] = field(default_factory=lambda: [8, 21, 50, 200])
    sma_periods: List[int] = field(default_factory=lambda: [10, 20, 50, 100, 200])
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    
    # Rolling statistics windows (in minutes for intraday)
    rolling_windows: List[int] = field(default_factory=lambda: [5, 15, 60, 240, 1440])
    
    # Lag features
    lag_periods: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10, 15, 30])
    
    # Options features
    options_strikes_range: int = 10  # +/- strikes from ATM
    iv_windows: List[int] = field(default_factory=lambda: [5, 15, 30])


@dataclass
class LabelConfig:
    """Label creation configuration"""
    # Forward return windows (in bars/minutes)
    forward_windows: Dict[str, int] = field(default_factory=lambda: {
        "intraday_15m": 15,
        "intraday_30m": 30,
        "intraday_1h": 60,
        "swing_1d": 1440,
        "swing_5d": 7200,
        "positional_1m": 43200,
    })
    
    # ATR multiplier for threshold
    atr_threshold_multiplier: float = 0.5
    
    # Fixed return thresholds as fallback
    return_threshold_buy: float = 0.005  # 0.5%
    return_threshold_sell: float = -0.005
    
    # Cost assumptions
    slippage_percent: float = 0.05
    commission_percent: float = 0.01
    stt_percent: float = 0.025  # Securities Transaction Tax


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    # Initial capital
    initial_capital: float = 1000000.0  # 10 Lakhs INR
    
    # Position sizing
    max_position_size_percent: float = 10.0
    risk_per_trade_percent: float = 2.0
    
    # Costs
    slippage_percent: float = 0.05
    commission_per_trade: float = 20.0
    stt_percent: float = 0.025
    
    # Risk management
    max_drawdown_limit: float = 0.20  # 20%
    daily_loss_limit: float = 0.03  # 3%
    
    # Walk-forward validation
    train_window_days: int = 252  # 1 year of trading days
    test_window_days: int = 63   # 3 months
    step_days: int = 21          # 1 month


@dataclass
class APIConfig:
    """API configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Authentication
    secret_key: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60
    
    # CORS
    allowed_origins: List[str] = field(default_factory=lambda: ["http://localhost:3000", "http://localhost:8000"])


# Market-specific configurations
@dataclass
class MarketConfig:
    """Indian market specific configuration"""
    timezone: str = "Asia/Kolkata"
    
    # Trading hours (IST)
    market_open_time: str = "09:15"
    market_close_time: str = "15:30"
    
    # Pre-market
    premarket_open: str = "09:00"
    premarket_close: str = "09:08"
    
    # Important indices
    indices: List[str] = field(default_factory=lambda: [
        "NIFTY 50", "NIFTY BANK", "NIFTY IT", "NIFTY FIN SERVICE",
        "NIFTY MIDCAP 50", "NIFTY SMLCAP 50", "NIFTY NEXT 50",
        "INDIA VIX"
    ])
    
    # F&O symbols (top liquid ones)
    fno_symbols: List[str] = field(default_factory=lambda: [
        "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
        "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT",
        "AXISBANK", "MARUTI", "BAJFINANCE", "ASIANPAINT", "HCLTECH",
        "WIPRO", "TATAMOTORS", "SUNPHARMA", "TECHM", "NTPC"
    ])
    
    # Expiry days
    weekly_expiry_day: str = "Thursday"
    monthly_expiry_day: str = "Thursday"  # Last Thursday
    
    # Circuit breaker levels
    circuit_limits: Dict = field(default_factory=lambda: {
        "upper_circuit": [0.05, 0.10, 0.15, 0.20],  # 5%, 10%, 15%, 20%
        "lower_circuit": [-0.05, -0.10, -0.15, -0.20]
    })


# Singleton instances
data_source_config = DataSourceConfig()
database_config = DatabaseConfig()
model_config = ModelConfig()
llm_config = LLMConfig()
feature_config = FeatureConfig()
label_config = LabelConfig()
backtest_config = BacktestConfig()
api_config = APIConfig()
market_config = MarketConfig()


# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "INFO"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(LOGS_DIR / "app.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "detailed",
            "level": "DEBUG"
        }
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console", "file"]
    }
}
