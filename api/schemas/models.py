from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union
from datetime import datetime

# --- Request Models ---

class SignalRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol (e.g., RELIANCE)")
    timeframe: str = Field("15m", description="Timeframe for analysis")
    strategy: str = Field("intraday", description="Trading strategy type")

class BacktestRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    strategy: str = Field("intraday", description="Strategy to test")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")
    initial_capital: float = Field(100000.0, description="Initial capital for backtest")

# --- Response Models ---

class SignalResponse(BaseModel):
    symbol: str
    timestamp: datetime
    signal: str # BUY, SELL, HOLD
    confidence: float
    reasoning: str
    quant_score: float
    technical_indicators: Dict[str, float]
    entry_price: Optional[float]
    stop_loss: Optional[float]
    target_price: Optional[float]

class TradeRecord(BaseModel):
    timestamp: datetime
    symbol: str
    side: str
    quantity: int
    price: float
    pnl: float = 0.0

class BacktestMetrics(BaseModel):
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profit_factor: float

class BacktestResponse(BaseModel):
    symbol: str
    metrics: BacktestMetrics
    trades: List[TradeRecord]
    equity_curve: List[Dict[str, float]]

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime
