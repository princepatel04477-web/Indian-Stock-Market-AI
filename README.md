# 🇮🇳 Indian Stock AI - Trading Signal System

A comprehensive AI-powered trading signal system for Indian stock markets (NSE/BSE) supporting F&O, Intraday, Swing, and Positional trading strategies.

## 🎯 Project Overview

This system combines:
- **Quantitative Models** (LightGBM, XGBoost, Temporal Fusion Transformer) for signal generation
- **Fine-tuned LLM** (Ollama-compatible) for explainable trade reasoning
- **REST API & Web UI** for easy access to signals
- **Backtesting Engine** for strategy validation

## 📁 Project Structure

```
Indian_Stock_AI/
├── config/                     # Configuration files
│   ├── settings.py            # Global settings
│   └── model_config.yaml      # Model hyperparameters
├── data/                       # Data storage
│   ├── raw/                   # Raw market data
│   ├── processed/             # Cleaned & processed data
│   ├── features/              # Engineered features
│   └── parquet/               # Parquet files for fast access
├── src/                        # Source code
│   ├── data_pipeline/         # Data collection & processing
│   │   ├── collectors/        # Data source collectors
│   │   ├── processors/        # Data cleaning & normalization
│   │   └── storage/           # Database interactions
│   ├── features/              # Feature engineering
│   │   ├── technical.py       # Technical indicators
│   │   ├── options.py         # Options-specific features
│   │   └── macro.py           # Macro & sentiment features
│   ├── models/                # ML Models
│   │   ├── quant/             # Quantitative models
│   │   │   ├── lgbm_model.py  # LightGBM baseline
│   │   │   ├── xgb_model.py   # XGBoost model
│   │   │   └── tft_model.py   # Temporal Fusion Transformer
│   │   └── llm/               # LLM components
│   │       ├── finetune.py    # LoRA fine-tuning
│   │       ├── inference.py   # LLM inference
│   │       └── prompts.py     # Prompt templates
│   ├── backtesting/           # Backtesting engine
│   │   ├── engine.py          # Core backtest logic
│   │   ├── metrics.py         # Performance metrics
│   │   └── simulator.py       # Trade simulator
│   └── utils/                 # Utility functions
├── api/                        # REST API
│   ├── main.py                # FastAPI application
│   ├── routes/                # API endpoints
│   └── schemas/               # Pydantic models
├── web/                        # Web UI (SaaS prototype)
│   ├── static/                # CSS, JS, images
│   └── templates/             # HTML templates
├── notebooks/                  # Jupyter notebooks for exploration
├── models/                     # Saved model artifacts
│   ├── quant/                 # Quant model checkpoints
│   └── llm/                   # LLM fine-tuned weights
├── tests/                      # Unit & integration tests
├── scripts/                    # Utility scripts
└── docker/                     # Docker configuration
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Settings

Edit `config/settings.py` with your API keys and preferences.

### 3. Collect Data

```bash
python -m src.data_pipeline.run_pipeline
```

### 4. Train Models

```bash
# Train baseline model
python -m src.models.quant.train --model lgbm

# Fine-tune LLM
python -m src.models.llm.finetune --base-model mistral-7b
```

### 5. Run API Server

```bash
uvicorn api.main:app --reload --port 8000
```

### 6. Access Web UI

Open http://localhost:8000 in your browser.

## 📊 Supported Trading Strategies

| Strategy | Timeframe | Holding Period | Model Focus |
|----------|-----------|----------------|-------------|
| F&O | Various | Options expiry | IV, Greeks, OI |
| Intraday | 1m-15m | Same day | Price action, Volume |
| Swing | 1h-Daily | 2-10 days | Trend, Momentum |
| Positional | Daily-Weekly | 1-3 months | Fundamentals, Macro |

## ⚠️ Disclaimer

This is for **educational and personal use only**. Always:
- Use paper trading for evaluation
- Understand risks before trading
- Comply with SEBI regulations
- Never risk more than you can afford to lose

## 📄 License

MIT License - See LICENSE file for details.
