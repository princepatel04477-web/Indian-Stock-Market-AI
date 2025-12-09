"""
Temporal Fusion Transformer - Deep learning for multi-horizon forecasting
"""

import torch
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TFTModel:
    """Temporal Fusion Transformer for time series forecasting"""
    
    def __init__(self, params: Dict = None):
        self.params = params or {
            'hidden_size': 32,
            'attention_head_size': 4,
            'dropout': 0.1,
            'hidden_continuous_size': 16,
            'learning_rate': 0.001,
            'max_epochs': 50,
            'batch_size': 128,
            'max_encoder_length': 96,
            'max_prediction_length': 12
        }
        self.model = None
        self.trainer = None
        self.training_data = None
    
    def prepare_data(self, df: pd.DataFrame, 
                     target_col: str = 'close',
                     time_col: str = 'timestamp',
                     group_col: str = 'symbol') -> 'TimeSeriesDataSet':
        """Prepare data for TFT"""
        try:
            from pytorch_forecasting import TimeSeriesDataSet
            from pytorch_forecasting.data import GroupNormalizer
        except ImportError:
            logger.error("pytorch-forecasting not installed. Install with: pip install pytorch-forecasting")
            return None
        
        df = df.copy()
        
        # Ensure required columns
        if time_col not in df.columns:
            raise ValueError(f"Time column {time_col} not found")
        
        # Add time index
        df = df.sort_values(time_col)
        df['time_idx'] = range(len(df))
        
        # Add group if not present
        if group_col not in df.columns:
            df[group_col] = 'default'
        
        # Define columns
        time_varying_known = ['time_idx']
        time_varying_unknown = [target_col]
        
        # Add available features
        feature_cols = ['open', 'high', 'low', 'volume', 'returns', 
                       'rsi', 'macd', 'atr', 'ema_8', 'ema_21']
        time_varying_unknown.extend([c for c in feature_cols if c in df.columns])
        
        # Create dataset
        self.training_data = TimeSeriesDataSet(
            df,
            time_idx='time_idx',
            target=target_col,
            group_ids=[group_col],
            max_encoder_length=self.params['max_encoder_length'],
            max_prediction_length=self.params['max_prediction_length'],
            time_varying_known_reals=time_varying_known,
            time_varying_unknown_reals=time_varying_unknown,
            target_normalizer=GroupNormalizer(groups=[group_col])
        )
        
        return self.training_data
    
    def train(self, df: pd.DataFrame, 
              val_df: pd.DataFrame = None) -> 'TFTModel':
        """Train TFT model"""
        try:
            from pytorch_forecasting import TemporalFusionTransformer
            from pytorch_lightning.callbacks import EarlyStopping
        except ImportError:
            logger.error("pytorch-forecasting not installed")
            return self
        
        # Prepare data
        train_data = self.prepare_data(df)
        if train_data is None:
            return self
        
        train_loader = train_data.to_dataloader(
            train=True, 
            batch_size=self.params['batch_size'],
            num_workers=0
        )
        
        # Create model
        self.model = TemporalFusionTransformer.from_dataset(
            train_data,
            hidden_size=self.params['hidden_size'],
            attention_head_size=self.params['attention_head_size'],
            dropout=self.params['dropout'],
            hidden_continuous_size=self.params['hidden_continuous_size'],
            learning_rate=self.params['learning_rate'],
            log_interval=10
        )
        
        # Train
        early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min')
        self.trainer = pl.Trainer(
            max_epochs=self.params['max_epochs'],
            accelerator='auto',
            callbacks=[early_stop],
            enable_progress_bar=True
        )
        
        if val_df is not None:
            val_data = self.prepare_data(val_df)
            val_loader = val_data.to_dataloader(train=False, batch_size=self.params['batch_size'])
            self.trainer.fit(self.model, train_loader, val_loader)
        else:
            self.trainer.fit(self.model, train_loader)
        
        logger.info("TFT training complete")
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict future values"""
        if self.model is None or self.training_data is None:
            raise ValueError("Model not trained")
        
        # Create prediction dataset
        pred_data = self.training_data.from_dataset(
            self.training_data, 
            df, 
            predict=True
        )
        pred_loader = pred_data.to_dataloader(train=False, batch_size=64)
        
        predictions = self.model.predict(pred_loader)
        return predictions.numpy()
    
    def save(self, path: str):
        """Save model checkpoint"""
        if self.model is None:
            return
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        self.trainer.save_checkpoint(str(path))
        logger.info(f"Saved TFT model to {path}")
    
    def load(self, path: str) -> 'TFTModel':
        """Load model checkpoint"""
        try:
            from pytorch_forecasting import TemporalFusionTransformer
            self.model = TemporalFusionTransformer.load_from_checkpoint(path)
            logger.info(f"Loaded TFT model from {path}")
        except Exception as e:
            logger.error(f"Failed to load TFT model: {e}")
        return self
    
    def get_attention_weights(self, df: pd.DataFrame) -> Dict:
        """Get attention weights for interpretability"""
        if self.model is None:
            return {}
        
        # This would return attention patterns from the model
        # Useful for understanding which features/timestamps are important
        return {'attention': 'Implementation depends on prediction batch'}
