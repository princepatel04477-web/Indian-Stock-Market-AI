"""
Options Features - Compute options-specific features for F&O trading
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class OptionsFeatures:
    """Calculate options-specific features"""
    
    def __init__(self):
        self.strikes_range = 10
    
    def compute_all(self, chain: pd.DataFrame) -> pd.DataFrame:
        """Compute all options features from chain data"""
        df = chain.copy()
        
        df = self.add_iv_features(df)
        df = self.add_oi_features(df)
        df = self.add_greek_features(df)
        
        return df
    
    def add_iv_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add IV-based features"""
        # ATM IV
        if 'moneyness' in df.columns:
            atm_mask = abs(df['moneyness'] - 1) < 0.02
            if atm_mask.any():
                df['atm_iv'] = df.loc[atm_mask, ['ce_iv', 'pe_iv']].mean().mean()
        
        # IV skew
        df['iv_skew'] = df['pe_iv'] - df['ce_iv']
        
        # IV percentile (requires historical data)
        if 'ce_iv' in df.columns:
            df['ce_iv_normalized'] = (df['ce_iv'] - df['ce_iv'].mean()) / df['ce_iv'].std()
            df['pe_iv_normalized'] = (df['pe_iv'] - df['pe_iv'].mean()) / df['pe_iv'].std()
        
        return df
    
    def add_oi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Open Interest features"""
        # Total OI
        df['total_oi'] = df['ce_oi'] + df['pe_oi']
        
        # PCR by OI
        total_call_oi = df['ce_oi'].sum()
        total_put_oi = df['pe_oi'].sum()
        df['pcr_oi'] = total_put_oi / max(total_call_oi, 1)
        
        # OI concentration
        df['ce_oi_pct'] = df['ce_oi'] / max(total_call_oi, 1)
        df['pe_oi_pct'] = df['pe_oi'] / max(total_put_oi, 1)
        
        # OI change ratio
        if 'ce_chg_oi' in df.columns:
            df['ce_oi_change_ratio'] = df['ce_chg_oi'] / df['ce_oi'].replace(0, 1)
            df['pe_oi_change_ratio'] = df['pe_chg_oi'] / df['pe_oi'].replace(0, 1)
        
        return df
    
    def add_greek_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Greek-based features"""
        greek_cols = ['call_delta', 'call_gamma', 'call_theta', 'call_vega',
                      'put_delta', 'put_gamma', 'put_theta', 'put_vega']
        
        for col in greek_cols:
            if col not in df.columns:
                continue
            
            # Net Greeks weighted by OI
            if 'delta' in col:
                oi_col = 'ce_oi' if 'call' in col else 'pe_oi'
                df[f'net_{col}'] = (df[col] * df[oi_col]).sum()
        
        return df
    
    def get_summary_features(self, chain: pd.DataFrame) -> Dict:
        """Get summary F&O features for a symbol"""
        if chain.empty:
            return {}
        
        # Calculate key metrics
        total_ce_oi = chain['ce_oi'].sum()
        total_pe_oi = chain['pe_oi'].sum()
        pcr = total_pe_oi / max(total_ce_oi, 1)
        
        # ATM IV
        underlying = chain['underlying_price'].iloc[0]
        atm_mask = abs(chain['strike'] - underlying) / underlying < 0.02
        atm_ce_iv = chain.loc[atm_mask, 'ce_iv'].mean() if atm_mask.any() else 0
        atm_pe_iv = chain.loc[atm_mask, 'pe_iv'].mean() if atm_mask.any() else 0
        
        # Max OI strikes
        max_ce_strike = chain.loc[chain['ce_oi'].idxmax(), 'strike'] if len(chain) > 0 else 0
        max_pe_strike = chain.loc[chain['pe_oi'].idxmax(), 'strike'] if len(chain) > 0 else 0
        
        return {
            'pcr_oi': round(pcr, 3),
            'total_ce_oi': int(total_ce_oi),
            'total_pe_oi': int(total_pe_oi),
            'atm_ce_iv': round(atm_ce_iv, 2),
            'atm_pe_iv': round(atm_pe_iv, 2),
            'iv_skew': round(atm_pe_iv - atm_ce_iv, 2),
            'max_ce_oi_strike': max_ce_strike,
            'max_pe_oi_strike': max_pe_strike,
            'underlying': underlying
        }
