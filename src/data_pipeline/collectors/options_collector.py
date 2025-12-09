"""
Options Chain Collector - F&O data with Greeks
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import logging
from scipy.stats import norm

from .nse_collector import NSEDataCollector

logger = logging.getLogger(__name__)


class OptionsChainCollector:
    """Collector for options chain data with IV and Greeks"""
    
    def __init__(self):
        self.nse_collector = NSEDataCollector()
        self.risk_free_rate = 0.065
        
    def fetch_full_chain(self, symbol: str) -> pd.DataFrame:
        """Fetch complete options chain with Greeks"""
        chain = self.nse_collector.fetch_option_chain(symbol)
        if chain.empty:
            return chain
        chain = self._calculate_greeks(chain)
        chain = self._calculate_derived_metrics(chain)
        return chain
    
    def _calculate_greeks(self, chain: pd.DataFrame) -> pd.DataFrame:
        """Calculate option Greeks using Black-Scholes"""
        for opt_type in ['call', 'put']:
            col_prefix = 'ce' if opt_type == 'call' else 'pe'
            deltas, gammas, thetas, vegas = [], [], [], []
            
            for _, row in chain.iterrows():
                S, K = row['underlying_price'], row['strike']
                T = max((pd.Timestamp(row['expiry']) - pd.Timestamp.now()).days / 365, 0.001)
                sigma = row.get(f'{col_prefix}_iv', 0) / 100
                
                if sigma <= 0:
                    deltas.append(np.nan)
                    gammas.append(np.nan)
                    thetas.append(np.nan)
                    vegas.append(np.nan)
                    continue
                
                try:
                    d1 = (np.log(S/K) + (self.risk_free_rate + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
                    d2 = d1 - sigma*np.sqrt(T)
                    
                    if opt_type == 'call':
                        delta = norm.cdf(d1)
                        theta = -(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) - self.risk_free_rate*K*np.exp(-self.risk_free_rate*T)*norm.cdf(d2)
                    else:
                        delta = norm.cdf(d1) - 1
                        theta = -(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) + self.risk_free_rate*K*np.exp(-self.risk_free_rate*T)*norm.cdf(-d2)
                    
                    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
                    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
                    
                    deltas.append(round(delta, 4))
                    gammas.append(round(gamma, 6))
                    thetas.append(round(theta/365, 2))
                    vegas.append(round(vega, 2))
                except:
                    deltas.append(np.nan)
                    gammas.append(np.nan)
                    thetas.append(np.nan)
                    vegas.append(np.nan)
            
            chain[f'{opt_type}_delta'] = deltas
            chain[f'{opt_type}_gamma'] = gammas
            chain[f'{opt_type}_theta'] = thetas
            chain[f'{opt_type}_vega'] = vegas
        
        return chain
    
    def _calculate_derived_metrics(self, chain: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived options metrics"""
        chain['moneyness'] = chain['strike'] / chain['underlying_price']
        chain['dte'] = (pd.to_datetime(chain['expiry']) - datetime.now()).dt.days
        chain['iv_skew'] = chain['pe_iv'] - chain['ce_iv']
        chain['total_oi'] = chain['ce_oi'] + chain['pe_oi']
        return chain
    
    def calculate_max_pain(self, chain: pd.DataFrame) -> float:
        """Calculate max pain strike"""
        strikes = chain['strike'].unique()
        max_pain_value, max_pain_strike = float('inf'), strikes[len(strikes)//2]
        
        for strike in strikes:
            call_loss = chain[chain['strike'] < strike].apply(
                lambda r: (strike - r['strike']) * r['ce_oi'], axis=1).sum()
            put_loss = chain[chain['strike'] > strike].apply(
                lambda r: (r['strike'] - strike) * r['pe_oi'], axis=1).sum()
            
            if call_loss + put_loss < max_pain_value:
                max_pain_value = call_loss + put_loss
                max_pain_strike = strike
        
        return max_pain_strike
    
    def calculate_pcr(self, chain: pd.DataFrame, method: str = 'oi') -> float:
        """Calculate Put-Call Ratio"""
        if method == 'oi':
            return chain['pe_oi'].sum() / max(chain['ce_oi'].sum(), 1)
        return chain['pe_volume'].sum() / max(chain['ce_volume'].sum(), 1)
