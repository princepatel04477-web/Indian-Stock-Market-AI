"""
Prompt Builder - Create prompts for LLM inference
"""

from typing import Dict, Optional
import json


class PromptBuilder:
    """Build prompts for trading signal generation"""
    
    SYSTEM_PROMPT = """You are an expert Indian stock market analyst and trader. 
Your role is to analyze market data and provide actionable trading signals with clear explanations.
Always respond in valid JSON format with the following structure:
{
    "signal": "BUY" | "SELL" | "HOLD",
    "confidence": 0.0-1.0,
    "position_size": "X% of capital",
    "stop_loss": "price or % description",
    "target": "price or RR ratio",
    "reasoning": "detailed explanation"
}

Consider these factors in your analysis:
1. Technical indicators (RSI, MACD, moving averages)
2. Volatility (ATR)
3. Options data (PCR, IV)
4. Volume patterns
5. Risk management
6. Current market regime

Always prioritize risk management. Never recommend more than 5% of capital per trade."""
    
    def build_signal_prompt(self, context: Dict) -> str:
        """Build prompt for signal generation"""
        prompt = f"""### System:
{self.SYSTEM_PROMPT}

### Input:
Analyze the following market data and provide a trading recommendation:

**Symbol:** {context.get('symbol', 'UNKNOWN')}
**Timeframe:** {context.get('timeframe', '15m')}
**Strategy:** {context.get('strategy', 'intraday')}

**Price Data:**
- Current Price: ₹{context.get('price', 0):,.2f}
- Day Change: {context.get('change', 0):.2f}%
- High: ₹{context.get('high', 0):,.2f}
- Low: ₹{context.get('low', 0):,.2f}

**Quant Model Output:**
- Signal: {context.get('quant_signal', 'HOLD')}
- Confidence: {context.get('quant_confidence', 0.5):.1%}
- Buy Probability: {context.get('prob_buy', 0.33):.1%}
- Sell Probability: {context.get('prob_sell', 0.33):.1%}

**Technical Indicators:**
- RSI (14): {context.get('rsi', 50):.1f}
- MACD: {context.get('macd', 0):.2f}
- MACD Histogram: {context.get('macd_histogram', 0):.2f}
- ATR%: {context.get('atr_percent', 1):.2f}%
- Price vs EMA21: {context.get('close_vs_ema_21', 0):.2f}%
- BB Position: {context.get('bb_position', 0.5):.2f}
- ADX: {context.get('adx', 25):.1f}

**Volume Analysis:**
- Volume Ratio (vs 20-SMA): {context.get('volume_ratio', 1):.2f}x

**Options Data (if F&O):**
- PCR (OI): {context.get('pcr', 1):.2f}
- ATM CE IV: {context.get('atm_ce_iv', 15):.1f}%
- ATM PE IV: {context.get('atm_pe_iv', 15):.1f}%
- IV Skew: {context.get('iv_skew', 0):.1f}
- Max Pain: ₹{context.get('max_pain', 0):,.0f}
- Days to Expiry: {context.get('dte', 7)}

**Market Context:**
- India VIX: {context.get('vix', 15):.1f}
- Market Regime: {context.get('market_regime', 'normal')}
- Is Expiry Day: {context.get('is_expiry', False)}

### Response:
Provide your analysis and recommendation in JSON format:"""
        
        return prompt
    
    def build_explanation_prompt(self, signal: Dict, context: Dict) -> str:
        """Build prompt for explaining an existing signal"""
        prompt = f"""### System:
{self.SYSTEM_PROMPT}

### Input:
Explain the following trading signal in detail:

**Signal Generated:**
- Direction: {signal.get('signal', 'HOLD')}
- Confidence: {signal.get('confidence', 0.5):.1%}

**Market Data:**
Symbol: {context.get('symbol', 'UNKNOWN')}
Price: ₹{context.get('price', 0):,.2f}
RSI: {context.get('rsi', 50):.1f}
MACD: {context.get('macd', 0):.2f}
ATR%: {context.get('atr_percent', 1):.2f}%

Explain why this signal was generated and provide risk management guidelines.

### Response:"""
        
        return prompt
    
    def build_portfolio_summary_prompt(self, positions: list) -> str:
        """Build prompt for portfolio summary"""
        positions_text = "\n".join([
            f"- {p.get('symbol')}: {p.get('quantity')} @ ₹{p.get('avg_price'):,.2f} (P&L: {p.get('pnl_percent', 0):.1f}%)"
            for p in positions
        ])
        
        prompt = f"""### System:
You are a portfolio analyst. Summarize the portfolio status and provide recommendations.

### Input:
Current Positions:
{positions_text}

Provide a brief portfolio summary with:
1. Overall exposure assessment
2. Top performing and underperforming positions
3. Risk concentration warnings
4. Suggested actions

### Response:"""
        
        return prompt
    
    def parse_response(self, response: str) -> Dict:
        """Parse LLM response into structured format"""
        try:
            # Try to find JSON in response
            if "{" in response and "}" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # Return default structure
        return {
            "signal": "HOLD",
            "confidence": 0.5,
            "reasoning": response,
            "parse_error": True
        }
