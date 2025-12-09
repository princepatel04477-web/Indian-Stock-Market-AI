"""
LLM Inference - Generate trading signal explanations
"""

import json
import requests
from typing import Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class LLMInference:
    """LLM inference for trading signal explanations"""
    
    def __init__(self, 
                 model_path: str = None,
                 ollama_host: str = "http://localhost:11434",
                 ollama_model: str = "trader-llm"):
        self.model_path = model_path
        self.ollama_host = ollama_host
        self.ollama_model = ollama_model
        self.model = None
        self.tokenizer = None
        self.use_ollama = model_path is None
    
    def load_model(self):
        """Load the fine-tuned model"""
        if self.use_ollama:
            # Check Ollama is running
            try:
                response = requests.get(f"{self.ollama_host}/api/tags")
                if response.status_code == 200:
                    logger.info("Connected to Ollama")
                    return True
            except:
                logger.warning("Ollama not available, falling back to local model")
                self.use_ollama = False
        
        if not self.use_ollama and self.model_path:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch
                
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
                logger.info(f"Loaded model from {self.model_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                return False
        
        return False
    
    def generate(self, prompt: str, max_tokens: int = 512,
                 temperature: float = 0.3) -> str:
        """Generate response from prompt"""
        if self.use_ollama:
            return self._generate_ollama(prompt, max_tokens, temperature)
        else:
            return self._generate_local(prompt, max_tokens, temperature)
    
    def _generate_ollama(self, prompt: str, max_tokens: int,
                         temperature: float) -> str:
        """Generate using Ollama API"""
        try:
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                logger.error(f"Ollama error: {response.text}")
                return ""
                
        except Exception as e:
            logger.error(f"Ollama request failed: {e}")
            return ""
    
    def _generate_local(self, prompt: str, max_tokens: int,
                        temperature: float) -> str:
        """Generate using local model"""
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        if self.model is None:
            return ""
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the response part
            if "### Response:" in response:
                response = response.split("### Response:")[-1].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Local generation failed: {e}")
            return ""
    
    def explain_signal(self, context: Dict) -> Dict:
        """Generate explanation for a trading signal"""
        from .prompts import PromptBuilder
        
        builder = PromptBuilder()
        prompt = builder.build_signal_prompt(context)
        
        response = self.generate(prompt)
        
        # Parse JSON response
        try:
            # Try to extract JSON from response
            if "{" in response and "}" in response:
                json_str = response[response.find("{"):response.rfind("}")+1]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # Fallback to structured response
        return {
            "signal": context.get("quant_signal", "HOLD"),
            "confidence": context.get("quant_confidence", 0.5),
            "reasoning": response,
            "raw_response": response
        }
    
    def batch_explain(self, contexts: list) -> list:
        """Generate explanations for multiple signals"""
        results = []
        for ctx in contexts:
            result = self.explain_signal(ctx)
            results.append(result)
        return results


class MockLLMInference:
    """Mock LLM inference for testing without GPU"""
    
    def explain_signal(self, context: Dict) -> Dict:
        """Generate mock explanation"""
        signal = context.get('quant_signal', 'HOLD')
        confidence = context.get('quant_confidence', 0.5)
        
        reasons = {
            'BUY': f"Technical indicators are bullish. RSI at {context.get('rsi', 50)} shows momentum. MACD is positive.",
            'SELL': f"Technical indicators are bearish. RSI at {context.get('rsi', 50)} suggests weakness. Consider reducing exposure.",
            'HOLD': "Mixed signals. Wait for clearer direction before taking position."
        }
        
        return {
            "signal": signal,
            "confidence": confidence,
            "position_size": f"{min(int(confidence * 5), 5)}% of capital",
            "stop_loss": f"{context.get('atr_percent', 1) * 1.5:.1f}% below entry",
            "target": "2:1 Risk-Reward",
            "reasoning": reasons.get(signal, "Analysis inconclusive.")
        }
