"""
LLM Fine-tuning - LoRA/PEFT for trading signal explanations
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FineTuneConfig:
    """Fine-tuning configuration"""
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    output_dir: str = "./models/llm/finetuned"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = None
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    warmup_ratio: float = 0.03
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


class LLMFineTuner:
    """Fine-tune LLM for trading signal explanations using LoRA"""
    
    def __init__(self, config: FineTuneConfig = None):
        self.config = config or FineTuneConfig()
        self.model = None
        self.tokenizer = None
        self.trainer = None
    
    def prepare_dataset(self, data: List[Dict]) -> 'Dataset':
        """Prepare dataset for fine-tuning"""
        try:
            from datasets import Dataset
        except ImportError:
            logger.error("datasets library not installed")
            return None
        
        # Format: list of {"prompt": ..., "completion": ...}
        formatted = []
        for item in data:
            text = f"### Input:\n{item['prompt']}\n\n### Response:\n{item['completion']}"
            formatted.append({"text": text})
        
        return Dataset.from_list(formatted)
    
    def create_training_data(self, signals: List[Dict]) -> List[Dict]:
        """Create training pairs from signal data"""
        training_data = []
        
        for signal in signals:
            prompt = self._format_prompt(signal)
            completion = self._format_completion(signal)
            training_data.append({
                "prompt": prompt,
                "completion": completion
            })
        
        return training_data
    
    def _format_prompt(self, signal: Dict) -> str:
        """Format input prompt"""
        return f"""Analyze the following market data and provide a trading signal with explanation:

Symbol: {signal.get('symbol', 'UNKNOWN')}
Timeframe: {signal.get('timeframe', '15m')}
Current Price: {signal.get('price', 0)}
Quant Model Score: {signal.get('quant_score', 0.5)}
RSI: {signal.get('rsi', 50)}
MACD: {signal.get('macd', 0)}
ATR%: {signal.get('atr_percent', 1)}
Volume Ratio: {signal.get('volume_ratio', 1)}
PCR (OI): {signal.get('pcr', 1)}
ATM IV: {signal.get('atm_iv', 15)}

Provide your analysis in JSON format."""
    
    def _format_completion(self, signal: Dict) -> str:
        """Format expected completion"""
        return json.dumps({
            "signal": signal.get('signal', 'HOLD'),
            "confidence": signal.get('confidence', 0.5),
            "position_size": signal.get('position_size', '2% of capital'),
            "stop_loss": signal.get('stop_loss', 'ATR-based'),
            "target": signal.get('target', '2:1 RR'),
            "reasoning": signal.get('reasoning', 'Based on technical and quantitative analysis.')
        }, indent=2)
    
    def setup_model(self):
        """Load base model and setup LoRA"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            import torch
        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            return False
        
        # Quantization config for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Prepare for training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # LoRA config
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        logger.info("Model setup complete")
        
        return True
    
    def train(self, train_data: List[Dict], val_data: List[Dict] = None):
        """Run fine-tuning"""
        try:
            from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
        except ImportError:
            logger.error("transformers library not installed")
            return None
        
        if self.model is None:
            self.setup_model()
        
        # Prepare datasets
        train_dataset = self.prepare_dataset(train_data)
        val_dataset = self.prepare_dataset(val_data) if val_data else None
        
        # Tokenize
        def tokenize(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length"
            )
        
        train_dataset = train_dataset.map(tokenize, remove_columns=["text"])
        if val_dataset:
            val_dataset = val_dataset.map(tokenize, remove_columns=["text"])
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=10,
            save_steps=100,
            fp16=True,
            optim="paged_adamw_8bit"
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator
        )
        
        # Train
        logger.info("Starting fine-tuning...")
        self.trainer.train()
        
        # Save
        self.save()
        
        return self.trainer
    
    def save(self, path: str = None):
        """Save fine-tuned model"""
        path = path or self.config.output_dir
        Path(path).mkdir(parents=True, exist_ok=True)
        
        if self.model:
            self.model.save_pretrained(path)
        if self.tokenizer:
            self.tokenizer.save_pretrained(path)
        
        logger.info(f"Saved model to {path}")
    
    def merge_and_export(self, output_path: str):
        """Merge LoRA weights and export for Ollama"""
        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM
        except ImportError:
            logger.error("Missing dependencies")
            return None
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            device_map="auto",
            torch_dtype="auto"
        )
        
        # Load LoRA
        model = PeftModel.from_pretrained(base_model, self.config.output_dir)
        
        # Merge
        merged = model.merge_and_unload()
        
        # Save merged model
        merged.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        logger.info(f"Merged model saved to {output_path}")
        return output_path


# Example usage
if __name__ == "__main__":
    # Create sample training data
    sample_data = [
        {
            "symbol": "NIFTY 50",
            "timeframe": "15m",
            "price": 22500,
            "quant_score": 0.75,
            "rsi": 65,
            "macd": 50,
            "atr_percent": 0.8,
            "volume_ratio": 1.5,
            "pcr": 0.9,
            "atm_iv": 14,
            "signal": "BUY",
            "confidence": 0.75,
            "reasoning": "Strong momentum with RSI above 60, positive MACD..."
        }
    ]
    
    config = FineTuneConfig(
        base_model="mistralai/Mistral-7B-Instruct-v0.2",
        num_epochs=1
    )
    
    finetuner = LLMFineTuner(config)
    training_data = finetuner.create_training_data(sample_data)
    print(f"Created {len(training_data)} training samples")
