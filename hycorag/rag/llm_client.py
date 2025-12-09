"""
LLM Client implementations for HyCoRAG.
"""
from abc import ABC, abstractmethod
from typing import Optional
import torch

class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate text from prompt."""
        pass

class DummyLLMClient(LLMClient):
    """Dummy LLM for testing (returns fixed response)."""
    
    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        return "This is a dummy response for testing purposes."

class LocalLLMClient(LLMClient):
    """
    Local LLM client using Hugging Face Transformers.
    Supports models like Qwen2.5-3B-Instruct, Mistral-7B, etc.
    """
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-3B-Instruct",
                 device_map: str = "auto",
                 load_in_8bit: bool = False):
        """
        Initialize local LLM.
        
        Args:
            model_name: HuggingFace model identifier
            device_map: Device mapping strategy ("auto", "cpu", "cuda")
            load_in_8bit: Use 8-bit quantization to reduce memory
        """
        print(f"Loading {model_name}...")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("transformers not installed. Run: pip install transformers")
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with optimizations
        load_kwargs = {
            "device_map": device_map,
            "torch_dtype": torch.float16,
        }
        
        if load_in_8bit:
            load_kwargs["load_in_8bit"] = True
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )
        
        print(f"Model loaded successfully on {self.model.device}")
    
    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """
        Generate response from prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        # Format prompt for instruction-tuned models
        if "Qwen" in self.model_name:
            # Qwen chat format
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            # Generic format
            formatted_prompt = f"### Question:\n{prompt}\n\n### Answer:\n"
        
        inputs = self.tokenizer(
            formatted_prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,  # Greedy decoding for consistency
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and extract only the generated part
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from output
        if formatted_prompt in full_text:
            response = full_text.split(formatted_prompt)[-1].strip()
        else:
            response = full_text
            
        return response
