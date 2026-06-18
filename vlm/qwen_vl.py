from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
from .base_vlm import BaseVLM, VLMResult

try:
    from transformers import AutoTokenizer, AutoModelForVisionAndLanguageGeneration
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

class QwenVLM(BaseVLM):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get("model_name", "Qwen/Qwen2.5-VL-7B-Instruct")
        self.model_path = config.get("model_path", "./models/qwen2.5-vl-7b")
        self.quantize = config.get("quantize", "int4")
        self.max_tokens = config.get("max_tokens", 1024)
        self.temperature = config.get("temperature", 0.7)
    
    def load_model(self):
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library not installed")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        if self.quantize == "int4":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            self.model = AutoModelForVisionAndLanguageGeneration.from_pretrained(
                self.model_path,
                quantization_config=quantization_config,
                device_map="auto"
            )
        else:
            self.model = AutoModelForVisionAndLanguageGeneration.from_pretrained(
                self.model_path,
                device_map="auto"
            )
    
    def generate_description(self, 
                           frames: List[np.ndarray], 
                           prompt: str = None,
                           timestamp: float = 0.0) -> VLMResult:
        if self.model is None:
            self.load_model()
        
        if prompt is None:
            prompt = "描述视频中发生的事件，包括人物行为、物体状态和异常情况。"
        
        images = [Image.fromarray(frame) for frame in frames]
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img} for img in images
                ] + [{"type": "text", "text": prompt}]
            }
        ]
        
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with self.tokenizer.as_target_tokenizer():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True
            )
        
        description = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        description = description.replace(text, "").strip()
        
        return VLMResult(
            description=description,
            confidence=0.9,
            tokens=len(outputs[0])
        )