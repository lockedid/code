from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np

class VLMResult:
    def __init__(self, description: str, confidence: float, tokens: int):
        self.description = description
        self.confidence = confidence
        self.tokens = tokens
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "confidence": self.confidence,
            "tokens": self.tokens
        }

class BaseVLM(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.processor = None
    
    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def generate_description(self, 
                           frames: List[np.ndarray], 
                           prompt: str = None,
                           timestamp: float = 0.0) -> VLMResult:
        pass