from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np

class DetectionResult:
    def __init__(self, bbox: List[float], confidence: float, class_id: int, class_name: str):
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": self.bbox,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name
        }

class BaseDetector(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
    
    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        pass
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        return frame
    
    def postprocess(self, outputs: Any) -> List[DetectionResult]:
        return outputs