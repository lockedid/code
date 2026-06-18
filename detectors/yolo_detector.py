from typing import List, Dict, Any
import numpy as np
from ultralytics import YOLO
from .base_detector import BaseDetector, DetectionResult

class YOLODetector(BaseDetector):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.class_names = []
        self.target_classes = config.get("target_classes", ["person", "car", "motorcycle"])
    
    def load_model(self):
        model_path = self.config.get("path", "./models/yolov11m.pt")
        self.model = YOLO(model_path)
        self.class_names = self.model.names
    
    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        if self.model is None:
            self.load_model()
        
        results = self.model(frame, conf=self.config.get("confidence", 0.5), 
                           iou=self.config.get("iou_threshold", 0.45))
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                bbox = box.xyxy[0].cpu().numpy().tolist()
                confidence = box.conf[0].cpu().numpy().item()
                class_id = int(box.cls[0].cpu().numpy().item())
                class_name = self.class_names.get(class_id, "unknown")
                
                if self.target_classes and class_name not in self.target_classes:
                    continue
                    
                detections.append(DetectionResult(bbox, confidence, class_id, class_name))
        
        return detections

class PersonDetector(YOLODetector):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.target_classes = ["person"]
    
    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        all_detections = super().detect(frame)
        return [d for d in all_detections if d.class_name in self.target_classes]

class VehicleDetector(YOLODetector):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.target_classes = ["car", "bus", "truck"]
    
    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        all_detections = super().detect(frame)
        return [d for d in all_detections if d.class_name in self.target_classes]

class FireDetector(YOLODetector):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.target_classes = ["fire", "smoke"]
    
    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        all_detections = super().detect(frame)
        return [d for d in all_detections if d.class_name in self.target_classes]

class EBikeDetector(YOLODetector):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.target_classes = ["electric_bike", "electric_scooter"]
    
    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        all_detections = super().detect(frame)
        return [d for d in all_detections if d.class_name in self.target_classes]