from typing import List, Dict, Any
import numpy as np
from ultralytics import YOLO
from .base_detector import BaseDetector

class PoseResult:
    def __init__(self, bbox: List[float], confidence: float, keypoints: np.ndarray, keypoint_confidences: np.ndarray):
        self.bbox = bbox
        self.confidence = confidence
        self.keypoints = keypoints
        self.keypoint_confidences = keypoint_confidences
    
    def get_aspect_ratio(self) -> float:
        if len(self.keypoints) < 17:
            return 1.0
        y_coords = self.keypoints[:, 1]
        valid = self.keypoint_confidences > 0.5
        if not np.any(valid):
            return 1.0
        min_y = np.min(y_coords[valid])
        max_y = np.max(y_coords[valid])
        height = max_y - min_y
        
        x_coords = self.keypoints[:, 0]
        min_x = np.min(x_coords[valid])
        max_x = np.max(x_coords[valid])
        width = max_x - min_x
        
        if width == 0:
            return 1.0
        return height / width

class PoseDetector(BaseDetector):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
    
    def load_model(self):
        model_path = self.config.get("path", "./models/yolov11m-pose.pt")
        self.model = YOLO(model_path)
    
    def detect(self, frame: np.ndarray) -> List[PoseResult]:
        if self.model is None:
            self.load_model()
        
        results = self.model(frame, conf=self.config.get("confidence", 0.5))
        
        pose_results = []
        for result in results:
            boxes = result.boxes
            keypoints = result.keypoints
            
            for i, box in enumerate(boxes):
                bbox = box.xyxy[0].cpu().numpy().tolist()
                confidence = box.conf[0].cpu().numpy().item()
                
                kpts = keypoints.data[i].cpu().numpy() if keypoints is not None else np.array([])
                kpt_conf = keypoints.conf[i].cpu().numpy() if keypoints is not None else np.array([])
                
                pose_results.append(PoseResult(bbox, confidence, kpts, kpt_conf))
        
        return pose_results