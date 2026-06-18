from abc import ABC, abstractmethod
from typing import List, Dict, Any
from detectors import DetectionResult

class Track:
    def __init__(self, track_id: int, bbox: List[float], confidence: float, class_name: str):
        self.track_id = track_id
        self.bbox = bbox
        self.confidence = confidence
        self.class_name = class_name
        self.history = [bbox]
        self.stationary_frames = 0
    
    def update(self, bbox: List[float], confidence: float):
        self.bbox = bbox
        self.confidence = confidence
        self.history.append(bbox)
        if len(self.history) > 1:
            prev_bbox = self.history[-2]
            dx = abs(bbox[0] - prev_bbox[0]) + abs(bbox[2] - prev_bbox[2])
            dy = abs(bbox[1] - prev_bbox[1]) + abs(bbox[3] - prev_bbox[3])
            self.stationary_frames = self.stationary_frames + 1 if (dx + dy) < 10 else 0
    
    def get_center(self) -> List[float]:
        return [(self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2]
    
    def get_stationary_time(self, fps: int) -> float:
        return self.stationary_frames / fps

class BaseTracker(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tracks: Dict[int, Track] = {}
    
    @abstractmethod
    def update(self, detections: List[DetectionResult], frame: Any = None) -> List[Track]:
        pass
    
    def get_active_tracks(self) -> List[Track]:
        return list(self.tracks.values())
    
    def reset(self):
        self.tracks = {}