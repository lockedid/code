from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from trackers import Track
from detectors import PoseResult

class BehaviorEvent:
    def __init__(self, 
                 event_type: str,
                 track_id: int,
                 class_name: str,
                 bbox: List[float],
                 timestamp: float,
                 description: str,
                 severity: str = "medium",
                 zone_name: str = None):
        self.event_type = event_type
        self.track_id = track_id
        self.class_name = class_name
        self.bbox = bbox
        self.timestamp = timestamp
        self.description = description
        self.severity = severity
        self.zone_name = zone_name
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "track_id": self.track_id,
            "class_name": self.class_name,
            "bbox": self.bbox,
            "timestamp": self.timestamp,
            "description": self.description,
            "severity": self.severity,
            "zone_name": self.zone_name
        }

class BaseBehaviorAnalyzer(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.active_events: Dict[int, BehaviorEvent] = {}
    
    @abstractmethod
    def analyze(self, 
                tracks: List[Track], 
                frame: Optional[Any] = None, 
                pose_results: Optional[List[PoseResult]] = None,
                timestamp: float = 0.0) -> List[BehaviorEvent]:
        pass
    
    def point_in_polygon(self, point: List[float], polygon: List[List[float]]) -> bool:
        x, y = point
        inside = False
        n = len(polygon)
        
        for i in range(n):
            j = (i + 1) % n
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
        
        return inside
    
    def reset(self):
        self.active_events = {}