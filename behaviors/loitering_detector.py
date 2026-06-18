from typing import List, Dict, Any, Optional
from trackers import Track
from detectors import PoseResult
from .base_behavior import BaseBehaviorAnalyzer, BehaviorEvent

class LoiteringDetector(BaseBehaviorAnalyzer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.loitering_threshold = config.get("loitering_threshold", 300)
    
    def analyze(self, 
                tracks: List[Track], 
                frame: Optional[Any] = None, 
                pose_results: Optional[List[PoseResult]] = None,
                timestamp: float = 0.0) -> List[BehaviorEvent]:
        events = []
        
        for track in tracks:
            if track.class_name != "person":
                continue
            
            stationary_time = track.stationary_frames
            
            if stationary_time >= self.loitering_threshold:
                event_key = f"loitering_{track.track_id}"
                
                if event_key not in self.active_events:
                    description = f"人员长时间逗留超过{self.loitering_threshold}秒"
                    event = BehaviorEvent(
                        event_type="loitering",
                        track_id=track.track_id,
                        class_name=track.class_name,
                        bbox=track.bbox,
                        timestamp=timestamp,
                        description=description,
                        severity="low"
                    )
                    self.active_events[event_key] = event
                    events.append(event)
        
        return events