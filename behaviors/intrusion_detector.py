from typing import List, Dict, Any, Optional
from trackers import Track
from detectors import PoseResult
from .base_behavior import BaseBehaviorAnalyzer, BehaviorEvent

class IntrusionDetector(BaseBehaviorAnalyzer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.rois = config.get("intrusion_rois", [])
    
    def analyze(self, 
                tracks: List[Track], 
                frame: Optional[Any] = None, 
                pose_results: Optional[List[PoseResult]] = None,
                timestamp: float = 0.0) -> List[BehaviorEvent]:
        events = []
        
        for track in tracks:
            if track.class_name != "person":
                continue
            
            center = track.get_center()
            
            for roi in self.rois:
                polygon = roi.get("polygon", [])
                zone_name = roi.get("name", "unknown")
                
                if self.point_in_polygon(center, polygon):
                    event_key = f"{track.track_id}_{zone_name}"
                    
                    if event_key not in self.active_events:
                        description = f"人员闯入危险区域: {zone_name}"
                        event = BehaviorEvent(
                            event_type="intrusion",
                            track_id=track.track_id,
                            class_name=track.class_name,
                            bbox=track.bbox,
                            timestamp=timestamp,
                            description=description,
                            severity="high",
                            zone_name=zone_name
                        )
                        self.active_events[event_key] = event
                        events.append(event)
        
        return events