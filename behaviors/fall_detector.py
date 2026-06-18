from typing import List, Dict, Any, Optional
from trackers import Track
from detectors import PoseResult
from .base_behavior import BaseBehaviorAnalyzer, BehaviorEvent

class FallDetector(BaseBehaviorAnalyzer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.aspect_ratio_threshold = config.get("fall_aspect_ratio_threshold", 0.7)
        self.min_frames = 5
        self.fall_candidates: Dict[int, List[float]] = {}
    
    def analyze(self, 
                tracks: List[Track], 
                frame: Optional[Any] = None, 
                pose_results: Optional[List[PoseResult]] = None,
                timestamp: float = 0.0) -> List[BehaviorEvent]:
        events = []
        
        if pose_results is None:
            return events
        
        for pose in pose_results:
            aspect_ratio = pose.get_aspect_ratio()
            
            for track in tracks:
                if not self._bbox_overlap(pose.bbox, track.bbox) > 0.5:
                    continue
                
                if track.track_id not in self.fall_candidates:
                    self.fall_candidates[track.track_id] = []
                
                self.fall_candidates[track.track_id].append(aspect_ratio)
                
                if len(self.fall_candidates[track.track_id]) >= self.min_frames:
                    recent_ratios = self.fall_candidates[track.track_id][-self.min_frames:]
                    avg_ratio = sum(recent_ratios) / len(recent_ratios)
                    
                    if avg_ratio < self.aspect_ratio_threshold:
                        event_key = f"fall_{track.track_id}"
                        
                        if event_key not in self.active_events:
                            description = f"检测到人员摔倒，身体纵横比: {avg_ratio:.2f}"
                            event = BehaviorEvent(
                                event_type="fall",
                                track_id=track.track_id,
                                class_name="person",
                                bbox=track.bbox,
                                timestamp=timestamp,
                                description=description,
                                severity="critical"
                            )
                            self.active_events[event_key] = event
                            events.append(event)
                    
                    self.fall_candidates[track.track_id] = self.fall_candidates[track.track_id][-10:]
        
        return events
    
    def _bbox_overlap(self, bbox1: List[float], bbox2: List[float]) -> float:
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        area_inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        
        return area_inter / area1 if area1 > 0 else 0