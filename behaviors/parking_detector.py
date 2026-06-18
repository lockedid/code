from typing import List, Dict, Any, Optional
from trackers import Track
from detectors import PoseResult
from .base_behavior import BaseBehaviorAnalyzer, BehaviorEvent

class ParkingDetector(BaseBehaviorAnalyzer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.corridor_zones = config.get("corridor_zones", [])
        self.fire_lanes = config.get("fire_lanes", [])
        self.ebike_threshold = config.get("ebike_parking_threshold", 300)
        self.vehicle_threshold = config.get("vehicle_parking_threshold", 180)
    
    def analyze(self, 
                tracks: List[Track], 
                frame: Optional[Any] = None, 
                pose_results: Optional[List[PoseResult]] = None,
                timestamp: float = 0.0) -> List[BehaviorEvent]:
        events = []
        
        for track in tracks:
            center = track.get_center()
            
            if track.class_name in ["electric_bike", "electric_scooter"]:
                for zone in self.corridor_zones:
                    polygon = zone.get("polygon", [])
                    zone_name = zone.get("name", "楼道")
                    
                    if self.point_in_polygon(center, polygon):
                        stationary_time = track.stationary_frames
                        
                        if stationary_time >= self.ebike_threshold:
                            event_key = f"ebike_{track.track_id}_{zone_name}"
                            
                            if event_key not in self.active_events:
                                description = f"电动车在{zone_name}停放超过{self.ebike_threshold}秒"
                                event = BehaviorEvent(
                                    event_type="ebike_parking",
                                    track_id=track.track_id,
                                    class_name=track.class_name,
                                    bbox=track.bbox,
                                    timestamp=timestamp,
                                    description=description,
                                    severity="medium",
                                    zone_name=zone_name
                                )
                                self.active_events[event_key] = event
                                events.append(event)
            
            elif track.class_name in ["car", "bus", "truck"]:
                for lane in self.fire_lanes:
                    polygon = lane.get("polygon", [])
                    zone_name = lane.get("name", "消防通道")
                    
                    if self.point_in_polygon(center, polygon):
                        stationary_time = track.stationary_frames
                        
                        if stationary_time >= self.vehicle_threshold:
                            event_key = f"vehicle_{track.track_id}_{zone_name}"
                            
                            if event_key not in self.active_events:
                                description = f"车辆占用{zone_name}超过{self.vehicle_threshold}秒"
                                event = BehaviorEvent(
                                    event_type="fire_lane_occupation",
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