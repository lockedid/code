from typing import List, Dict, Any
import numpy as np
from detectors import DetectionResult
from .base_tracker import BaseTracker, Track

try:
    from bytetrack import BYTETracker
    HAS_BYTETRACK = True
except ImportError:
    HAS_BYTETRACK = False

class ByteTrackTracker(BaseTracker):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.bytetrack = None
        self.frame_id = 0
    
    def update(self, detections: List[DetectionResult], frame: Any = None) -> List[Track]:
        if not HAS_BYTETRACK:
            return self._fallback_update(detections)
        
        if self.bytetrack is None:
            self.bytetrack = BYTETracker(
                track_thresh=self.config.get("confidence", 0.5),
                track_buffer=self.config.get("max_age", 30),
                match_thresh=self.config.get("iou_threshold", 0.5)
            )
        
        dets = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            w = x2 - x1
            h = y2 - y1
            dets.append([x1, y1, w, h, det.confidence])
        
        dets = np.array(dets, dtype=np.float32)
        
        online_targets = self.bytetrack.update(dets, frame.shape[:2])
        
        updated_tracks = []
        for t in online_targets:
            track_id = t.track_id
            x1, y1, x2, y2 = t.box
            bbox = [x1, y1, x2, y2]
            
            if track_id in self.tracks:
                self.tracks[track_id].update(bbox, t.score)
            else:
                class_name = self._get_class_name(detections, bbox)
                self.tracks[track_id] = Track(track_id, bbox, t.score, class_name)
            
            updated_tracks.append(self.tracks[track_id])
        
        self.frame_id += 1
        return updated_tracks
    
    def _get_class_name(self, detections: List[DetectionResult], bbox: List[float]) -> str:
        best_match = None
        max_iou = 0
        
        for det in detections:
            iou = self._calculate_iou(bbox, det.bbox)
            if iou > max_iou:
                max_iou = iou
                best_match = det.class_name
        
        return best_match if best_match else "unknown"
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        area_inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        return area_inter / (area1 + area2 - area_inter) if (area1 + area2 - area_inter) > 0 else 0
    
    def _fallback_update(self, detections: List[DetectionResult]) -> List[Track]:
        updated_tracks = []
        
        for det in detections:
            best_track = None
            max_iou = 0
            
            for track in self.tracks.values():
                iou = self._calculate_iou(det.bbox, track.bbox)
                if iou > max_iou and iou > 0.3:
                    max_iou = iou
                    best_track = track
            
            if best_track:
                best_track.update(det.bbox, det.confidence)
                updated_tracks.append(best_track)
            else:
                new_id = max(self.tracks.keys(), default=0) + 1
                self.tracks[new_id] = Track(new_id, det.bbox, det.confidence, det.class_name)
                updated_tracks.append(self.tracks[new_id])
        
        old_tracks = set(self.tracks.keys()) - {t.track_id for t in updated_tracks}
        for track_id in old_tracks:
            if self.tracks[track_id].stationary_frames > self.config.get("max_age", 30):
                del self.tracks[track_id]
        
        return updated_tracks