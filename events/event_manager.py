from typing import List, Dict, Any, Optional
import json
import os
from datetime import datetime
from behaviors import BehaviorEvent
from vlm import VLMResult

class EventRecord:
    def __init__(self, 
                 event: BehaviorEvent, 
                 video_path: str = None, 
                 frame_indices: List[int] = None,
                 vlm_description: str = None):
        self.event = event
        self.video_path = video_path
        self.frame_indices = frame_indices
        self.vlm_description = vlm_description
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event.to_dict(),
            "video_path": self.video_path,
            "frame_indices": self.frame_indices,
            "vlm_description": self.vlm_description,
            "timestamp": self.timestamp
        }

class EventManager:
    def __init__(self, output_dir: str = "./outputs"):
        self.output_dir = output_dir
        self.events: List[EventRecord] = []
        self.event_id_counter = 0
        
        os.makedirs(output_dir, exist_ok=True)
    
    def add_event(self, 
                  event: BehaviorEvent, 
                  video_path: str = None, 
                  frame_indices: List[int] = None) -> int:
        record = EventRecord(event, video_path, frame_indices)
        self.events.append(record)
        self.event_id_counter += 1
        return self.event_id_counter
    
    def add_vlm_description(self, event_id: int, description: str):
        if 0 <= event_id - 1 < len(self.events):
            self.events[event_id - 1].vlm_description = description
    
    def save_events(self, filename: str = "events.json"):
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in self.events], f, ensure_ascii=False, indent=2)
    
    def get_events_by_type(self, event_type: str) -> List[EventRecord]:
        return [e for e in self.events if e.event.event_type == event_type]
    
    def get_events_by_severity(self, severity: str) -> List[EventRecord]:
        return [e for e in self.events if e.event.severity == severity]
    
    def get_all_events(self) -> List[EventRecord]:
        return self.events
    
    def clear_events(self):
        self.events = []