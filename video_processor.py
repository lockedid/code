import cv2
import os
from typing import List, Dict, Any, Optional
from detectors import PersonDetector, VehicleDetector, FireDetector, EBikeDetector, PoseDetector
from trackers import ByteTrackTracker
from behaviors import IntrusionDetector, ParkingDetector, FallDetector, LoiteringDetector
from events import EventManager
from vlm import QwenVLM, get_prompt
from configs import CONFIG

class VideoProcessor:
    def __init__(self, config: Any = CONFIG):
        self.config = config
        self.detectors = {}
        self.tracker = None
        self.behavior_analyzers = {}
        self.event_manager = EventManager(config.video.output_dir)
        self.vlm = None
        
        self._init_detectors()
        self._init_tracker()
        self._init_behavior_analyzers()
    
    def _init_detectors(self):
        self.detectors["person"] = PersonDetector(self.config.detection.person.__dict__)
        self.detectors["vehicle"] = VehicleDetector(self.config.detection.vehicle.__dict__)
        self.detectors["fire"] = FireDetector(self.config.detection.fire.__dict__)
        self.detectors["ebike"] = EBikeDetector(self.config.detection.ebike.__dict__)
        self.detectors["pose"] = PoseDetector(self.config.detection.pose.__dict__)
    
    def _init_tracker(self):
        self.tracker = ByteTrackTracker(self.config.tracker.__dict__)
    
    def _init_behavior_analyzers(self):
        self.behavior_analyzers["intrusion"] = IntrusionDetector(self.config.behavior.__dict__)
        self.behavior_analyzers["parking"] = ParkingDetector(self.config.behavior.__dict__)
        self.behavior_analyzers["fall"] = FallDetector(self.config.behavior.__dict__)
        self.behavior_analyzers["loitering"] = LoiteringDetector(self.config.behavior.__dict__)
    
    def load_models(self):
        for name, detector in self.detectors.items():
            print(f"Loading {name} detector...")
            detector.load_model()
        
        print("Models loaded successfully")
    
    def process_video(self, video_path: str, output_video: bool = False) -> List[Dict[str, Any]]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if output_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_path = os.path.join(self.config.video.output_dir, "output.mp4")
            out = cv2.VideoWriter(output_path, fourcc, fps, 
                                (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        
        frame_count = 0
        all_events = []
        frame_buffer = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_count / fps
            
            detections = []
            for detector_name in ["person", "vehicle", "fire", "ebike"]:
                dets = self.detectors[detector_name].detect(frame)
                detections.extend(dets)
            
            tracks = self.tracker.update(detections, frame)
            
            pose_results = self.detectors["pose"].detect(frame)
            
            events = []
            for analyzer in self.behavior_analyzers.values():
                analyzer_events = analyzer.analyze(tracks, frame, pose_results, timestamp)
                events.extend(analyzer_events)
            
            for event in events:
                event_id = self.event_manager.add_event(
                    event,
                    video_path=video_path,
                    frame_indices=[frame_count]
                )
                
                frame_buffer.append(frame)
                if len(frame_buffer) > 20:
                    frame_buffer = frame_buffer[-20:]
                
                if self.vlm and len(frame_buffer) >= 10:
                    prompt = get_prompt(event.event_type)
                    vlm_result = self.vlm.generate_description(frame_buffer, prompt, timestamp)
                    self.event_manager.add_vlm_description(event_id, vlm_result.description)
                
                all_events.append(event.to_dict())
            
            if output_video:
                for track in tracks:
                    x1, y1, x2, y2 = track.bbox
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"{track.class_name}: {track.track_id}", 
                                (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                out.write(frame)
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        if output_video:
            out.release()
        
        self.event_manager.save_events()
        return all_events
    
    def process_video_file(self, video_path: str) -> Dict[str, Any]:
        self.load_models()
        
        print(f"Processing video: {video_path}")
        events = self.process_video(video_path, output_video=True)
        
        result = {
            "video_path": video_path,
            "total_events": len(events),
            "events": events,
            "output_path": os.path.join(self.config.video.output_dir, "output.mp4")
        }
        
        return result