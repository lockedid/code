import os
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class ModelConfig:
    name: str
    path: str
    confidence: float = 0.5
    iou_threshold: float = 0.45
    input_size: int = 640

@dataclass
class DetectionConfig:
    person: ModelConfig
    vehicle: ModelConfig
    fire: ModelConfig
    ebike: ModelConfig
    pose: ModelConfig

@dataclass
class TrackerConfig:
    type: str = "bytetrack"
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.5

@dataclass
class BehaviorConfig:
    intrusion_rois: List[Dict[str, Any]] = None
    corridor_zones: List[Dict[str, Any]] = None
    fire_lanes: List[Dict[str, Any]] = None
    loitering_threshold: int = 300  
    ebike_parking_threshold: int = 300
    vehicle_parking_threshold: int = 180
    fall_aspect_ratio_threshold: float = 0.7

@dataclass
class VLMConfig:
    model_name: str = "Qwen2.5-VL"
    model_path: str = "./models/qwen2.5-vl-7b"
    quantize: str = "int4"
    max_tokens: int = 1024
    temperature: float = 0.7

@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    dbname: str = "video_analysis"
    user: str = "admin"
    password: str = "password"

@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

@dataclass
class VideoConfig:
    input_dir: str = "./videos"
    output_dir: str = "./outputs"
    frame_rate: int = 30
    max_duration: int = 3600

@dataclass
class SystemConfig:
    detection: DetectionConfig
    tracker: TrackerConfig
    behavior: BehaviorConfig
    vlm: VLMConfig
    database: DatabaseConfig
    api: APIConfig
    video: VideoConfig

def load_config() -> SystemConfig:
    return SystemConfig(
        detection=DetectionConfig(
            person=ModelConfig(
                name="yolo11m-person",
                path="./models/yolo11m.pt",
                confidence=0.45
            ),
            vehicle=ModelConfig(
                name="yolo11m-vehicle",
                path="./models/yolo11m.pt",
                confidence=0.4
            ),
            fire=ModelConfig(
                name="yolo11s-fire",
                path="./models/yolo11s.pt",
                confidence=0.5
            ),
            ebike=ModelConfig(
                name="yolo11s-ebike",
                path="./models/yolo11s.pt",
                confidence=0.45
            ),
            pose=ModelConfig(
                name="yolo11-pose",
                path="./models/yolo11m-pose.pt",
                confidence=0.5
            )
        ),
        tracker=TrackerConfig(),
        behavior=BehaviorConfig(
            intrusion_rois=[
                {"name": "高压区", "polygon": [(100, 100), (300, 100), (300, 300), (100, 300)]},
                {"name": "配电室", "polygon": [(500, 200), (700, 200), (700, 400), (500, 400)]}
            ],
            corridor_zones=[
                {"name": "楼道A", "polygon": [(0, 0), (1920, 0), (1920, 200), (0, 200)]}
            ],
            fire_lanes=[
                {"name": "消防通道", "polygon": [(0, 1080-150), (1920, 1080-150), (1920, 1080), (0, 1080)]}
            ],
            loitering_threshold=300,
            ebike_parking_threshold=300,
            vehicle_parking_threshold=180,
            fall_aspect_ratio_threshold=0.7
        ),
        vlm=VLMConfig(),
        database=DatabaseConfig(),
        api=APIConfig(),
        video=VideoConfig()
    )

CONFIG = load_config()