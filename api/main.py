from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, Query, Body, Depends, Request
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from typing import List, Dict, Any, Tuple
import os
import shutil
import cv2
import asyncio
import numpy as np
from datetime import datetime, timedelta
import hashlib
import json
import re
import threading
from functools import lru_cache
from .event_deduplicator import event_deduplicator
from .fall_detector import fall_tracker
from .async_processor import async_processor
from .cache_manager import frame_cache, detection_cache, vlm_result_cache, compute_frame_hash

app = FastAPI(title="智能视频分析系统 API", version="1.0")

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")
app.mount("/recordings", StaticFiles(directory="recordings"), name="recordings")

video_dir = "./videos"
output_dir = "./outputs"
recording_dir = "./recordings"

os.makedirs(recording_dir, exist_ok=True)

DETECTION_SKIP_FRAMES = 0
MODEL_WARMUP_FRAMES = 3

def open_video_capture(url):
    if url.startswith("rtsp://"):
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;500000|analyzeduration;500000|probesize;500000'
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        if 'OPENCV_FFMPEG_CAPTURE_OPTIONS' in os.environ:
            del os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS']
        cap = cv2.VideoCapture(url)
    return cap

# 流管理器：跟踪每个视频流的状态
class StreamManager:
    def __init__(self):
        self.streams = {}
        self.latest_frames = {}
        self.rtmp_streams = {}
        self.lock = asyncio.Lock()
    
    async def get_latest_frame(self, stream_id="default"):
        async with self.lock:
            return self.latest_frames.get(stream_id)
    
    async def set_latest_frame(self, frame, stream_id="default"):
        async with self.lock:
            self.latest_frames[stream_id] = frame.copy()
    
    def register_stream(self, stream_id, video_path):
        self.streams[stream_id] = {"video_path": video_path, "active": True}
    
    def unregister_stream(self, stream_id):
        if stream_id in self.streams:
            del self.streams[stream_id]
        if stream_id in self.latest_frames:
            del self.latest_frames[stream_id]
    
    def get_stream_count(self):
        return len(self.streams)
    
    def register_rtmp_stream(self, stream_id, rtmp_url, camera_id=None, record=False):
        self.rtmp_streams[stream_id] = {
            "rtmp_url": rtmp_url,
            "camera_id": camera_id,
            "record": record,
            "active": True,
            "started_at": datetime.now().isoformat()
        }
    
    def unregister_rtmp_stream(self, stream_id):
        if stream_id in self.rtmp_streams:
            self.rtmp_streams[stream_id]["active"] = False
            del self.rtmp_streams[stream_id]
        if stream_id in self.latest_frames:
            del self.latest_frames[stream_id]
    
    def get_rtmp_streams(self):
        return dict(self.rtmp_streams)
    
    def get_rtmp_stream(self, stream_id):
        return self.rtmp_streams.get(stream_id)

stream_manager = StreamManager()

# 用于VLM分析的主流帧（取第一个活跃流）
latest_frame = None
latest_frame_lock = asyncio.Lock()

try:
    from ultralytics import YOLO
    DET_MODEL = YOLO("./models/yolo11m.pt")
    POSE_MODEL = YOLO("./models/yolo11m-pose.pt")
    MODEL_LOADED = True
except Exception as e:
    print(f"模型加载失败: {e}")
    DET_MODEL = None
    POSE_MODEL = None
    MODEL_LOADED = False

danger_zones = {
    "zone1": {
        "name": "危险区域1",
        "points": [(100, 100), (300, 100), (300, 300), (100, 300)],
        "color": (0, 0, 255),
        "enabled": True,
        "zone_type": "general"
    },
    "fire_exit": {
        "name": "消防通道",
        "points": [(500, 50), (700, 50), (700, 200), (500, 200)],
        "color": (255, 165, 0),
        "enabled": True,
        "zone_type": "fire_exit"
    },
    "corridor": {
        "name": "楼道区域",
        "points": [(200, 400), (400, 400), (400, 600), (200, 600)],
        "color": (128, 0, 128),
        "enabled": True,
        "zone_type": "corridor"
    }
}

# 改后（使用正确的 COCO class ID）
vehicle_category_mapping = {
    1: "bicycle",
    2: "car",                # 2 = car ✅
    3: "electric_vehicle",   # 3 = motorcycle ✅
    5: "bus",                # 5 = bus ✅
    7: "truck"               # 7 = truck ✅
}

current_recording = {}
detection_stats = {}

def get_event_key_base(event_type: str, camera_id: int, zone_id: str = None) -> str:
    """生成基础事件key，不包含精确坐标"""
    if zone_id:
        return f"{event_type}_{camera_id}_{zone_id}"
    return f"{event_type}_{camera_id}"

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

class AuthHandler:
    def __init__(self):
        from .database import get_user, add_user, init_db
        init_db()
    
    def login(self, username: str, password: str) -> dict:
        from .database import get_user
        user = get_user(username)
        if not user:
            raise HTTPException(status_code=401, detail="用户名或密码错误")
        
        if user["password"] != hash_password(password):
            raise HTTPException(status_code=401, detail="用户名或密码错误")
        
        return {
            "id": user["id"],
            "username": user["username"],
            "role": user["role"]
        }
    
    def register(self, username: str, password: str, role: str = "user") -> dict:
        from .database import add_user, get_user
        if get_user(username):
            raise HTTPException(status_code=400, detail="用户名已存在")
        
        user_id = add_user(username, hash_password(password), role)
        if user_id == -1:
            raise HTTPException(status_code=400, detail="注册失败")
        
        return {"id": user_id, "username": username, "role": role}

auth_handler = AuthHandler()

def point_in_polygon(point: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:
    x, y = point
    n = len(polygon)
    inside = False
    
    for i in range(n):
        j = (i + 1) % n
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        
        if ((yi > y) != (yj > y)):
            x_intersect = (y - yi) * (xj - xi) / (yj - yi) + xi
            if x < x_intersect:
                inside = not inside
    
    return inside

def detect_fall(keypoints, confidences, threshold=0.5) -> bool:
    if keypoints is None or len(keypoints) < 17:
        return False
    
    nose = keypoints[0]
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    left_knee = keypoints[13]
    right_knee = keypoints[14]
    left_ankle = keypoints[15]
    right_ankle = keypoints[16]
    
    if confidences is None or (confidences[0] < threshold or confidences[5] < threshold or confidences[6] < threshold):
        return False
    
    shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2, 
                       (left_shoulder[1] + right_shoulder[1]) / 2)
    hip_center = ((left_hip[0] + right_hip[0]) / 2, 
                  (left_hip[1] + right_hip[1]) / 2)
    
    body_height = abs(shoulder_center[1] - hip_center[1])
    body_width = abs(left_shoulder[0] - right_shoulder[0])
    
    if body_width > body_height * 1.5:
        return True
    
    if confidences[13] > threshold and confidences[14] > threshold:
        knee_distance = abs(left_knee[0] - right_knee[0])
        if knee_distance > body_height:
            return True
    
    if confidences[15] > threshold and confidences[16] > threshold:
        ankle_distance = abs(left_ankle[0] - right_ankle[0])
        if ankle_distance > body_height * 1.2:
            return True
    
    if nose[1] > shoulder_center[1] + 50:
        return True
    
    return False

async def generate_frames(video_path: str, record=False, camera_id=None, stream_id="default"):
    global DET_MODEL, POSE_MODEL, MODEL_LOADED, latest_frame
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = None
    start_time = None
    
    if record:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.mp4"
        filepath = os.path.join(recording_dir, filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
        start_time = datetime.now()
    
    frame_count = 0
    last_detection_time = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            frame_count += 1
            
            should_detect = frame_count <= MODEL_WARMUP_FRAMES or (frame_count % (DETECTION_SKIP_FRAMES + 1) == 0)
            
            det_results = None
            pose_results = None
            
            if MODEL_LOADED and should_detect:
                frame_hash = compute_frame_hash(frame)
                
                det_results = frame_cache.get_detection(frame_hash)
                if det_results is None:
                    # det_results = DET_MODEL(frame, verbose=False)
                    det_results = DET_MODEL(frame, verbose=False, classes=[0, 1, 2, 3])
                    frame_cache.set_detection(frame_hash, det_results)
                
                pose_results = frame_cache.get_pose(frame_hash)
                if pose_results is None:
                    pose_results = POSE_MODEL(frame, verbose=False)
                    frame_cache.set_pose(frame_hash, pose_results)
            
            if MODEL_LOADED and det_results is not None:
                process_detection_results(frame, det_results, pose_results, camera_id)
            
            draw_danger_zones(frame)
            
            if out:
                out.write(frame)
            
            # 更新全局帧（用于VLM分析）
            async with latest_frame_lock:
                latest_frame = frame.copy()
            
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            await asyncio.sleep(max(0, 1.0/fps - 0.01))
    finally:
        cap.release()
        if out:
            out.release()
            from .database import add_recording
            if start_time:
                duration = int((datetime.now() - start_time).total_seconds())
                add_recording(camera_id or 1, filename, start_time, datetime.now(), duration)

async def generate_rtmp_frames(rtmp_url: str, stream_id: str, record=False, camera_id=None):
    global DET_MODEL, POSE_MODEL, MODEL_LOADED, latest_frame
    
    cap = open_video_capture(rtmp_url)
    
    if not cap.isOpened():
        print(f"无法连接RTMP流: {rtmp_url}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 25
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = None
    start_time = None
    filename = None
    
    if record:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_rtmp_{stream_id}_{timestamp}.mp4"
        filepath = os.path.join(recording_dir, filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
        start_time = datetime.now()
    
    latest = {'frame': None, 'lock': threading.Lock()}
    stop_event = threading.Event()
    
    def reader_thread():
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            with latest['lock']:
                latest['frame'] = frame
    
    t = threading.Thread(target=reader_thread, daemon=True)
    t.start()
    
    frame_count = 0
    detect_interval = 3
    
    try:
        while stream_manager.get_rtmp_stream(stream_id) is not None and t.is_alive():
            frame = None
            with latest['lock']:
                if latest['frame'] is not None:
                    frame = latest['frame'].copy()
                    latest['frame'] = None
            
            if frame is None:
                await asyncio.sleep(0.01)
                continue
            
            frame_count += 1
            
            should_detect = frame_count <= MODEL_WARMUP_FRAMES or (frame_count % detect_interval == 0)
            
            det_results = None
            pose_results = None
            
            if MODEL_LOADED and should_detect:
                frame_hash = compute_frame_hash(frame)
                
                det_results = frame_cache.get_detection(frame_hash)
                if det_results is None:
                    det_results = DET_MODEL(frame, verbose=False, classes=[0, 1, 2, 3])
                    frame_cache.set_detection(frame_hash, det_results)
                
                pose_results = frame_cache.get_pose(frame_hash)
                if pose_results is None:
                    pose_results = POSE_MODEL(frame, verbose=False)
                    frame_cache.set_pose(frame_hash, pose_results)
            
            if MODEL_LOADED and det_results is not None:
                process_detection_results(frame, det_results, pose_results, camera_id)
            
            draw_danger_zones(frame)
            
            if out:
                out.write(frame)
            
            async with latest_frame_lock:
                latest_frame = frame.copy()
            
            await stream_manager.set_latest_frame(frame, stream_id)
            
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            await asyncio.sleep(0.001)
    finally:
        stop_event.set()
        t.join(timeout=3)
        try:
            cap.release()
        except Exception:
            pass
        if out:
            try:
                out.release()
            except Exception:
                pass
            from .database import add_recording
            if start_time and filename:
                duration = int((datetime.now() - start_time).total_seconds())
                add_recording(camera_id or 1, filename, start_time, datetime.now(), duration)
        stream_manager.unregister_rtmp_stream(stream_id)
        print(f"RTMP流已断开: {rtmp_url}, 共处理 {frame_count} 帧")

zone_cache = None
zone_cache_timestamp = 0

def get_cached_danger_zones():
    global zone_cache, zone_cache_timestamp
    from .config import config_manager
    
    current_time = datetime.now().timestamp()
    if zone_cache is None or (current_time - zone_cache_timestamp) > 5:
        zone_cache = config_manager.get_danger_zones()
        zone_cache_timestamp = current_time
    
    return zone_cache

def invalidate_zone_cache():
    global zone_cache, zone_cache_timestamp
    zone_cache = None
    zone_cache_timestamp = 0

def draw_danger_zones(frame):
    danger_zones_config = get_cached_danger_zones()
    for zone_id, zone in danger_zones_config.items():
        if zone.get("enabled", True) and len(zone.get("points", [])) >= 3:
            pts = np.array(zone["points"], np.int32)
            pts = pts.reshape((-1, 1, 2))
            color = zone.get("color", [0, 0, 255])
            color_bgr = (color[2], color[1], color[0])
            cv2.polylines(frame, [pts], isClosed=True, color=color_bgr, thickness=2)

def process_detection_results(frame, det_results, pose_results, camera_id):
    from .database import add_event
    from .alarm import alarm_manager
    from .config import config_manager
    
    alarm_rules = config_manager.get_alarm_rules()
    
    if pose_results is not None:
        fall_rule = alarm_rules.get("fall", {})
        fall_enabled = fall_rule.get("enabled", True)
        fall_threshold = fall_rule.get("threshold", 0.8)
        
        for result in pose_results:
            if result.keypoints is not None and len(result.keypoints) > 0:
                if result.boxes is None:
                    continue
                    
                for idx in range(len(result.keypoints)):
                    keypoints = result.keypoints.xy[idx].cpu().numpy().reshape(-1, 2)
                    confidences = result.keypoints.conf[idx].cpu().numpy() if result.keypoints.conf is not None else None
                    
                    try:
                        bbox = result.boxes[idx].xyxy[0].cpu().numpy()
                    except (IndexError, AttributeError):
                        continue
                    
                    if bbox is not None:
                        x1, y1, x2, y2 = map(int, bbox)
                        bbox_tuple = (x1, y1, x2, y2)
                        
                        track_id = idx + 1000
                        if hasattr(result.boxes[idx], 'id') and result.boxes[idx].id is not None:
                            try:
                                track_id = int(result.boxes[idx].id[0].cpu().numpy())
                            except (IndexError, TypeError, AttributeError):
                                track_id = idx + 1000
                        
                        try:
                            should_alarm, fall_score, factors = fall_tracker.update_person(
                                track_id, keypoints, confidences, bbox_tuple
                            )
                        except Exception as e:
                            print(f"Fall detection error: {e}")
                            continue
                        
                        if should_alarm and fall_enabled:
                            try:
                                confidence = result.boxes[idx].conf[0].cpu().numpy().item() if result.boxes[idx].conf is not None else fall_score
                            except (AttributeError, IndexError):
                                confidence = fall_score
                                
                            add_event("fall", f"检测到人员摔倒 (分数: {fall_score:.2f})", camera_id)
                            alarm_manager.add_alarm("fall", f"检测到人员摔倒 (分数: {fall_score:.2f})", camera_id)
    
    try:
        fall_tracker.cleanup()
    except Exception as e:
        print(f"Fall tracker cleanup error: {e}")
    
    danger_zones_config = get_cached_danger_zones()
    
    for result in det_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            confidence = box.conf[0].cpu().numpy().item() if box.conf is not None else 0.5
            
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            alert_text = None
            alert_color = (0, 255, 0)
            event_type = None
            zone_id = None
            
            for zid, zone in danger_zones_config.items():
                if zone.get("enabled", True) and point_in_polygon((center_x, center_y), zone.get("points", [])):
                    zone_id = zid
                    zone_type = zone.get("zone_type", "general")
                    
                    if cls in vehicle_category_mapping:
                        vehicle_type = vehicle_category_mapping[cls]
                        
                        if zone_type == "corridor" and vehicle_type == "electric_vehicle":
                            alert_text = "⚠ 楼道停放电动车"
                            alert_color = (0, 0, 255)
                            event_type = "corridor_parking"
                        elif zone_type == "fire_exit":
                            alert_text = "⚠ 消防通道停车"
                            alert_color = (0, 0, 255)
                            event_type = "fire_exit_parking"
                        elif zone_type == "general":
                            alert_text = "⚠ 闯入危险区域"
                            alert_color = (0, 0, 255)
                            event_type = "danger_zone"
                    # else:
                    #     alert_text = "⚠ 闯入危险区域"
                    #     alert_color = (0, 0, 255)
                    #     event_type = "danger_zone"
                    else:
                        if zone_type == "corridor":
                            alert_text = "⚠ 楼道区域异常"
                            alert_color = (0, 0, 255)
                            event_type = "corridor"
                        elif zone_type == "fire_exit":
                            alert_text = "⚠ 消防通道占用"
                            alert_color = (0, 0, 255)
                            event_type = "fire_exit"
                        elif zone_type == "entrance":
                            alert_text = "⚠ 出入口异常"
                            alert_color = (0, 0, 255)
                            event_type = "entrance"
                        else:
                            alert_text = "⚠ 闯入危险区域"
                            alert_color = (0, 0, 255)
                            event_type = "danger_zone"
                    break
                    break
            
            if event_type:
                rule = alarm_rules.get(event_type, {})
                rule_enabled = rule.get("enabled", True)
                rule_threshold = rule.get("threshold", 0.5)
                
                if rule_enabled and confidence >= rule_threshold:
                    if event_deduplicator.record_event(event_type, camera_id, zone_id, (x1, y1, x2, y2), confidence):
                        add_event(event_type, alert_text, camera_id)
                        alarm_manager.add_alarm(event_type, alert_text, camera_id)
                elif not rule_enabled:
                    alert_text = None
                    alert_color = (0, 255, 0)
            
            draw_detection_box(frame, x1, y1, x2, y2, cls, alert_text, alert_color)

def draw_detection_box(frame, x1, y1, x2, y2, cls, alert_text, alert_color):
    if cls in vehicle_category_mapping:
        if vehicle_category_mapping[cls] == "electric_vehicle":
            color = (255, 165, 0)
        else:
            color = (0, 191, 255)
        
        if alert_text:
            cv2.rectangle(frame, (x1, y1), (x2, y2), alert_color, 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    else:
        if alert_text:
            cv2.rectangle(frame, (x1, y1), (x2, y2), alert_color, 1)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

@app.get("/")
async def root():
    return {"message": "智能视频分析系统 API"}

@app.get("/view")
async def view():
    return FileResponse("./frontend/index.html")

@app.get("/playback")
async def playback():
    return FileResponse("./frontend/playback.html")

@app.get("/dashboard")
async def dashboard():
    return FileResponse("./frontend/dashboard.html")

@app.post("/api/login")
async def login(data: dict = Body(...)):
    username = data.get("username")
    password = data.get("password")
    
    if not username or not password:
        raise HTTPException(status_code=400, detail="用户名和密码不能为空")
    
    return auth_handler.login(username, password)

@app.post("/api/register")
async def register(data: dict = Body(...)):
    username = data.get("username")
    password = data.get("password")
    role = data.get("role", "user")
    
    if not username or not password:
        raise HTTPException(status_code=400, detail="用户名和密码不能为空")
    
    return auth_handler.register(username, password, role)

@app.get("/api/recordings")
async def list_recordings(
    camera_id: int = None,
    start_time: str = None,
    end_time: str = None
):
    from .database import get_recordings
    recordings = get_recordings(camera_id, start_time, end_time)
    
    valid_recordings = []
    for recording in recordings:
        filepath = os.path.join(recording_dir, recording["filename"])
        if os.path.exists(filepath):
            valid_recordings.append(recording)
    
    return {"recordings": valid_recordings}

@app.get("/api/recordings/{filename}")
async def get_recording(filename: str, request: Request):
    filepath = os.path.join(recording_dir, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="录制文件不存在")
    
    file_size = os.path.getsize(filepath)
    range_header = request.headers.get('range')
    
    if range_header is not None:
        range_header = range_header.strip()
        range_match = re.match(r'bytes=(\d+)-(\d+)?', range_header)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2)) if range_match.group(2) else file_size - 1
            length = end - start + 1
            
            with open(filepath, 'rb') as f:
                f.seek(start)
                content = f.read(length)
            
            headers = {
                'Content-Range': f'bytes {start}-{end}/{file_size}',
                'Content-Length': str(length),
                'Content-Type': 'video/mp4',
                'Accept-Ranges': 'bytes'
            }
            return Response(content, status_code=206, headers=headers)
    
    return FileResponse(filepath, media_type="video/mp4")

@app.get("/api/stats")
async def get_stats(date: str = None):
    from .database import get_daily_stats
    if date:
        return {"stats": get_daily_stats(date)}
    
    today = datetime.now().date().isoformat()
    return {"stats": get_daily_stats(today)}

@app.get("/api/stats/week")
async def get_weekly_stats():
    from .database import get_daily_stats
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=7)
    
    all_stats = get_daily_stats()
    
    weekly_stats = []
    for stat in all_stats:
        stat_date = stat["date"]
        if start_date.isoformat() <= stat_date <= end_date.isoformat():
            weekly_stats.append(stat)
    
    return {"weekly_stats": weekly_stats}

@app.get("/api/cameras")
async def list_cameras():
    from .database import get_cameras
    return {"cameras": get_cameras()}

@app.post("/api/cameras")
async def add_camera(data: dict = Body(...)):
    name = data.get("name")
    url = data.get("url")
    
    if not name:
        raise HTTPException(status_code=400, detail="摄像头名称不能为空")
    
    from .database import add_camera
    camera_id = add_camera(name, url)
    return {"id": camera_id, "name": name, "url": url}

@app.get("/api/cameras/{camera_id}/stream")
async def stream_camera(camera_id: int):
    from .database import get_cameras
    cameras = get_cameras()
    camera = next((c for c in cameras if c["id"] == camera_id), None)
    
    if not camera:
        raise HTTPException(status_code=404, detail="摄像头不存在")
    
    video_path = os.path.join(video_dir, "test.mp4")
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="视频文件不存在")
    
    return StreamingResponse(generate_frames(video_path, camera_id=camera_id), media_type='multipart/x-mixed-replace; boundary=frame')

@app.post("/api/recording/start")
async def start_recording(camera_id: int = Query(1)):
    global current_recording
    
    if camera_id in current_recording:
        raise HTTPException(status_code=400, detail="该摄像头正在录制中")
    
    video_source = None
    is_rtsp = False
    
    active_streams = stream_manager.get_rtmp_streams()
    for stream_id, info in active_streams.items():
        if info.get("camera_id") == camera_id:
            video_source = info["rtmp_url"]
            is_rtsp = video_source.startswith("rtsp://")
            break
    
    if video_source is None:
        video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
        if video_files:
            video_source = os.path.join(video_dir, video_files[0])
    
    if video_source is None:
        raise HTTPException(status_code=400, detail="没有可录制的视频源")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recording_{timestamp}.mp4"
    filepath = os.path.join(recording_dir, filename)
    
    cap = open_video_capture(video_source) if is_rtsp else cv2.VideoCapture(video_source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
    
    current_recording[camera_id] = {
        "status": "recording",
        "start_time": datetime.now(),
        "cap": cap,
        "out": out,
        "filename": filename,
        "filepath": filepath,
        "fps": fps,
        "source": video_source,
        "is_rtsp": is_rtsp
    }
    
    threading.Thread(target=recording_thread, args=(camera_id,), daemon=True).start()
    
    return {"message": "开始录制", "camera_id": camera_id, "filename": filename, "source": video_source, "start_time": datetime.now().isoformat()}

def recording_thread(camera_id: int):
    import traceback
    import time
    
    print(f"Recording thread started for camera {camera_id}")
    
    frame_count = 0
    prev_frame_hash = None
    is_rtsp = current_recording.get(camera_id, {}).get("is_rtsp", False)
    
    while camera_id in current_recording:
        try:
            if camera_id not in current_recording:
                break
            
            recording_info = current_recording[camera_id]
            if recording_info["status"] != "recording":
                break
            
            cap = recording_info["cap"]
            out = recording_info["out"]
            
            if cap is None or out is None:
                break
            
            ret, frame = cap.read()
            if not ret:
                if is_rtsp:
                    print(f"RTSP stream ended for camera {camera_id}")
                    break
                print("End of video, looping")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                prev_frame_hash = None
                time.sleep(0.1)
                continue
            
            current_hash = hash(frame.tobytes())
            if current_hash == prev_frame_hash:
                time.sleep(0.04)
                continue
            prev_frame_hash = current_hash
            
            out.write(frame)
            frame_count += 1
            
            fps = recording_info.get("fps", 25)
            time.sleep(1/fps)
            
        except Exception as e:
            print(f"Recording error: {e}")
            print(traceback.format_exc())
            break
    
    if camera_id in current_recording:
        current_recording[camera_id]["status"] = "done"
    
    print(f"Recording thread finished for camera {camera_id}, frames recorded: {frame_count}")

@app.post("/api/recording/stop")
async def stop_recording(camera_id: int = Query(1)):
    global current_recording
    
    if camera_id not in current_recording:
        raise HTTPException(status_code=400, detail="该摄像头未在录制")
    
    recording_info = current_recording[camera_id]
    
    try:
        recording_info["status"] = "stopping"
        
        import time
        for _ in range(50):
            if camera_id not in current_recording:
                break
            time.sleep(0.1)
        
        if camera_id in current_recording:
            try:
                recording_info["cap"].release()
            except Exception:
                pass
            try:
                recording_info["out"].release()
            except Exception:
                pass
        
        duration = (datetime.now() - recording_info["start_time"]).total_seconds()
        
        filename = recording_info["filename"]
        filepath = recording_info["filepath"]
        
        compatible_filepath = filepath.replace('.mp4', '_web.mp4')
        
        try:
            import subprocess
            result = subprocess.run([
                'ffmpeg', '-i', filepath,
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-c:a', 'aac', '-b:a', '128k',
                '-y', compatible_filepath
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                os.remove(filepath)
                os.rename(compatible_filepath, filepath)
                print(f"Video transcoded successfully to H.264")
            else:
                print(f"FFmpeg transcoding failed: {result.stderr}")
        except FileNotFoundError:
            print("FFmpeg not found, keeping original video format")
        
        from .database import add_recording
        add_recording(camera_id, filename, recording_info["start_time"], datetime.now(), int(duration))
        
    finally:
        if camera_id in current_recording:
            del current_recording[camera_id]
    
    return {"message": "停止录制", "camera_id": camera_id, "duration": f"{duration:.2f}秒"}

@app.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...)):
    file_path = os.path.join(video_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {"message": f"视频已上传: {file.filename}", "file_path": file_path}

@app.get("/videos")
async def list_videos():
    videos = []
    for f in os.listdir(video_dir):
        if f.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            videos.append(f)
    return {"videos": videos}

@app.post("/api/vlm/load")
async def load_vlm_model():
    try:
        from .vlm import vlm_processor
        vlm_processor.load_model()
        return {"message": "Qwen2.5-VL模型加载成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型加载失败: {str(e)}")

@app.get("/api/vlm/status")
async def get_vlm_status():
    from .vlm import vlm_processor
    return {"loaded": vlm_processor.loaded, "device": vlm_processor.device}

@app.post("/api/vlm/describe")
async def describe_video(file: UploadFile = File(...)):
    try:
        from .vlm import vlm_processor
        
        temp_path = os.path.join(video_dir, f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_hash = hashlib.md5(open(temp_path, 'rb').read()).hexdigest()
        cached_result = vlm_result_cache.get("describe", file_hash)
        
        if cached_result is not None:
            os.remove(temp_path)
            return {"description": cached_result, "cached": True}
        
        result = vlm_processor.analyze_video_segment(temp_path, frame_interval=15)
        
        vlm_result_cache.set("describe", result.get("description", ""), file_hash)
        
        os.remove(temp_path)
        return {"description": result.get("description", ""), "cached": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"视频描述失败: {str(e)}")

@app.get("/api/vlm/summarize")
async def summarize_video(filename: str):
    try:
        from .vlm import vlm_processor
        
        video_path = os.path.join(recording_dir, filename)
        if not os.path.exists(video_path):
            video_path = os.path.join(video_dir, filename)
        
        if not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail="视频文件不存在")
        
        file_hash = hashlib.md5(open(video_path, 'rb').read()).hexdigest()
        cached_result = vlm_result_cache.get("summarize", file_hash)
        
        if cached_result is not None:
            return {"summary": cached_result, "cached": True}
        
        result = vlm_processor.summarize_video(video_path, max_frames=8)
        
        vlm_result_cache.set("summarize", result.get("summary", ""), file_hash)
        
        return {"summary": result.get("summary", ""), "cached": False}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"视频总结失败: {str(e)}")

@app.post("/api/vlm/generate-report")
async def generate_event_report(event_type: str = Query(...), description: str = None):
    try:
        from .vlm import vlm_processor
        
        context = {
            "event_type": event_type,
            "description": description,
            "timestamp": datetime.now().isoformat()
        }
        
        result = vlm_processor.generate_event_report(event_type, frame=None, context=context)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"报告生成失败: {str(e)}")

latest_frame = None
latest_frame_lock = asyncio.Lock()

@app.post("/api/vlm/analyze-frame")
async def analyze_frame(data: dict = Body(...)):
    try:
        from .vlm import vlm_processor
        
        global latest_frame
        
        async with latest_frame_lock:
            if latest_frame is None:
                raise HTTPException(status_code=400, detail="暂无视频帧数据")
            
            frame = latest_frame.copy()
        
        prompt = data.get("prompt", "请描述当前画面中的场景、物体和人物活动")
        
        loop = asyncio.get_event_loop()
        description = await loop.run_in_executor(vlm_processor.executor, vlm_processor.describe_frame, frame, prompt)
        
        return {"description": description, "timestamp": datetime.now().isoformat()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"帧分析失败: {str(e)}")

@app.post("/api/vlm/chat")
async def chat_with_vlm(data: dict = Body(...)):
    try:
        from .vlm import vlm_processor
        
        message = data.get("message", "")
        use_image = data.get("use_image", False)
        
        if not message.strip():
            raise HTTPException(status_code=400, detail="消息不能为空")
        
        frame = None
        if use_image:
            global latest_frame
            async with latest_frame_lock:
                if latest_frame is None:
                    raise HTTPException(status_code=400, detail="暂无视频帧数据")
                frame = latest_frame.copy()
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(vlm_processor.executor, vlm_processor.chat, message, frame)
        
        return {"response": response, "timestamp": datetime.now().isoformat()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"对话失败: {str(e)}")

@app.post("/api/vlm/clear-history")
async def clear_chat_history():
    try:
        from .vlm import vlm_processor
        result = vlm_processor.clear_history()
        return {"message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清空历史失败: {str(e)}")

@app.get("/danger-zones")
async def get_danger_zones():
    return {"danger_zones": danger_zones}

@app.post("/danger-zones")
async def add_danger_zone(data: dict = Body(...)):
    zone_id = data.get("zone_id")
    name = data.get("name")
    points = data.get("points", [])
    color = data.get("color", [0, 0, 255])
    enabled = data.get("enabled", True)
    zone_type = data.get("zone_type", "general")
    
    if not zone_id:
        raise HTTPException(status_code=400, detail="区域ID不能为空")
    
    if not name:
        raise HTTPException(status_code=400, detail="区域名称不能为空")
    
    if zone_id in danger_zones:
        raise HTTPException(status_code=400, detail="危险区域ID已存在")
    
    if len(points) < 3:
        raise HTTPException(status_code=400, detail="危险区域至少需要3个点")
    
    danger_zones[zone_id] = {
        "name": name,
        "points": points,
        "color": color,
        "enabled": enabled,
        "zone_type": zone_type
    }
    
    return {"message": f"危险区域 {name} 已添加"}

@app.put("/danger-zones/{zone_id}")
async def update_danger_zone(zone_id: str, data: dict = Body(...)):
    if zone_id not in danger_zones:
        raise HTTPException(status_code=404, detail="危险区域不存在")
    
    if "name" in data:
        danger_zones[zone_id]["name"] = data["name"]
    if "points" in data:
        if len(data["points"]) < 3:
            raise HTTPException(status_code=400, detail="危险区域至少需要3个点")
        danger_zones[zone_id]["points"] = data["points"]
    if "color" in data:
        danger_zones[zone_id]["color"] = data["color"]
    if "enabled" in data:
        danger_zones[zone_id]["enabled"] = data["enabled"]
    if "zone_type" in data:
        danger_zones[zone_id]["zone_type"] = data["zone_type"]
    
    return {"message": f"危险区域 {zone_id} 已更新"}

@app.delete("/danger-zones/{zone_id}")
async def delete_danger_zone(zone_id: str):
    if zone_id not in danger_zones:
        raise HTTPException(status_code=404, detail="危险区域不存在")
    
    del danger_zones[zone_id]
    return {"message": f"危险区域 {zone_id} 已删除"}

active_push_streams = {}
push_stream_lock = asyncio.Lock()

@app.post("/api/push/frame")
async def push_frame(
    frame_data: UploadFile = File(...),
    stream_id: str = Query("default", description="流ID"),
    camera_id: int = Query(1, description="摄像头ID"),
    timestamp: str = Query(None, description="帧时间戳")
):
    global latest_frame, DET_MODEL, POSE_MODEL, MODEL_LOADED
    
    contents = await frame_data.read()
    frame = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    
    if frame is None:
        raise HTTPException(status_code=400, detail="无法解码图像数据")
    
    async with push_stream_lock:
        if stream_id not in active_push_streams:
            active_push_streams[stream_id] = {
                "camera_id": camera_id,
                "started_at": datetime.now().isoformat(),
                "frame_count": 0,
                "last_frame_time": datetime.now().isoformat()
            }
        active_push_streams[stream_id]["frame_count"] += 1
        active_push_streams[stream_id]["last_frame_time"] = datetime.now().isoformat()
    
    if MODEL_LOADED:
        frame_hash = compute_frame_hash(frame)
        det_results = frame_cache.get_detection(frame_hash)
        if det_results is None:
            det_results = DET_MODEL(frame, verbose=False, classes=[0, 1, 2, 3])
            frame_cache.set_detection(frame_hash, det_results)
        pose_results = frame_cache.get_pose(frame_hash)
        if pose_results is None:
            pose_results = POSE_MODEL(frame, verbose=False)
            frame_cache.set_pose(frame_hash, pose_results)
        process_detection_results(frame, det_results, pose_results, camera_id)
    
    draw_danger_zones(frame)
    
    async with latest_frame_lock:
        latest_frame = frame.copy()
    
    await stream_manager.set_latest_frame(frame, stream_id)
    
    return {"message": "帧接收成功", "stream_id": stream_id, "frame_count": active_push_streams[stream_id]["frame_count"]}

@app.get("/api/push/streams")
async def list_push_streams():
    async with push_stream_lock:
        streams = []
        for sid, info in active_push_streams.items():
            streams.append({
                "stream_id": sid,
                "camera_id": info["camera_id"],
                "started_at": info["started_at"],
                "frame_count": info["frame_count"],
                "last_frame_time": info["last_frame_time"]
            })
    return {"streams": streams, "count": len(streams)}

@app.get("/api/push/stream/{stream_id}/view")
async def view_push_stream(stream_id: str):
    async with push_stream_lock:
        if stream_id not in active_push_streams:
            raise HTTPException(status_code=404, detail="推流不存在")
    
    async def stream_pushed_frames():
        while True:
            frame = await stream_manager.get_latest_frame(stream_id)
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            await asyncio.sleep(0.033)
    
    return StreamingResponse(stream_pushed_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.post("/api/push/stream/{stream_id}/stop")
async def stop_push_stream(stream_id: str):
    async with push_stream_lock:
        if stream_id not in active_push_streams:
            raise HTTPException(status_code=404, detail="推流不存在")
        del active_push_streams[stream_id]
    stream_manager.unregister_stream(stream_id)
    return {"message": f"推流已停止: {stream_id}"}

@app.post("/api/rtmp/connect")
async def connect_rtmp_stream(data: dict = Body(...)):
    rtmp_url = data.get("url", "")
    camera_id = data.get("camera_id", 1)
    record = data.get("record", False)
    stream_name = data.get("name", "")
    
    if not rtmp_url:
        raise HTTPException(status_code=400, detail="RTMP URL不能为空")
    
    if not rtmp_url.startswith(("rtmp://", "rtmps://", "rtsp://")):
        raise HTTPException(status_code=400, detail="URL必须以rtmp://、rtmps://或rtsp://开头")
    
    stream_id = f"rtmp_{camera_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    cap_test = open_video_capture(rtmp_url)
    if not cap_test.isOpened():
        raise HTTPException(status_code=400, detail=f"无法连接到流: {rtmp_url}")
    
    fps = cap_test.get(cv2.CAP_PROP_FPS)
    width = int(cap_test.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_test.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_test.release()
    
    stream_manager.register_rtmp_stream(stream_id, rtmp_url, camera_id, record)
    
    return {
        "message": "RTMP流连接成功",
        "stream_id": stream_id,
        "url": rtmp_url,
        "camera_id": camera_id,
        "record": record,
        "info": {
            "fps": fps if fps > 0 else 25,
            "width": width,
            "height": height
        }
    }

@app.get("/api/rtmp/streams")
async def list_rtmp_streams():
    streams = stream_manager.get_rtmp_streams()
    result = []
    for stream_id, info in streams.items():
        result.append({
            "stream_id": stream_id,
            "url": info["rtmp_url"],
            "camera_id": info.get("camera_id"),
            "record": info.get("record", False),
            "active": info.get("active", True),
            "started_at": info.get("started_at")
        })
    return {"streams": result, "count": len(result)}

@app.get("/api/rtmp/stream/{stream_id}")
async def get_rtmp_stream_info(stream_id: str):
    info = stream_manager.get_rtmp_stream(stream_id)
    if not info:
        raise HTTPException(status_code=404, detail="RTMP流不存在")
    return {
        "stream_id": stream_id,
        "url": info["rtmp_url"],
        "camera_id": info.get("camera_id"),
        "record": info.get("record", False),
        "active": info.get("active", True),
        "started_at": info.get("started_at")
    }

@app.get("/api/rtmp/stream/{stream_id}/view")
async def view_rtmp_stream(stream_id: str):
    info = stream_manager.get_rtmp_stream(stream_id)
    if not info:
        raise HTTPException(status_code=404, detail="RTMP流不存在")
    
    return StreamingResponse(
        generate_rtmp_frames(
            info["rtmp_url"],
            stream_id,
            record=info.get("record", False),
            camera_id=info.get("camera_id")
        ),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

@app.post("/api/rtmp/stream/{stream_id}/disconnect")
async def disconnect_rtmp_stream(stream_id: str):
    info = stream_manager.get_rtmp_stream(stream_id)
    if not info:
        raise HTTPException(status_code=404, detail="RTMP流不存在")
    
    stream_manager.unregister_rtmp_stream(stream_id)
    return {"message": f"RTMP流已断开: {stream_id}"}

@app.get("/api/rtmp/stream/{stream_id}/snapshot")
async def get_rtmp_snapshot(stream_id: str):
    frame = await stream_manager.get_latest_frame(stream_id)
    if frame is None:
        raise HTTPException(status_code=404, detail="暂无可用帧，请确认流正在推送中")
    
    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ret:
        raise HTTPException(status_code=500, detail="图像编码失败")
    
    return Response(content=buffer.tobytes(), media_type="image/jpeg")

@app.get("/stream/rtmp")
async def stream_rtmp_direct(
    url: str = Query(..., description="RTMP流地址"),
    camera_id: int = Query(1, description="摄像头ID"),
    record: bool = Query(False, description="是否录制")
):
    if not url.startswith(("rtmp://", "rtmps://", "rtsp://")):
        raise HTTPException(status_code=400, detail="URL必须以rtmp://、rtmps://或rtsp://开头")
    
    cap_test = open_video_capture(url)
    if not cap_test.isOpened():
        raise HTTPException(status_code=400, detail=f"无法连接到流: {url}")
    cap_test.release()
    
    stream_id = f"rtmp_{camera_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    stream_manager.register_rtmp_stream(stream_id, url, camera_id, record)
    
    return StreamingResponse(
        generate_rtmp_frames(url, stream_id, record=record, camera_id=camera_id),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

@app.get("/stream")
async def stream_video(
    filename: str = Query("test.mp4", description="视频文件名"),
    record: bool = Query(False, description="是否录制"),
    camera_id: int = Query(1, description="摄像头ID"),
    skip_frames: int = Query(0, description="检测跳帧数，0表示每帧都检测"),
    t: str = Query(None, description="时间戳，用于缓存刷新")
):
    global DETECTION_SKIP_FRAMES
    
    # 根据并发流数量动态调整检测频率
    stream_count = stream_manager.get_stream_count()
    if stream_count > 3:
        # 当有多个流时，降低检测频率
        effective_skip = max(skip_frames, stream_count - 1)
    else:
        effective_skip = skip_frames
    
    # 临时设置全局跳帧参数
    original_skip = DETECTION_SKIP_FRAMES
    DETECTION_SKIP_FRAMES = effective_skip
    
    video_path = os.path.join(video_dir, filename)
    if not os.path.exists(video_path):
        DETECTION_SKIP_FRAMES = original_skip
        raise HTTPException(status_code=404, detail=f"视频文件不存在: {filename}")
    
    # 生成唯一的流ID
    stream_id = f"{filename}_{camera_id}_{datetime.now().timestamp()}"
    
    try:
        return StreamingResponse(
            generate_frames(video_path, record=record, camera_id=camera_id, stream_id=stream_id), 
            media_type='multipart/x-mixed-replace; boundary=frame'
        )
    finally:
        # 恢复原始设置
        DETECTION_SKIP_FRAMES = original_skip

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket, filename: str = "test.mp4"):
    await websocket.accept()
    
    video_path = os.path.join(video_dir, filename)
    if not os.path.exists(video_path):
        await websocket.send_text("错误: 视频文件不存在")
        await websocket.close()
        return
    
    from ultralytics import YOLO
    det_model = YOLO("./models/yolo11m.pt")
    pose_model = YOLO("./models/yolo11m-pose.pt")
    
    cap = cv2.VideoCapture(video_path)
    connected = True
    
    try:
        while connected:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # det_results = det_model(frame, verbose=False)
            det_results = det_model(frame, verbose=False, classes=[0, 1, 3, 4])
            pose_results = pose_model(frame, verbose=False)
            
            for zone_id, zone in danger_zones.items():
                if zone["enabled"] and len(zone["points"]) >= 3:
                    pts = np.array(zone["points"], np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    color_bgr = (zone["color"][2], zone["color"][1], zone["color"][0])
                    cv2.polylines(frame, [pts], isClosed=True, color=color_bgr, thickness=2)
            
            for result in pose_results:
                if result.keypoints is not None and len(result.keypoints) > 0:
                    for idx in range(len(result.keypoints)):
                        keypoints = result.keypoints.xy[idx].cpu().numpy().reshape(-1, 2)
                        confidences = result.keypoints.conf[idx].cpu().numpy() if result.keypoints.conf is not None else None
                        
                        is_fall = detect_fall(keypoints, confidences)
                        
                        if is_fall:
                            bbox = result.boxes[idx].xyxy[0].cpu().numpy() if result.boxes is not None else None
                            if bbox is not None:
                                x1, y1, x2, y2 = map(int, bbox)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            for result in det_results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    alert_text = None
                    alert_color = (0, 255, 0)
                    
                    for zone_id, zone in danger_zones.items():
                        if zone["enabled"] and point_in_polygon((center_x, center_y), zone["points"]):
                            alert_color = (0, 0, 255)
                            break
                    
                    if cls in vehicle_category_mapping:
                        if vehicle_category_mapping[cls] == "electric_vehicle":
                            color = (255, 165, 0)
                        else:
                            color = (0, 191, 255)
                        
                        if alert_text:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), alert_color, 2)
                        else:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                    else:
                        if alert_text:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), alert_color, 1)
                        else:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            
            try:
                await websocket.send_bytes(buffer.tobytes())
            except Exception as send_error:
                connected = False
                break
            
            await asyncio.sleep(0.033)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        pass
    finally:
        cap.release()
        await websocket.close()

@app.get("/api/alarm/active")
async def get_active_alarms():
    from .alarm import alarm_manager
    return {"alarms": alarm_manager.get_active_alarms()}

@app.get("/api/alarm/history")
async def get_alarm_history(limit: int = Query(100, description="返回记录数")):
    from .alarm import alarm_manager
    return {"alarms": alarm_manager.get_alarm_history(limit)}

@app.post("/api/alarm/acknowledge/{alarm_id}")
async def acknowledge_alarm(alarm_id: int):
    from .alarm import alarm_manager
    success = alarm_manager.acknowledge_alarm(alarm_id)
    if success:
        return {"message": "告警已确认"}
    else:
        raise HTTPException(status_code=404, detail="告警不存在")

@app.get("/api/events")
async def get_events_api(
    start_time: str = Query(None, description="开始时间"),
    end_time: str = Query(None, description="结束时间"),
    event_type: str = Query(None, description="事件类型")
):
    from .database import get_events
    events = get_events(start_time, end_time, event_type)
    return {"events": events}

@app.get("/api/events/types")
async def get_event_types():
    event_types = [
        {"type": "fall", "name": "摔倒检测", "description": "检测到人员摔倒"},
        {"type": "danger_zone", "name": "危险区域入侵", "description": "检测到人员或车辆进入危险区域"},
        {"type": "fire_exit", "name": "消防通道占用", "description": "检测到消防通道被占用"},
        {"type": "corridor", "name": "楼道异常", "description": "检测到楼道区域有异常"},
        {"type": "corridor_parking", "name": "楼道停车", "description": "检测到楼道停放电动车"},
        {"type": "fire_exit_parking", "name": "消防通道停车", "description": "检测到消防通道停车"},
        {"type": "crossing", "name": "越界检测", "description": "检测到人员越界"},
        {"type": "loitering", "name": "徘徊检测", "description": "检测到可疑徘徊行为"},
        {"type": "abandoned", "name": "遗留物检测", "description": "检测到疑似遗留物"},
        {"type": "intrusion", "name": "入侵检测", "description": "检测到入侵行为"}
    ]
    return {"event_types": event_types}

@app.get("/api/config/danger-zones")
async def get_config_danger_zones():
    from .config import config_manager
    return {"danger_zones": config_manager.get_danger_zones()}

@app.post("/api/config/danger-zones")
async def add_config_danger_zone(data: dict = Body(...)):
    from .config import config_manager
    zone_id = data.get("zone_id", f"zone_{len(config_manager.get_danger_zones()) + 1}")
    config_manager.add_danger_zone(zone_id, data)
    invalidate_zone_cache()
    return {"message": "危险区域添加成功", "zone_id": zone_id}

@app.put("/api/config/danger-zones/{zone_id}")
async def update_config_danger_zone(zone_id: str, data: dict = Body(...)):
    from .config import config_manager
    success = config_manager.update_danger_zone(zone_id, data)
    if success:
        invalidate_zone_cache()
        return {"message": "危险区域更新成功"}
    else:
        raise HTTPException(status_code=404, detail="危险区域不存在")

@app.delete("/api/config/danger-zones/{zone_id}")
async def delete_config_danger_zone(zone_id: str):
    from .config import config_manager
    success = config_manager.delete_danger_zone(zone_id)
    if success:
        invalidate_zone_cache()
        return {"message": "危险区域删除成功"}
    else:
        raise HTTPException(status_code=404, detail="危险区域不存在")

@app.get("/api/config/alarm-rules")
async def get_config_alarm_rules():
    from .config import config_manager
    return {"alarm_rules": config_manager.get_alarm_rules()}

@app.put("/api/config/alarm-rules/{rule_id}")
async def update_config_alarm_rule(rule_id: str, data: dict = Body(...)):
    from .config import config_manager
    from .alarm import alarm_manager
    success = config_manager.update_alarm_rule(rule_id, data)
    if success:
        if not data.get("enabled", True):
            alarm_manager.active_alarms = [
                a for a in alarm_manager.active_alarms
                if a["type"] != rule_id
            ]
        return {"message": "告警规则更新成功"}
    else:
        raise HTTPException(status_code=404, detail="告警规则不存在")

@app.get("/api/config/alarm-levels")
async def get_config_alarm_levels():
    from .config import config_manager
    return {"alarm_levels": config_manager.get_alarm_levels()}

@app.get("/api/config/notify-providers")
async def get_config_notify_providers():
    from .config import config_manager
    return {"notify_providers": config_manager.get_notify_providers()}

@app.put("/api/config/notify-providers/{provider_id}")
async def update_config_notify_provider(provider_id: str, data: dict = Body(...)):
    from .config import config_manager
    enabled = data.get("enabled", False)
    success = config_manager.update_notify_provider(provider_id, enabled)
    if success:
        return {"message": "通知提供者更新成功"}
    else:
        raise HTTPException(status_code=404, detail="通知提供者不存在")

@app.post("/api/config/reload")
async def reload_config():
    from .config import config_manager
    config_manager.reload()
    return {"message": "配置已重新加载"}

@app.post("/api/config/reorder-zones")
async def reorder_danger_zones(data: dict = Body(...)):
    from .config import config_manager
    zones = data.get("zones", {})
    config_manager.set_danger_zones(zones)
    return {"message": "危险区域顺序已更新"}

@app.post("/api/reasoning/reason")
async def reasoning_analysis(data: dict = Body(...)):
    """AI推理分析 - 根据事件类型进行推理，返回风险评估、推理链和处置建议"""
    event_type = data.get("event_type", "unknown")
    event_id = data.get("event_id", None)
    description = data.get("description", "")
    confidence = data.get("confidence", 0.85)
    
    risk_rules = {
        "fall": {"level": "CRITICAL", "score": 95, "reasoning": ["检测到人员摔倒", "可能导致严重伤害", "需要立即响应", "评估风险等级"], "recommendations": ["立即派遣安保人员前往现场", "联系急救中心", "持续监控现场情况", "通知相关负责人"]},
        "danger_zone": {"level": "HIGH", "score": 75, "reasoning": ["检测到人员进入危险区域", "违反安全规定", "存在安全隐患", "评估风险等级"], "recommendations": ["通知安保人员前往查看", "通过广播提醒离开", "持续监控该区域", "记录事件以备后续分析"]},
        "fire_exit": {"level": "CRITICAL", "score": 90, "reasoning": ["消防通道被占用", "违反消防法规", "紧急情况下影响逃生", "评估风险等级"], "recommendations": ["立即通知安保人员处理", "联系消防部门", "清理通道确保畅通", "对责任人进行警告"]},
        "fire_exit_parking": {"level": "CRITICAL", "score": 90, "reasoning": ["消防通道停车", "严重违反消防规定", "紧急情况阻碍救援", "评估风险等级"], "recommendations": ["立即通知拖车", "联系车主移车", "通知消防部门", "记录并上报管理部门"]},
        "corridor": {"level": "MEDIUM", "score": 55, "reasoning": ["楼道区域有异常活动", "可能影响通行", "存在潜在安全问题", "评估风险等级"], "recommendations": ["派遣巡逻人员查看", "通过监控持续观察", "必要时进行干预", "记录事件"]},
        "corridor_parking": {"level": "HIGH", "score": 70, "reasoning": ["楼道停放电动车", "存在火灾隐患", "违反安全规定", "评估风险等级"], "recommendations": ["通知车主移车", "进行安全提醒", "加强该区域巡逻", "考虑加装监控"]},
        "crossing": {"level": "HIGH", "score": 80, "reasoning": ["检测到人员越界", "可能存在安全风险", "需要确认意图", "评估风险等级"], "recommendations": ["通知安保人员核实", "通过广播询问意图", "记录越界行为", "必要时采取强制措施"]},
        "loitering": {"level": "MEDIUM", "score": 60, "reasoning": ["检测到可疑徘徊行为", "可能存在安全隐患", "需要进一步观察", "评估风险等级"], "recommendations": ["加强该区域监控", "派遣安保人员询问", "记录可疑行为", "通知上级"]},
        "abandoned": {"level": "MEDIUM", "score": 55, "reasoning": ["检测到疑似遗留物", "可能存在安全威胁", "需要排查", "评估风险等级"], "recommendations": ["通知排爆人员", "设置警戒区域", "疏散周边人员", "记录并上报"]},
        "intrusion": {"level": "CRITICAL", "score": 95, "reasoning": ["检测到入侵行为", "严重安全威胁", "需要立即响应", "评估风险等级"], "recommendations": ["立即启动应急预案", "通知安保人员拦截", "联系警方", "封锁相关区域"]}
    }
    
    rule = risk_rules.get(event_type, {"level": "MEDIUM", "score": 50, "reasoning": ["未知事件类型", "进行基本风险评估", "生成处置建议"], "recommendations": ["进一步核实事件详情", "通知相关负责人", "记录事件", "持续监控"]})
    
    report = f"""## AI分析报告

### 事件信息
- 事件ID: {event_id or 'N/A'}
- 事件类型: {event_type}
- 描述: {description or '暂无描述'}
- 置信度: {(confidence * 100):.1f}%
- 分析时间: {datetime.now().isoformat()}

### 风险评估
- 风险等级: {rule['level']}
- 风险评分: {rule['score']}/100

### 推理分析
{chr(10).join([f"{i+1}. {step}" for i, step in enumerate(rule['reasoning'])])}

### 处置建议
{chr(10).join([f"{i+1}. {rec}" for i, rec in enumerate(rule['recommendations'])])}

---
*本报告由AI系统自动生成*
"""
    
    return {
        "event_id": event_id,
        "event_type": event_type,
        "description": description,
        "risk_level": rule["level"],
        "risk_score": rule["score"],
        "confidence": confidence,
        "reasoning_chain": rule["reasoning"],
        "recommendation": rule["recommendations"],
        "report": report,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/reasoning/current")
async def get_current_reasoning():
    """获取当前决策数据（演示数据）"""
    return {
        "event_type": "fall",
        "timestamp": datetime.now().isoformat(),
        "location": "摄像头1 - 大厅区域",
        "track_id": None,
        "risk_level": "CRITICAL",
        "risk_score": 95,
        "reasoning_chain": ["检测到人员摔倒", "可能导致严重伤害", "需要立即响应", "评估风险等级为CRITICAL"],
        "recommendation": ["立即派遣安保人员前往现场", "联系急救中心", "持续监控现场情况", "通知相关负责人"],
        "report": """## AI分析报告

### 事件信息
- 事件类型: 摔倒检测
- 位置: 摄像头1 - 大厅区域
- 分析时间: {datetime.now().isoformat()}

### 风险评估
- 风险等级: CRITICAL
- 风险评分: 95/100

### 分析结论
检测到人员摔倒事件，属于高优先级紧急事件，需要立即响应。

### 处置建议
1. 立即派遣安保人员前往现场
2. 联系急救中心
3. 持续监控现场情况
4. 通知相关负责人"""
    }

@app.post("/api/reasoning/demo/simulate")
async def simulate_scenario(data: dict = Body(...)):
    """模拟场景推理（演示用）"""
    scenario = data.get("scenario", "fall")
    
    scenarios = {
        "fall": {
            "event_type": "fall",
            "risk_level": "CRITICAL",
            "risk_score": 95,
            "reasoning_chain": ["检测到人员摔倒", "可能导致严重伤害", "需要立即响应", "评估风险等级为CRITICAL"],
            "recommendation": ["立即派遣安保人员前往现场", "联系急救中心", "持续监控现场情况", "通知相关负责人"],
            "report": "模拟摔倒事件分析完成"
        },
        "intrusion": {
            "event_type": "intrusion",
            "risk_level": "CRITICAL",
            "risk_score": 95,
            "reasoning_chain": ["检测到入侵行为", "严重安全威胁", "需要立即响应", "评估风险等级为CRITICAL"],
            "recommendation": ["立即启动应急预案", "通知安保人员拦截", "联系警方", "封锁相关区域"],
            "report": "模拟入侵事件分析完成"
        },
        "fire": {
            "event_type": "fire",
            "risk_level": "CRITICAL",
            "risk_score": 98,
            "reasoning_chain": ["检测到火灾/烟雾", "严重安全威胁", "需要立即响应", "评估风险等级为CRITICAL"],
            "recommendation": ["立即启动火警系统", "通知消防部门", "组织人员疏散", "封锁相关区域"],
            "report": "模拟火灾事件分析完成"
        },
        "parking": {
            "event_type": "corridor_parking",
            "risk_level": "HIGH",
            "risk_score": 70,
            "reasoning_chain": ["检测到楼道停放电动车", "存在火灾隐患", "违反安全规定", "评估风险等级为HIGH"],
            "recommendation": ["通知车主移车", "进行安全提醒", "加强该区域巡逻", "考虑加装监控"],
            "report": "模拟楼道停车事件分析完成"
        },
        "loitering": {
            "event_type": "loitering",
            "risk_level": "MEDIUM",
            "risk_score": 60,
            "reasoning_chain": ["检测到可疑徘徊行为", "可能存在安全隐患", "需要进一步观察", "评估风险等级为MEDIUM"],
            "recommendation": ["加强该区域监控", "派遣安保人员询问", "记录可疑行为", "通知上级"],
            "report": "模拟徘徊事件分析完成"
        }
    }
    
    result = scenarios.get(scenario, scenarios["fall"])
    result["timestamp"] = datetime.now().isoformat()
    
    return {"success": True, "data": result}

@app.get("/api/fall/status")
async def get_fall_detection_status():
    """获取摔倒检测的当前状态信息"""
    from .fall_detector import fall_tracker
    
    active_falls = fall_tracker.get_active_falls()
    tracked_people = len(fall_tracker.people)
    
    # 获取每个人的详细状态
    people_details = []
    for track_id, person in fall_tracker.people.items():
        people_details.append({
            "track_id": track_id,
            "state": person.state.value,
            "recent_score": person.fall_scores[-1] if person.fall_scores else 0,
            "avg_score": sum(person.fall_scores) / len(person.fall_scores) if person.fall_scores else 0,
            "has_alarmed": person.has_alarmed,
            "last_alarm_time": person.last_alarm_time.isoformat() if person.last_alarm_time else None,
            "suspect_start_time": person.suspect_start_time.isoformat() if person.suspect_start_time else None,
            "confirmed_time": person.confirmed_time.isoformat() if person.confirmed_time else None
        })
    
    return {
        "tracked_people_count": tracked_people,
        "active_falls": active_falls,
        "people_details": people_details,
        "thresholds": {
            "fall_score_threshold": fall_tracker.fall_score_threshold,
            "pose_conf_threshold": fall_tracker.pose_conf_threshold
        }
    }

async def check_alarm_timeout():
    """定时检查超时告警的后台任务"""
    from .alarm import alarm_manager
    
    while True:
        alarm_manager.check_timeout_alarms()
        await asyncio.sleep(1)

async def cleanup_deduplicator_cache():
    """定时清理去重器缓存的后台任务"""
    while True:
        event_deduplicator.cleanup()
        await asyncio.sleep(60)

async def cleanup_frame_cache():
    """定时清理帧缓存的后台任务"""
    while True:
        frame_cache.cleanup()
        detection_cache.cleanup()
        vlm_result_cache.cleanup()
        await asyncio.sleep(120)

@app.get("/api/deduplication/stats")
async def get_deduplication_stats():
    return {"stats": event_deduplicator.get_stats()}

@app.post("/api/deduplication/reset")
async def reset_deduplication(event_type: str = None, camera_id: int = None):
    event_deduplicator.reset(event_type, camera_id)
    return {"message": "去重状态已重置", "event_type": event_type, "camera_id": camera_id}

@app.post("/api/deduplication/cooldown/{event_type}")
async def set_deduplication_cooldown(event_type: str, cooldown: float = Body(..., embed=True)):
    event_deduplicator.set_cooldown_override(event_type, cooldown)
    return {"message": f"已设置 {event_type} 去重间隔为 {cooldown} 秒"}

@app.get("/api/cache/stats")
async def get_cache_stats():
    return {
        "frame_cache": frame_cache.get_stats(),
        "detection_cache": detection_cache.get_stats(),
        "vlm_cache": vlm_result_cache.get_stats()
    }

@app.post("/api/cache/clear")
async def clear_cache():
    """清空所有缓存"""
    frame_cache.clear()
    detection_cache.clear()
    vlm_result_cache.clear()
    return {"message": "缓存已清空"}

@app.on_event("startup")
async def startup_event():
    """应用启动时启动后台任务"""
    asyncio.create_task(check_alarm_timeout())
    asyncio.create_task(cleanup_deduplicator_cache())
    asyncio.create_task(cleanup_frame_cache())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)