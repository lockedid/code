import cv2
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import json
import os

try:
    from .sms import send_alarm_sms
    SMS_AVAILABLE = True
except ImportError:
    SMS_AVAILABLE = False
    
ON_CALL_PHONES = ["13800138000"]

class AlarmManager:
    def __init__(self):
        self.active_alarms = []
        self.alarm_history = []
        self.object_tracks = {}
        self.loitering_threshold = 30
        self.max_track_age = 60
        self.notified_alarms = set()
        
    def add_alarm(self, event_type: str, message: str, camera_id: int = None, frame_path: str = None):
        from .config import config_manager
        
        rule = config_manager.get_alarm_rule(event_type)
        if rule and not rule.get("enabled", True):
            return None
        
        now = datetime.now()
        
        alarm_level = rule.get("alarm_level", "high") if rule else "high"
        timeout_seconds = config_manager.get_alarm_timeout(alarm_level)
        
        alarm = {
            "id": len(self.alarm_history) + 1,
            "type": event_type,
            "message": message,
            "camera_id": camera_id,
            "frame_path": frame_path,
            "timestamp": now.isoformat(),
            "timeout_at": (now + timedelta(seconds=timeout_seconds)).isoformat(),
            "status": "active",
            "acknowledged": False,
            "notified": False,
            "alarm_level": alarm_level,
            "notify_strategy": rule.get("notify_strategy", ["wecom"]) if rule else ["wecom"]
        }
        self.active_alarms.append(alarm)
        self.alarm_history.append(alarm)
        
        if len(self.alarm_history) > 1000:
            self.alarm_history = self.alarm_history[-1000:]
            
        return alarm
    
    def acknowledge_alarm(self, alarm_id: int):
        for alarm in self.active_alarms:
            if alarm["id"] == alarm_id:
                alarm["acknowledged"] = True
                alarm["status"] = "acknowledged"
                if alarm_id in self.notified_alarms:
                    self.notified_alarms.remove(alarm_id)
                return True
        return False
    
    def check_timeout_alarms(self):
        """检查超时未确认的告警并发送通知"""
        now = datetime.now()
        notifications = []
        
        for alarm in self.active_alarms:
            if not alarm["acknowledged"] and not alarm["notified"]:
                timeout_time = datetime.fromisoformat(alarm["timeout_at"])
                if now >= timeout_time:
                    alarm["notified"] = True
                    alarm["status"] = "timeout"
                    self.notified_alarms.add(alarm["id"])
                    
                    notification = self.send_notification(alarm)
                    notifications.append(notification)
        
        return notifications
    
    def send_notification(self, alarm):
        """发送告警通知到值班人员通信设备"""
        notification = {
            "alarm_id": alarm["id"],
            "type": alarm["type"],
            "message": alarm["message"],
            "timestamp": datetime.now().isoformat(),
            "camera_id": alarm["camera_id"],
            "status": "sent",
            "sms_sent": False,
            "sms_errors": []
        }
        
        print(f"🚨 发送告警通知到值班人员: {alarm['message']} (告警ID: {alarm['id']})")
        print(f"   通知时间: {notification['timestamp']}")
        print(f"   摄像头: {alarm['camera_id']}")
        
        if SMS_AVAILABLE and ON_CALL_PHONES:
            for phone in ON_CALL_PHONES:
                result = send_alarm_sms(phone, alarm)
                if result["success"]:
                    notification["sms_sent"] = True
                    print(f"📱 短信发送成功: {phone}")
                else:
                    notification["sms_errors"].append(f"{phone}: {result.get('message', '发送失败')}")
                    print(f"❌ 短信发送失败 {phone}: {result.get('message', '未知错误')}")
        else:
            print("ℹ️ 短信服务未配置，跳过短信发送")
        
        self.log_notification(notification)
        
        return notification
    
    def log_notification(self, notification):
        """记录通知日志"""
        log_dir = "./notifications"
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"notifications_{datetime.now().strftime('%Y%m%d')}.log")
        with open(log_file, "a") as f:
            f.write(f"{notification['timestamp']} | ID:{notification['alarm_id']} | {notification['type']} | {notification['message']}\n")
    
    def get_active_alarms(self):
        return [a for a in self.active_alarms if not a["acknowledged"]]
    
    def get_alarm_history(self, limit: int = 100):
        return self.alarm_history[-limit:]
    
    def track_object(self, object_id: str, bbox: Tuple[int, int, int, int], timestamp: float):
        if object_id not in self.object_tracks:
            self.object_tracks[object_id] = {
                "positions": [],
                "first_seen": timestamp,
                "last_seen": timestamp
            }
        
        self.object_tracks[object_id]["positions"].append({
            "bbox": bbox,
            "timestamp": timestamp
        })
        self.object_tracks[object_id]["last_seen"] = timestamp
        
        if len(self.object_tracks[object_id]["positions"]) > 60:
            self.object_tracks[object_id]["positions"] = self.object_tracks[object_id]["positions"][-60:]
        
        self.clean_old_tracks(timestamp)
        
    def clean_old_tracks(self, current_timestamp: float):
        to_remove = []
        for obj_id, track in self.object_tracks.items():
            if current_timestamp - track["last_seen"] > self.max_track_age:
                to_remove.append(obj_id)
        
        for obj_id in to_remove:
            del self.object_tracks[obj_id]
    
    def detect_loitering(self, object_id: str, current_timestamp: float) -> bool:
        if object_id not in self.object_tracks:
            return False
        
        track = self.object_tracks[object_id]
        elapsed = current_timestamp - track["first_seen"]
        
        if elapsed < self.loitering_threshold:
            return False
        
        positions = track["positions"]
        if len(positions) < 10:
            return False
        
        centers = []
        for pos in positions[-10:]:
            x1, y1, x2, y2 = pos["bbox"]
            centers.append(((x1 + x2) // 2, (y1 + y2) // 2))
        
        if len(centers) < 2:
            return False
        
        max_dist = 0
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dist = np.sqrt((centers[i][0] - centers[j][0])**2 + (centers[i][1] - centers[j][1])**2)
                max_dist = max(max_dist, dist)
        
        return max_dist < 50
    
    def detect_abandoned_object(self, object_id: str, current_timestamp: float, frame: np.ndarray) -> bool:
        if object_id not in self.object_tracks:
            return False
        
        track = self.object_tracks[object_id]
        elapsed = current_timestamp - track["first_seen"]
        
        if elapsed < 60:
            return False
        
        positions = track["positions"]
        if len(positions) < 30:
            return False
        
        recent_positions = positions[-30:]
        centers = []
        for pos in recent_positions:
            x1, y1, x2, y2 = pos["bbox"]
            centers.append(((x1 + x2) // 2, (y1 + y2) // 2))
        
        if len(centers) < 2:
            return False
        
        max_dist = 0
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dist = np.sqrt((centers[i][0] - centers[j][0])**2 + (centers[i][1] - centers[j][1])**2)
                max_dist = max(max_dist, dist)
        
        return max_dist < 30

alarm_manager = AlarmManager()

def detect_crossing_line(bbox: Tuple[int, int, int, int], line_points: List[Tuple[int, int]], prev_bbox: Optional[Tuple[int, int, int, int]] = None) -> bool:
    if prev_bbox is None:
        return False
    
    curr_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
    prev_center = ((prev_bbox[0] + prev_bbox[2]) // 2, (prev_bbox[1] + prev_bbox[3]) // 2)
    
    return line_intersect_segment(curr_center, prev_center, line_points[0], line_points[1])

def line_intersect_segment(p1: Tuple[int, int], p2: Tuple[int, int], l1: Tuple[int, int], l2: Tuple[int, int]) -> bool:
    def ccw(A, B, C):
        return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
    
    A, B = p1, p2
    C, D = l1, l2
    
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def generate_alarm_message(event_type: str, details: Dict = None) -> str:
    messages = {
        "fall": "⚠ 检测到人员摔倒",
        "danger_zone": "⚠ 检测到人员进入危险区域",
        "fire_exit": "⚠ 检测到消防通道被占用",
        "corridor": "⚠ 检测到楼道区域有异常",
        "crossing": "⚠ 检测到人员越界",
        "loitering": "⚠ 检测到可疑徘徊行为",
        "abandoned": "⚠ 检测到疑似遗留物",
        "intrusion": "⚠ 检测到入侵行为"
    }
    
    base_message = messages.get(event_type, f"⚠ 检测到异常事件: {event_type}")
    
    if details:
        base_message += f" ({', '.join(f'{k}: {v}' for k, v in details.items())})"
    
    return base_message