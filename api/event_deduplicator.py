import time
import threading
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict


class EventDeduplicator:
    
    def __init__(self):
        self._lock = threading.Lock()
        self._last_event_time: Dict[str, float] = {}
        self._event_counts: Dict[str, List[float]] = {}
        self._cooldown_overrides: Dict[str, float] = {}
        self._count_window = 60.0
        self._count_max = 1000
    
    def _get_cooldown(self, event_type: str) -> float:
        try:
            from .config import config_manager
            rule = config_manager.get_alarm_rule(event_type)
            if rule and "min_interval" in rule:
                return float(rule["min_interval"])
        except Exception:
            pass
        defaults = {
            "fall": 30,
            "danger_zone": 10,
            "fire_exit": 5,
            "fire_exit_parking": 5,
            "corridor": 15,
            "corridor_parking": 10,
            "crossing": 8,
            "loitering": 30,
            "abandoned": 60,
            "intrusion": 8,
        }
        return defaults.get(event_type, 10.0)
    
    def _make_key(self, event_type: str, camera_id: int,
                  zone_id: str = None,
                  bbox: Tuple[int, int, int, int] = None) -> str:
        parts = [event_type, str(camera_id)]
        if zone_id:
            parts.append(zone_id)
        if bbox:
            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2
            grid = max(50, 50)
            parts.append(f"g{cx // grid}_{cy // grid}")
        return "_".join(parts)
    
    def _get_frequency_multiplier(self, base_key: str, now: float) -> float:
        with self._lock:
            timestamps = self._event_counts.get(base_key, [])
            cutoff = now - self._count_window
            recent = [t for t in timestamps if t > cutoff]
            self._event_counts[base_key] = recent
            
            if not recent:
                return 1.0
            
            freq = len(recent) / self._count_window
            if freq > 0.2:
                return min(3.0, 1.0 + freq * 2.0)
            return 1.0
    
    def record_event(self, event_type: str, camera_id: int,
                     zone_id: str = None,
                     bbox: Tuple[int, int, int, int] = None,
                     confidence: float = 1.0,
                     event_id: str = None) -> bool:
        now = time.time()
        key = self._make_key(event_type, camera_id, zone_id, bbox)
        base_key = self._make_key(event_type, camera_id, zone_id, None)
        
        cooldown = self._get_cooldown(event_type)
        
        override = self._cooldown_overrides.get(event_type)
        if override is not None:
            cooldown = override
        
        freq_mult = self._get_frequency_multiplier(base_key, now)
        effective_cooldown = cooldown * freq_mult
        
        with self._lock:
            last_time = self._last_event_time.get(key, 0)
            elapsed = now - last_time
            
            if elapsed < effective_cooldown:
                return False
            
            self._last_event_time[key] = now
            
            if base_key not in self._event_counts:
                self._event_counts[base_key] = []
            self._event_counts[base_key].append(now)
            if len(self._event_counts[base_key]) > self._count_max:
                self._event_counts[base_key] = self._event_counts[base_key][-self._count_max:]
        
        return True
    
    def is_duplicate(self, event_type: str, camera_id: int,
                     zone_id: str = None,
                     bbox: Tuple[int, int, int, int] = None,
                     confidence: float = 1.0) -> bool:
        return not self.record_event(event_type, camera_id, zone_id, bbox, confidence)
    
    def set_cooldown_override(self, event_type: str, cooldown: float):
        with self._lock:
            if cooldown is None:
                self._cooldown_overrides.pop(event_type, None)
            else:
                self._cooldown_overrides[event_type] = cooldown
    
    def reset(self, event_type: str = None, camera_id: int = None):
        with self._lock:
            if event_type is None and camera_id is None:
                self._last_event_time.clear()
                self._event_counts.clear()
                return
            
            prefix_parts = []
            if event_type:
                prefix_parts.append(event_type)
            if camera_id is not None:
                prefix_parts.append(str(camera_id))
            prefix = "_".join(prefix_parts)
            
            keys_to_remove = [
                k for k in self._last_event_time if k.startswith(prefix)
            ]
            for k in keys_to_remove:
                del self._last_event_time[k]
            
            count_keys = [
                k for k in self._event_counts if k.startswith(prefix)
            ]
            for k in count_keys:
                del self._event_counts[k]
    
    def cleanup(self):
        now = time.time()
        with self._lock:
            expired = [
                k for k, t in self._last_event_time.items()
                if now - t > 300
            ]
            for k in expired:
                del self._last_event_time[k]
            
            count_expired = []
            for k, timestamps in self._event_counts.items():
                cutoff = now - self._count_window
                recent = [t for t in timestamps if t > cutoff]
                if recent:
                    self._event_counts[k] = recent
                else:
                    count_expired.append(k)
            for k in count_expired:
                del self._event_counts[k]
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "tracked_keys": len(self._last_event_time),
                "frequency_entries": len(self._event_counts),
                "cooldown_overrides": dict(self._cooldown_overrides),
            }


event_deduplicator = EventDeduplicator()