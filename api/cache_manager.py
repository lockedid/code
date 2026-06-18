from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import hashlib
import json
import os
from functools import lru_cache
import cv2
import numpy as np

class CacheEntry:
    """缓存条目"""
    def __init__(self, key: str, value: Any, ttl: int = 300):
        self.key = key
        self.value = value
        self.created_at = datetime.now().timestamp()
        self.ttl = ttl
        self.access_count = 1
        self.last_accessed = self.created_at
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        return datetime.now().timestamp() - self.created_at > self.ttl
    
    def update_access(self):
        """更新访问次数和时间"""
        self.access_count += 1
        self.last_accessed = datetime.now().timestamp()

class CacheManager:
    """智能缓存管理器"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        """
        初始化缓存管理器
        
        Args:
            max_size: 最大缓存条目数
            default_ttl: 默认过期时间（秒）
        """
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.hits = 0
        self.misses = 0
        
    def _generate_key(self, prefix: str, *args) -> str:
        """生成缓存key"""
        key_str = f"{prefix}:{':'.join(str(arg) for arg in args)}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, prefix: str, *args) -> Optional[Any]:
        """获取缓存"""
        key = self._generate_key(prefix, *args)
        
        if key not in self.cache:
            self.misses += 1
            return None
        
        entry = self.cache[key]
        
        if entry.is_expired():
            del self.cache[key]
            self.misses += 1
            return None
        
        entry.update_access()
        self.hits += 1
        return entry.value
    
    def set(self, prefix: str, value: Any, *args, ttl: Optional[int] = None) -> str:
        """设置缓存"""
        key = self._generate_key(prefix, *args)
        
        # 如果缓存已满，执行淘汰策略
        if len(self.cache) >= self.max_size:
            self._evict()
        
        self.cache[key] = CacheEntry(key, value, ttl or self.default_ttl)
        return key
    
    def _evict(self):
        """缓存淘汰策略：LRU + LFU混合"""
        if not self.cache:
            return
        
        # 优先淘汰过期的
        expired_keys = [k for k, entry in self.cache.items() if entry.is_expired()]
        if expired_keys:
            for key in expired_keys:
                del self.cache[key]
            return
        
        # LRU + LFU混合策略
        now = datetime.now().timestamp()
        scores = {}
        
        for key, entry in self.cache.items():
            # 得分 = 访问频率 * 时间衰减因子
            time_since_access = now - entry.last_accessed
            decay_factor = 1 / (1 + time_since_access / 300)  # 5分钟半衰期
            score = entry.access_count * decay_factor
            scores[key] = score
        
        # 删除得分最低的10%
        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k])
        to_remove = sorted_keys[:max(1, len(sorted_keys) // 10)]
        
        for key in to_remove:
            del self.cache[key]
    
    def invalidate(self, prefix: str, *args):
        """失效指定缓存"""
        key = self._generate_key(prefix, *args)
        if key in self.cache:
            del self.cache[key]
    
    def invalidate_prefix(self, prefix: str):
        """失效指定前缀的所有缓存"""
        keys_to_remove = [k for k in self.cache.keys() if k.startswith(self._generate_key(prefix))]
        for key in keys_to_remove:
            del self.cache[key]
    
    def clear(self):
        """清空所有缓存"""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        hit_rate = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate * 100, 2)
        }
    
    def cleanup(self):
        """清理过期缓存"""
        expired_keys = [k for k, entry in self.cache.items() if entry.is_expired()]
        for key in expired_keys:
            del self.cache[key]

class FrameCache:
    """帧级缓存"""
    
    def __init__(self):
        self.detection_cache = CacheManager(max_size=500, default_ttl=60)
        self.pose_cache = CacheManager(max_size=500, default_ttl=60)
        self.vlm_cache = CacheManager(max_size=100, default_ttl=300)
    
    def get_detection(self, frame_hash: str) -> Optional[Any]:
        """获取检测结果缓存"""
        return self.detection_cache.get("detection", frame_hash)
    
    def set_detection(self, frame_hash: str, result: Any):
        """设置检测结果缓存"""
        self.detection_cache.set("detection", result, frame_hash)
    
    def get_pose(self, frame_hash: str) -> Optional[Any]:
        """获取姿态估计缓存"""
        return self.pose_cache.get("pose", frame_hash)
    
    def set_pose(self, frame_hash: str, result: Any):
        """设置姿态估计缓存"""
        self.pose_cache.set("pose", result, frame_hash)
    
    def get_vlm_description(self, video_segment_key: str) -> Optional[str]:
        """获取VLM描述缓存"""
        return self.vlm_cache.get("vlm", video_segment_key)
    
    def set_vlm_description(self, video_segment_key: str, description: str):
        """设置VLM描述缓存"""
        self.vlm_cache.set("vlm", description, video_segment_key, ttl=600)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取所有缓存统计"""
        return {
            "detection": self.detection_cache.get_stats(),
            "pose": self.pose_cache.get_stats(),
            "vlm": self.vlm_cache.get_stats()
        }
    
    def cleanup(self):
        """清理所有过期缓存"""
        self.detection_cache.cleanup()
        self.pose_cache.cleanup()
        self.vlm_cache.cleanup()
    
    def clear(self):
        """清空所有缓存"""
        self.detection_cache.clear()
        self.pose_cache.clear()
        self.vlm_cache.clear()

def compute_frame_hash(frame: np.ndarray) -> str:
    """计算帧的哈希值用于缓存key"""
    # 使用帧的缩小版本计算哈希，提高效率
    small_frame = cv2.resize(frame, (64, 64))
    return hashlib.md5(small_frame.tobytes()).hexdigest()

import cv2

# 全局缓存实例
frame_cache = FrameCache()

# 全局检测结果缓存
detection_cache = CacheManager(max_size=1000, default_ttl=120)

# 全局VLM结果缓存
vlm_result_cache = CacheManager(max_size=200, default_ttl=600)