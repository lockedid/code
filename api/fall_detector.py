from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import numpy as np


class FallState(Enum):
    """摔倒检测状态枚举"""
    NORMAL = "normal"              # 正常状态
    SUSPECT = "suspect"            # 疑似摔倒
    CONFIRMED = "confirmed"        # 确认摔倒（已报警）
    COOLDOWN = "cooldown"          # 冷却中
    RECOVERED = "recovered"        # 已恢复


class PersonState:
    """个人状态记录"""
    def __init__(self, track_id: int):
        self.track_id = track_id
        self.state = FallState.NORMAL
        self.fall_scores: List[float] = []  # 最近15帧的分数
        self.bboxes: List[Tuple[int, int, int, int]] = []  # 最近帧的bbox
        self.recent_keypoints: List[List] = []  # 最近帧的关键点
        self.has_alarmed = False
        self.last_alarm_time: Optional[datetime] = None
        self.suspect_start_time: Optional[datetime] = None
        self.confirmed_time: Optional[datetime] = None
        self.last_update_time: datetime = datetime.now()
        self.center_y_history: List[float] = []  # 中心点Y坐标历史，用于检测快速下移
        
        # 计算参数
        self.pose_conf_threshold = 0.5
        self.fall_score_threshold = 0.65
        self.confirm_window_frames = 15
        self.confirm_min_fall_frames = 8
        self.confirm_min_duration = 1.0  # 秒
        self.recovery_duration = 2.0  # 秒
        self.cooldown_seconds = 30
        
    def update(self, fall_score: float, bbox: Tuple[int, int, int, int], keypoints: List):
        """更新个人状态"""
        now = datetime.now()
        self.last_update_time = now
        
        # 保存分数和bbox
        self.fall_scores.append(fall_score)
        self.bboxes.append(bbox)
        self.recent_keypoints.append(keypoints)
        
        # 计算中心点Y坐标并保存
        center_y = (bbox[1] + bbox[3]) / 2
        self.center_y_history.append(center_y)
        
        # 限制历史长度
        if len(self.fall_scores) > self.confirm_window_frames:
            self.fall_scores.pop(0)
        if len(self.bboxes) > self.confirm_window_frames:
            self.bboxes.pop(0)
        if len(self.recent_keypoints) > self.confirm_window_frames:
            self.recent_keypoints.pop(0)
        if len(self.center_y_history) > 10:
            self.center_y_history.pop(0)
            
        # 根据分数更新状态
        self._update_state(now)
        
    def _update_state(self, now: datetime):
        """根据当前分数历史更新状态"""
        if len(self.fall_scores) < 3:
            return
            
        # 计算疑似帧数
        suspect_frames = sum(1 for score in self.fall_scores if score >= self.fall_score_threshold)
        
        if self.state == FallState.NORMAL:
            # 如果连续多帧高分数，进入疑似状态
            if suspect_frames >= self.confirm_min_fall_frames:
                self.state = FallState.SUSPECT
                self.suspect_start_time = now
                
        elif self.state == FallState.SUSPECT:
            # 检查是否满足确认条件
            duration_seconds = (now - self.suspect_start_time).total_seconds() if self.suspect_start_time else 0
            if (suspect_frames >= self.confirm_min_fall_frames and 
                duration_seconds >= self.confirm_min_duration and
                not self.has_alarmed):
                # 满足条件，确认摔倒
                self.state = FallState.CONFIRMED
                self.confirmed_time = now
                self.has_alarmed = True
                self.last_alarm_time = now
                
        elif self.state == FallState.CONFIRMED:
            # 保持确认状态直到恢复
            if self._check_recovered():
                self.state = FallState.RECOVERED
            
        elif self.state == FallState.COOLDOWN:
            # 检查是否应该结束冷却
            if self.last_alarm_time:
                cooldown_passed = (now - self.last_alarm_time).total_seconds() >= self.cooldown_seconds
                recovered = self._check_recovered()
                
                if cooldown_passed and recovered:
                    self.state = FallState.RECOVERED
                    
        elif self.state == FallState.RECOVERED:
            # 重置状态
            if self._check_fully_normal():
                self.state = FallState.NORMAL
                self.has_alarmed = False
                self.last_alarm_time = None
                
    def _check_recovered(self) -> bool:
        """检查是否恢复站立（简单版）"""
        if len(self.fall_scores) < 5:
            return False
        # 最近5帧平均分数较低表示恢复
        recent_avg = sum(self.fall_scores[-5:]) / 5
        return recent_avg < 0.3
        
    def _check_fully_normal(self) -> bool:
        """检查是否完全恢复正常"""
        if len(self.fall_scores) < 10:
            return False
        # 最近10帧平均分数很低
        recent_avg = sum(self.fall_scores[-10:]) / 10
        return recent_avg < 0.2
        
    def should_alarm(self) -> bool:
        """判断是否应该报警（每次确认摔倒都触发报警）"""
        return self.state == FallState.CONFIRMED
        
    def get_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """获取最新的bbox"""
        return self.bboxes[-1] if self.bboxes else None


class FallStateTracker:
    """摔倒检测状态追踪器 - 按人维护状态"""
    
    def __init__(self):
        self.people: Dict[int, PersonState] = {}  # track_id -> PersonState
        self.last_cleanup_time = datetime.now()
        self.last_alarm_time: Optional[datetime] = None
        self.cooldown_seconds = 5  # 检测到摔倒后暂停5秒
        
        # 计算参数
        self.pose_conf_threshold = 0.5
        self.fall_score_threshold = 0.65
        
    def calculate_fall_score(self, keypoints: List, confidences: List, 
                            bbox: Tuple[int, int, int, int], 
                            track_id: Optional[int] = None) -> Tuple[float, Dict[str, float]]:
        """
        计算单帧摔倒分数，返回分数和各因子的贡献
        
        返回:
            score: 0.0 ~ 1.0 的摔倒分数
            factors: 各因子的具体分数
        """
        factors = {}
        
        # 安全检查
        if keypoints is None or len(keypoints) < 17 or confidences is None or len(confidences) < 17:
            factors['error'] = 'invalid_keypoints'
            return 0.0, factors
            
        if bbox is None or len(bbox) != 4:
            factors['error'] = 'invalid_bbox'
            return 0.0, factors
        
        # 1. 关键点质量检查
        quality_score = self._check_keypoint_quality(keypoints, confidences)
        if quality_score < 0.5:
            factors['quality_penalty'] = -0.30
            return 0.0, factors
            
        try:
            nose = keypoints[0]
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]
            left_hip = keypoints[11]
            right_hip = keypoints[12]
            left_knee = keypoints[13]
            right_knee = keypoints[14]
            left_ankle = keypoints[15]
            right_ankle = keypoints[16]
            
            shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2, 
                              (left_shoulder[1] + right_shoulder[1]) / 2)
            hip_center = ((left_hip[0] + right_hip[0]) / 2, 
                         (left_hip[1] + right_hip[1]) / 2)
            
            x1, y1, x2, y2 = bbox
            box_width = x2 - x1
            box_height = y2 - y1
            
            # 安全检查边界框尺寸
            if box_width <= 0 or box_height <= 0:
                factors['error'] = 'invalid_box_size'
                return 0.0, factors
            
            score = 0.0
            total_weight = 0.0
            
            # 2. 人体框高宽比变低 (权重: 25%)
            if box_height > 0 and box_width > 0:
                aspect_ratio = box_height / box_width
                if aspect_ratio < 0.8:  # 横躺
                    factors['aspect_ratio'] = 1.0
                elif aspect_ratio < 1.2:
                    factors['aspect_ratio'] = 0.6
                else:
                    factors['aspect_ratio'] = 0.0
                score += factors['aspect_ratio'] * 0.25
                total_weight += 0.25
                    
            # 3. 肩膀-髋部方向接近水平 (权重: 20%)
            shoulder_hip_dx = abs(hip_center[0] - shoulder_center[0])
            shoulder_hip_dy = abs(hip_center[1] - shoulder_center[1])
            if shoulder_hip_dy > 0:
                horizontal_ratio = shoulder_hip_dx / (shoulder_hip_dy + 1e-6)
                if horizontal_ratio > 1.5:  # 身体接近水平
                    factors['horizontal_body'] = 1.0
                elif horizontal_ratio > 1.0:
                    factors['horizontal_body'] = 0.5
                else:
                    factors['horizontal_body'] = 0.0
                score += factors['horizontal_body'] * 0.20
                total_weight += 0.20
                    
            # 4. 头部明显低于肩膀/髋部 (权重: 15%)
            nose_y = nose[1]
            if nose_y > shoulder_center[1] + 30:  # 鼻子在肩膀下方
                factors['head_low'] = 1.0
            elif nose_y > hip_center[1]:
                factors['head_low'] = 0.67
            else:
                factors['head_low'] = 0.0
            score += factors['head_low'] * 0.15
            total_weight += 0.15
                
            # 5. 关键点整体高度/宽度变低 (权重: 25%)
            if shoulder_hip_dy > 0:
                if shoulder_hip_dy < box_height * 0.3:  # 躯干压缩得很短
                    factors['compressed_trunk'] = 1.0
                elif shoulder_hip_dy < box_height * 0.5:
                    factors['compressed_trunk'] = 0.4
                else:
                    factors['compressed_trunk'] = 0.0
                score += factors['compressed_trunk'] * 0.25
                total_weight += 0.25
                    
            # 6. 如果有track_id，可以检查身体中心快速下移 (权重: 10%)
            factors['fast_down'] = 0.0
            if track_id is not None and track_id in self.people:
                person = self.people[track_id]
                if len(person.center_y_history) >= 5:
                    # 计算最近5帧的Y移动速度
                    recent_ys = person.center_y_history[-5:]
                    avg_y_change = abs(recent_ys[-1] - recent_ys[0]) / 5
                    if avg_y_change > 20:  # 快速下移
                        factors['fast_down'] = 1.0
                    elif avg_y_change > 10:
                        factors['fast_down'] = 0.5
                    else:
                        factors['fast_down'] = 0.0
                    score += factors['fast_down'] * 0.10
                    total_weight += 0.10
                
            # 7. 膝盖和脚踝距离很近 (权重: 5%)
            knee_score = 0.0
            if confidences[13] > self.pose_conf_threshold and confidences[14] > self.pose_conf_threshold:
                knee_distance = abs(left_knee[0] - right_knee[0])
                if knee_distance > box_height * 0.4:
                    knee_score = 1.0
            factors['knee_spread'] = knee_score
            score += knee_score * 0.05
            total_weight += 0.05
            
            # 归一化到0-1范围
            if total_weight > 0:
                score = min(score / total_weight, 1.0)
            else:
                score = 0.0
            
            # 8. 脚踝距离很近 (权重: 5%)
            ankle_score = 0.0
            if confidences[15] > self.pose_conf_threshold and confidences[16] > self.pose_conf_threshold:
                ankle_distance = abs(left_ankle[0] - right_ankle[0])
                if ankle_distance > box_height * 0.5:
                    ankle_score = 1.0
            factors['ankle_spread'] = ankle_score
            score += ankle_score * 0.05
            total_weight += 0.05
            
            # 重新归一化到0-1范围
            if total_weight > 0:
                score = min(score / total_weight, 1.0)
            else:
                score = 0.0
                
            # 限制最高分
            score = min(1.0, score)
            factors['total'] = score
            
            return score, factors
            
        except Exception as e:
            factors['error'] = str(e)
            return 0.0, factors
        
    def _check_keypoint_quality(self, keypoints: List, confidences: List) -> float:
        """
        检查关键点质量
        返回: 0.0 ~ 1.0 的质量分数
        """
        if keypoints is None or len(keypoints) < 17:
            return 0.0
            
        visible_count = 0
        min_visible = 6
        
        # 检查关键部位
        left_shoulder_ok = confidences[5] > self.pose_conf_threshold
        right_shoulder_ok = confidences[6] > self.pose_conf_threshold
        left_hip_ok = confidences[11] > self.pose_conf_threshold
        right_hip_ok = confidences[12] > self.pose_conf_threshold
        
        # 至少一个肩和一个髋可见
        if not (left_shoulder_ok or right_shoulder_ok):
            return 0.2
        if not (left_hip_ok or right_hip_ok):
            return 0.2
            
        # 统计可见点数量
        for i, conf in enumerate(confidences):
            if conf > self.pose_conf_threshold:
                visible_count += 1
                
        if visible_count < min_visible:
            return 0.3
            
        # 根据可见点数量给分
        quality_score = min(1.0, visible_count / 12.0)
        return quality_score
        
    def update_person(self, track_id: int, keypoints: List, confidences: List, 
                     bbox: Tuple[int, int, int, int]) -> Tuple[bool, float, Dict]:
        """
        更新一个人的状态
        
        返回:
            should_alarm: 是否应该报警
            fall_score: 当前帧的摔倒分数
            factors: 分数因子
        """
        # 检查冷却状态
        now = datetime.now()
        if self.last_alarm_time:
            time_since_alarm = (now - self.last_alarm_time).total_seconds()
            if time_since_alarm < self.cooldown_seconds:
                # 仍在冷却期，不进行检测
                fall_score = 0.0
                factors = {'cooldown': f'冷却中，剩余{self.cooldown_seconds - time_since_alarm:.1f}秒'}
                return False, fall_score, factors
        
        # 计算分数
        fall_score, factors = self.calculate_fall_score(keypoints, confidences, bbox, track_id)
        
        # 获取或创建个人状态
        if track_id not in self.people:
            self.people[track_id] = PersonState(track_id)
            
        person = self.people[track_id]
        
        # 更新状态
        person.update(fall_score, bbox, keypoints)
        
        # 检查是否应该报警
        alarm = person.should_alarm()
        
        # 如果触发报警，更新全局最后报警时间
        if alarm:
            self.last_alarm_time = datetime.now()
        
        return alarm, fall_score, factors
        
    def cleanup(self):
        """清理长时间未更新的人"""
        now = datetime.now()
        timeout = timedelta(seconds=5)
        
        expired_ids = []
        for track_id, person in self.people.items():
            if now - person.last_update_time > timeout:
                expired_ids.append(track_id)
                
        for track_id in expired_ids:
            del self.people[track_id]
            
        self.last_cleanup_time = now
        
    def get_active_falls(self) -> List[Dict]:
        """获取当前活跃的摔倒事件"""
        active = []
        for track_id, person in self.people.items():
            if person.state in [FallState.CONFIRMED, FallState.COOLDOWN]:
                active.append({
                    'track_id': track_id,
                    'state': person.state.value,
                    'score': person.fall_scores[-1] if person.fall_scores else 0,
                    'bbox': person.get_bbox(),
                    'alarmed': person.has_alarmed,
                    'alarm_time': person.last_alarm_time
                })
        return active


# 全局摔倒追踪器实例
fall_tracker = FallStateTracker()