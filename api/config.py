import json
import os
from typing import Dict, List, Any

CONFIG_FILE = "./config/alarm_rules.json"

DEFAULT_CONFIG = {
    "danger_zones": {
        "zone1": {
            "name": "危险区域1",
            "points": [(100, 100), (300, 100), (300, 300), (100, 300)],
            "color": [0, 0, 255],
            "enabled": True,
            "zone_type": "general",
            "alarm_level": "high",
            "notify_strategy": ["wecom"]
        },
        "fire_exit": {
            "name": "消防通道",
            "points": [(500, 50), (700, 50), (700, 200), (500, 200)],
            "color": [255, 165, 0],
            "enabled": True,
            "zone_type": "fire_exit",
            "alarm_level": "critical",
            "notify_strategy": ["wecom"]
        },
        "corridor": {
            "name": "楼道区域",
            "points": [(200, 400), (400, 400), (400, 600), (200, 600)],
            "color": [128, 0, 128],
            "enabled": True,
            "zone_type": "corridor",
            "alarm_level": "medium",
            "notify_strategy": ["wecom"]
        }
    },
    "alarm_rules": {
        "fall": {"name": "摔倒检测", "enabled": True, "threshold": 0.8, "alarm_level": "critical", "notify_strategy": ["wecom"], "min_interval": 10},
        "danger_zone": {"name": "危险区域入侵", "enabled": True, "threshold": 0.5, "alarm_level": "high", "notify_strategy": ["wecom"], "min_interval": 10},
        "fire_exit": {"name": "消防通道占用", "enabled": True, "threshold": 0.5, "alarm_level": "critical", "notify_strategy": ["wecom"], "min_interval": 10},
        "corridor": {"name": "楼道异常", "enabled": True, "threshold": 0.5, "alarm_level": "medium", "notify_strategy": ["wecom"], "min_interval": 10},
        "corridor_parking": {"name": "楼道停车", "enabled": True, "threshold": 0.5, "alarm_level": "high", "notify_strategy": ["wecom"], "min_interval": 10},
        "fire_exit_parking": {"name": "消防通道停车", "enabled": True, "threshold": 0.5, "alarm_level": "critical", "notify_strategy": ["wecom"], "min_interval": 10},
        "crossing": {"name": "越界检测", "enabled": True, "threshold": 0.5, "alarm_level": "high", "notify_strategy": ["wecom"], "min_interval": 10},
        "loitering": {"name": "徘徊检测", "enabled": True, "threshold": 30, "alarm_level": "medium", "notify_strategy": ["wecom"], "min_interval": 30},
        "abandoned": {"name": "遗留物检测", "enabled": True, "threshold": 60, "alarm_level": "medium", "notify_strategy": ["wecom"], "min_interval": 60},
        "intrusion": {"name": "入侵检测", "enabled": True, "threshold": 0.5, "alarm_level": "critical", "notify_strategy": ["wecom"], "min_interval": 10}
    },
    "notify_providers": {
        "wecom": {"enabled": True, "priority": 1},
        "dingtalk": {"enabled": False, "priority": 2},
        "email": {"enabled": False, "priority": 3},
        "sms": {"enabled": False, "priority": 4}
    },
    "alarm_levels": {
        "critical": {"name": "紧急", "color": "#ff0000", "timeout": 5},
        "high": {"name": "高", "color": "#ff6600", "timeout": 10},
        "medium": {"name": "中", "color": "#ffff00", "timeout": 15},
        "low": {"name": "低", "color": "#00ff00", "timeout": 30}
    }
}

class ConfigManager:
    def __init__(self):
        self.config = self.load_config()
        self.danger_zones = self.config.get("danger_zones", {})
        self.alarm_rules = self.config.get("alarm_rules", {})
        self.notify_providers = self.config.get("notify_providers", {})
        self.alarm_levels = self.config.get("alarm_levels", {})
    
    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    return json.load(f)
            except:
                return DEFAULT_CONFIG.copy()
        return DEFAULT_CONFIG.copy()
    
    def save_config(self):
        os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
        with open(CONFIG_FILE, "w") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    def get_danger_zones(self):
        return self.danger_zones
    
    def get_danger_zone(self, zone_id):
        return self.danger_zones.get(zone_id, {})
    
    def add_danger_zone(self, zone_id, config):
        defaults = {"name": "新区域", "points": [], "color": [0, 0, 255], "enabled": True, "zone_type": "general", "alarm_level": "high", "notify_strategy": ["wecom"]}
        self.danger_zones[zone_id] = {**defaults, **config}
        self.config["danger_zones"] = self.danger_zones
        self.save_config()
    
    def update_danger_zone(self, zone_id, config):
        if zone_id in self.danger_zones:
            self.danger_zones[zone_id].update(config)
            self.config["danger_zones"] = self.danger_zones
            self.save_config()
            return True
        return False
    
    def delete_danger_zone(self, zone_id):
        if zone_id in self.danger_zones:
            del self.danger_zones[zone_id]
            self.config["danger_zones"] = self.danger_zones
            self.save_config()
            return True
        return False
    
    def get_alarm_rules(self):
        return self.alarm_rules
    
    def get_alarm_rule(self, rule_id):
        return self.alarm_rules.get(rule_id, {})
    
    def update_alarm_rule(self, rule_id, config):
        if rule_id in self.alarm_rules:
            self.alarm_rules[rule_id].update(config)
            self.config["alarm_rules"] = self.alarm_rules
            self.save_config()
            return True
        return False
    
    def get_notify_providers(self):
        return self.notify_providers
    
    def update_notify_provider(self, provider_id, enabled):
        if provider_id in self.notify_providers:
            self.notify_providers[provider_id]["enabled"] = enabled
            self.config["notify_providers"] = self.notify_providers
            self.save_config()
            return True
        return False
    
    def get_alarm_levels(self):
        return self.alarm_levels
    
    def get_alarm_timeout(self, level):
        return self.alarm_levels.get(level, {}).get("timeout", 10)
    
    def reload(self):
        self.config = self.load_config()
        self.danger_zones = self.config.get("danger_zones", {})
        self.alarm_rules = self.config.get("alarm_rules", {})
        self.notify_providers = self.config.get("notify_providers", {})
        self.alarm_levels = self.config.get("alarm_levels", {})
    
    def set_danger_zones(self, zones):
        self.danger_zones = zones
        self.config["danger_zones"] = zones
        self.save_config()

config_manager = ConfigManager()