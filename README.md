# NeuroCortex Eye - 智能视频分析系统

基于 YOLO 和 Qwen2.5-VL 本地大模型的智能视频分析系统，支持多种目标检测、行为识别、AI 推理分析与实时告警。

## 系统架构

```
视频输入 → 视频解码 → YOLO 目标检测 → ByteTrack 多目标跟踪 → 行为分析 → 事件管理 → AI 推理分析
                ↓                              ↓                      ↓
          姿态估计 (YOLO-Pose)            摔倒检测             Qwen2.5-VL 视觉描述
                ↓                              ↓                      ↓
          缓存管理                        事件去重              告警通知 (短信/Webhook)
```

## 核心功能

### 1. 目标检测
- **人员检测** (Person)
- **机动车检测** (Car, Bus, Truck)
- **电动车检测** (Electric Bike, Electric Scooter)
- **姿态估计** (YOLO-Pose, 17 关键点)

### 2. 行为分析
| 行为类型 | 描述 | 严重级别 |
|---------|------|---------|
| 危险区域闯入 | 人员进入预设危险区域 | High |
| 楼道电动车停放 | 电动车在楼道违规停放 | Medium |
| 消防通道占用 | 机动车占用消防通道 | High |
| 人员摔倒检测 | 基于姿态估计的摔倒判断 | Critical |

### 3. AI 大模型集成
- **Qwen2.5-VL-7B**: 本地部署视觉语言模型，支持 4-bit 量化推理
- 自动生成事件文字描述
- 视频帧分析问答
- 对话历史管理

### 4. AI 推理分析
- 多事件类型风险评估 (Critical / High / Medium / Low)
- 事件推理链分析
- 自动生成处置建议和 AI 分析报告
- 场景模拟推理

### 5. 告警系统
- 多级告警等级 (紧急 / 高 / 中 / 低)
- 超时未确认自动升级通知
- 多渠道通知策略 (企业微信 / 钉钉 / 邮件 / 短信)
- 事件去重与冷却机制
- 告警确认与历史记录

### 6. 视频流处理
- 本地视频文件上传分析
- RTSP 网络摄像头实时流接入
- RTMP 推流支持
- MJPEG 实时流输出
- WebSocket 实时帧推送
- 录像存储与回放

### 7. Web 管理平台
- 实时视频监控面板 (Dashboard)
- 视频流在线预览
- 录像回放
- 危险区域可视化配置
- 告警规则动态管理

## 目录结构

```
smart_video_ai/
├── main.py                  # CLI 命令行入口
├── video_processor.py       # 视频处理核心流水线
├── requirements.txt         # Python 依赖
├── install.sh               # 环境安装脚本
├── download_cn.py           # 模型下载脚本 (国内镜像)
├── configs/                 # 系统配置模块
│   ├── config.py            # 配置数据类定义
│   └── __init__.py
├── config/                  # 运行时配置文件
│   └── alarm_rules.json     # 危险区域 & 告警规则配置
├── detectors/               # 目标检测器
│   ├── base_detector.py     # 检测器基类
│   ├── yolo_detector.py     # YOLO 检测器 (Person/Vehicle/Fire/EBike)
│   └── pose_detector.py     # 姿态检测器 (YOLO-Pose)
├── trackers/                # 多目标跟踪器
│   ├── base_tracker.py      # 跟踪器基类
│   └── bytetrack_tracker.py # ByteTrack 跟踪器实现
├── behaviors/               # 行为分析引擎
│   ├── base_behavior.py     # 行为分析基类 & 事件定义
│   ├── intrusion_detector.py # 闯入检测
│   ├── parking_detector.py  # 违规停车检测
│   ├── fall_detector.py     # 摔倒检测
│   └── loitering_detector.py # 逗留检测
├── events/                  # 事件管理中心
│   └── event_manager.py     # 事件记录、存储、查询
├── database/                # 数据库模块
│   └── database.py          # PostgreSQL 连接 & 事件持久化
├── api/                     # REST API 服务
│   ├── main.py              # FastAPI 主应用 (全部路由)
│   ├── config.py            # 配置管理器 (JSON 读写)
│   ├── database.py          # SQLite 数据库 (用户/摄像机/事件/录像)
│   ├── alarm.py             # 告警管理器 (告警生命周期)
│   ├── sms.py               # 短信服务 (阿里/腾讯/华为云)
│   ├── vlm.py               # VLM 大模型处理器
│   ├── async_processor.py   # 异步视频处理
│   ├── cache_manager.py     # 智能缓存管理
│   ├── event_deduplicator.py # 事件去重器
│   └── fall_detector.py     # 摔倒状态机 (Normal/Suspect/Confirmed/Cooldown)
├── reasoning/               # AI 推理模块
│   ├── event_reasoner.py    # 事件推理引擎
│   ├── context_builder.py   # 上下文构建器
│   ├── decision_engine.py   # 决策引擎
│   ├── event_graph.py       # 事件图谱
│   ├── recommendation_engine.py # 处置建议引擎
│   ├── report_generator.py  # 报告生成器
│   ├── risk_evaluator.py    # 风险评估器
│   ├── rule_engine.py       # 规则引擎
│   ├── schemas.py           # 数据模型定义
│   └── timeline_builder.py  # 时间线构建器
├── frontend/                # Web 前端
│   ├── index.html           # 主页面 (视频预览)
│   ├── dashboard.html       # 仪表盘 (统计面板)
│   └── playback.html        # 录像回放
├── models/                  # 模型文件目录
│   ├── yolo11m.pt           # YOLOv11 检测模型
│   ├── yolo11s.pt           # YOLOv11 轻量模型
│   ├── yolo11m-pose.pt      # YOLOv11 姿态估计模型
│   └── Qwen2.5-VL-7B-Instruct/ # Qwen 视觉语言模型
├── videos/                  # 输入视频目录
├── outputs/                 # 输出文件目录
├── recordings/              # 录像存储目录
├── data/                    # 本地数据库 (SQLite)
└── notifications/           # 通知日志
```

## 快速开始

### 环境要求

- Python 3.10+
- CUDA 12.1+ (推荐，用于 GPU 推理)
- 16GB+ 显存 (Qwen2.5-VL 7B 模型)

### 1. 安装依赖

```bash
# 使用安装脚本 (配置国内镜像源)
bash install.sh

# 或手动安装
pip install -r requirements.txt
```

### 2. 下载模型

```bash
# 从国内镜像下载 YOLO 模型
python download_cn.py

# Qwen2.5-VL 模型需手动下载到 models/Qwen2.5-VL-7B-Instruct/
```

### 3. 命令行分析视频

```bash
python main.py --video ./videos/test.mp4 --output ./outputs
```

### 4. 启动 API 服务

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

启动后访问:
- **Web 管理界面**: http://localhost:8000/view
- **仪表盘**: http://localhost:8000/dashboard
- **录像回放**: http://localhost:8000/playback
- **API 文档**: http://localhost:8000/docs

## API 接口概览

### 用户认证
| 接口 | 方法 | 描述 |
|------|------|------|
| `/api/login` | POST | 用户登录 |
| `/api/register` | POST | 用户注册 |

### 视频分析
| 接口 | 方法 | 描述 |
|------|------|------|
| `/analyze/video` | POST | 上传并分析视频文件 |
| `/stream` | GET | MJPEG 实时流 |
| `/stream/rtmp` | GET | RTMP 实时流 |
| `/ws/stream` | WebSocket | WebSocket 实时流 |

### 摄像头管理
| 接口 | 方法 | 描述 |
|------|------|------|
| `/api/cameras` | GET | 获取摄像头列表 |
| `/api/cameras` | POST | 添加摄像头 |
| `/api/cameras/{id}/stream` | GET | 获取摄像头流 |
| `/api/rtmp/connect` | POST | 连接 RTMP 流 |
| `/api/rtmp/streams` | GET | 获取 RTMP 流列表 |
| `/api/rtmp/stream/{id}/disconnect` | POST | 断开 RTMP 流 |

### 录像管理
| 接口 | 方法 | 描述 |
|------|------|------|
| `/api/recording/start` | POST | 开始录像 |
| `/api/recording/stop` | POST | 停止录像 |
| `/api/recordings` | GET | 获取录像列表 |
| `/api/recordings/{filename}` | GET | 获取录像文件 |

### 事件管理
| 接口 | 方法 | 描述 |
|------|------|------|
| `/api/events` | GET | 获取事件列表 |
| `/api/events/types` | GET | 获取事件类型统计 |
| `/api/stats` | GET | 获取统计信息 |
| `/api/stats/week` | GET | 获取周统计 |

### 告警管理
| 接口 | 方法 | 描述 |
|------|------|------|
| `/api/alarm/active` | GET | 获取活跃告警 |
| `/api/alarm/history` | GET | 获取告警历史 |
| `/api/alarm/acknowledge/{id}` | POST | 确认告警 |

### VLM 视觉语言模型
| 接口 | 方法 | 描述 |
|------|------|------|
| `/api/vlm/load` | POST | 加载 VLM 模型 |
| `/api/vlm/status` | GET | 获取模型状态 |
| `/api/vlm/describe` | POST | 描述当前帧 |
| `/api/vlm/analyze-frame` | POST | 分析指定帧 |
| `/api/vlm/chat` | POST | 对话交互 |
| `/api/vlm/summarize` | GET | 生成摘要 |
| `/api/vlm/generate-report` | POST | 生成分析报告 |
| `/api/vlm/clear-history` | POST | 清除对话历史 |

### AI 推理
| 接口 | 方法 | 描述 |
|------|------|------|
| `/api/reasoning/reason` | POST | AI 推理分析 |
| `/api/reasoning/current` | GET | 当前决策数据 |
| `/api/reasoning/demo/simulate` | POST | 场景模拟推理 |

### 配置管理
| 接口 | 方法 | 描述 |
|------|------|------|
| `/danger-zones` | GET/POST | 危险区域管理 |
| `/danger-zones/{id}` | PUT/DELETE | 危险区域修改/删除 |
| `/api/config/danger-zones` | GET/POST | 配置化危险区域 |
| `/api/config/alarm-rules` | GET | 告警规则查询 |
| `/api/config/alarm-rules/{id}` | PUT | 告警规则修改 |
| `/api/config/reload` | POST | 重新加载配置 |

### 系统状态
| 接口 | 方法 | 描述 |
|------|------|------|
| `/api/fall/status` | GET | 摔倒检测状态 |
| `/api/deduplication/stats` | GET | 去重统计 |
| `/api/cache/stats` | GET | 缓存统计 |

## 配置说明

### 系统配置 (`configs/config.py`)

```python
DetectionConfig   # 检测器配置 (置信度、IOU 阈值、模型路径)
TrackerConfig     # 跟踪器配置 (最大跟踪时长、匹配阈值)
BehaviorConfig    # 行为分析配置 (ROI 区域、时间阈值)
VLMConfig         # 大模型配置 (模型路径、量化方式、温度)
DatabaseConfig    # 数据库配置 (PostgreSQL)
APIConfig         # API 服务配置
VideoConfig       # 视频处理配置
```

### 告警规则配置 (`config/alarm_rules.json`)

```json
{
  "danger_zones": {      // 危险区域定义 (多边形坐标、颜色、告警等级)
  "alarm_rules": {       // 告警规则 (启用状态、阈值、最小间隔)
  "notify_providers": {  // 通知渠道 (企业微信/钉钉/邮件/短信)
  "alarm_levels": {      // 告警等级 (紧急/高/中/低，超时时间)
}
```

## 技术栈

| 类别 | 技术 |
|------|------|
| 视频处理 | OpenCV, FFmpeg |
| 目标检测 | YOLOv11 (Ultralytics) |
| 目标跟踪 | ByteTrack |
| 姿态估计 | YOLO-Pose |
| 大模型 | Qwen2.5-VL-7B (Transformers) |
| API 框架 | FastAPI |
| 数据库 | SQLite (本地), PostgreSQL (生产) |
| 深度学习 | PyTorch 2.2+, CUDA 12.1 |
| 短信服务 | 阿里云 / 腾讯云 / 华为云 SDK |
| 前端 | 原生 HTML/CSS/JS (毛玻璃 UI) |

## 后期扩展

- [ ] Kafka 消息队列集成
- [ ] 分布式推理集群
- [ ] 更多大模型支持 (LLaVA, InternVL)
- [ ] 移动端 App 推送
- [ ] ONNX 模型导出与推理加速
- [ ] Docker 容器化部署
- [ ] Kubernetes 集群编排
