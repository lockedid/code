# 智能视频分析系统

基于YOLO和本地大模型的智能视频分析系统，支持多种目标检测与行为识别功能。

## 系统架构

```
视频输入模块 → 视频解码模块 → YOLO目标检测引擎 → 多目标跟踪模块 → 行为分析引擎 → 事件管理中心 → VLM → 文字描述
```

## 功能特性

### 目标检测
- 人员检测 (Person)
- 机动车检测 (Car, Bus, Truck)
- 电动车检测 (Electric Bike, Electric Scooter)
- 火灾检测 (Fire, Smoke)

### 行为识别
- 危险区域闯入检测
- 楼道电动车停放检测
- 消防通道占用检测
- 人员摔倒检测
- 长时间逗留检测

### 大模型集成
- Qwen2.5-VL 视觉语言模型
- 自动生成事件文字描述

## 目录结构

```
smart_video_ai/
├── configs/          # 配置文件
├── detectors/        # 检测器模块
├── trackers/         # 跟踪器模块
├── behaviors/        # 行为分析模块
├── vlm/              # 视觉语言模型模块
├── events/           # 事件管理模块
├── database/         # 数据库模块
├── api/              # REST API模块
├── videos/           # 视频文件目录
├── outputs/          # 输出目录
├── main.py           # 主入口
├── video_processor.py # 视频处理器
└── requirements.txt  # 依赖列表
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行分析

```bash
python main.py --video ./videos/test.mp4
```

### 启动API服务

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## API接口

| 接口 | 方法 | 描述 |
|------|------|------|
| / | GET | 健康检查 |
| /analyze/video | POST | 上传并分析视频 |
| /events | GET | 获取事件列表 |
| /events/{event_id} | GET | 获取单个事件详情 |
| /outputs/{filename} | GET | 获取输出文件 |

## 配置说明

配置文件位于 `configs/config.py`，包含以下配置项：

- **DetectionConfig**: 检测器配置（置信度、IOU阈值等）
- **TrackerConfig**: 跟踪器配置（最大跟踪时长等）
- **BehaviorConfig**: 行为分析配置（ROI区域、时间阈值等）
- **VLMConfig**: 大模型配置（模型路径、量化方式等）
- **DatabaseConfig**: 数据库配置
- **APIConfig**: API服务配置
- **VideoConfig**: 视频处理配置

## 技术栈

- **视频处理**: OpenCV, FFmpeg
- **目标检测**: YOLOv11
- **目标跟踪**: ByteTrack
- **姿态估计**: YOLO-Pose
- **大模型**: Qwen2.5-VL
- **API框架**: FastAPI
- **数据库**: PostgreSQL

## 后期扩展

当前版本支持本地视频文件处理，后续可扩展：
- RTSP网络摄像头接入
- Kafka消息队列
- 分布式推理集群
- Web管理平台