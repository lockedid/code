import argparse
import os
from video_processor import VideoProcessor

def main():
    parser = argparse.ArgumentParser(description="智能视频分析系统")
    parser.add_argument("--video", type=str, required=True, help="输入视频文件路径")
    parser.add_argument("--output", type=str, default="./outputs", help="输出目录")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"错误：视频文件不存在: {args.video}")
        return
    
    os.makedirs(args.output, exist_ok=True)
    
    processor = VideoProcessor()
    
    try:
        result = processor.process_video_file(args.video)
        
        print("\n分析完成！")
        print(f"视频路径: {result['video_path']}")
        print(f"检测到事件数: {result['total_events']}")
        print(f"输出文件: {result['output_path']}")
        
        if result['events']:
            print("\n事件详情:")
            for i, event in enumerate(result['events'], 1):
                print(f"{i}. [{event['event_type']}] {event['description']}")
        
    except Exception as e:
        print(f"处理过程中发生错误: {e}")

if __name__ == "__main__":
    main()