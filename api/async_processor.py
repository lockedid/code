import asyncio
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import hashlib
import time

class FrameProcessor:
    """帧处理器基类"""
    def __init__(self):
        self.model = None
    
    def load_model(self):
        raise NotImplementedError("子类必须实现load_model方法")
    
    def process(self, frame: np.ndarray) -> Any:
        raise NotImplementedError("子类必须实现process方法")

class DetectionTask:
    """检测任务"""
    def __init__(self, frame: np.ndarray, frame_index: int, timestamp: float):
        self.frame = frame
        self.frame_index = frame_index
        self.timestamp = timestamp
        self.future = None

class AsyncVideoProcessor:
    """异步视频处理器"""
    
    def __init__(self, max_workers: int = 4, use_process_pool: bool = False):
        """
        初始化异步视频处理器
        
        Args:
            max_workers: 最大工作线程数
            use_process_pool: 是否使用进程池（适用于CPU密集型任务）
        """
        self.executor_type = ProcessPoolExecutor if use_process_pool else ThreadPoolExecutor
        self.executor = self.executor_type(max_workers=max_workers)
        self.processors: List[FrameProcessor] = []
        self.running = False
        self.task_queue = asyncio.Queue()
        self.result_queue = asyncio.Queue()
        self._lock = asyncio.Lock()
    
    def add_processor(self, processor: FrameProcessor):
        """添加帧处理器"""
        processor.load_model()
        self.processors.append(processor)
    
    async def submit_frame(self, frame: np.ndarray, frame_index: int, timestamp: float) -> asyncio.Future:
        """提交帧处理任务"""
        task = DetectionTask(frame, frame_index, timestamp)
        task.future = asyncio.get_event_loop().create_future()
        await self.task_queue.put(task)
        return task.future
    
    async def _worker(self):
        """工作协程"""
        while self.running:
            try:
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            
            try:
                results = {}
                for processor in self.processors:
                    result = await asyncio.get_event_loop().run_in_executor(
                        self.executor, processor.process, task.frame.copy()
                    )
                    results[processor.__class__.__name__] = result
                
                task.future.set_result({
                    "frame_index": task.frame_index,
                    "timestamp": task.timestamp,
                    "results": results
                })
            except Exception as e:
                task.future.set_exception(e)
            finally:
                self.task_queue.task_done()
    
    async def start(self):
        """启动处理器"""
        self.running = True
        self.workers = [asyncio.create_task(self._worker()) for _ in range(len(self.processors))]
    
    async def stop(self):
        """停止处理器"""
        self.running = False
        await self.task_queue.join()
        
        for worker in self.workers:
            worker.cancel()
        
        self.executor.shutdown(wait=True)
    
    async def process_video(self, video_path: str, callback: Optional[Callable] = None) -> List[Dict]:
        """异步处理整个视频"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        results = []
        frame_index = 0
        
        await self.start()
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = frame_index / fps
                
                # 提交帧处理任务
                future = await self.submit_frame(frame, frame_index, timestamp)
                
                # 如果有回调，立即处理结果
                if callback:
                    asyncio.create_task(self._handle_result(future, callback))
                
                results.append(future)
                frame_index += 1
                
                if frame_index % 100 == 0:
                    print(f"已提交 {frame_index}/{total_frames} 帧")
            
            # 等待所有任务完成
            completed_results = await asyncio.gather(*results, return_exceptions=True)
            return completed_results
            
        finally:
            cap.release()
            await self.stop()
    
    async def _handle_result(self, future: asyncio.Future, callback: Callable):
        """处理单个结果"""
        try:
            result = await future
            await callback(result)
        except Exception as e:
            print(f"帧处理失败: {e}")

class AsyncResult:
    """异步处理结果"""
    def __init__(self, frame_index: int, timestamp: float, results: Dict[str, Any]):
        self.frame_index = frame_index
        self.timestamp = timestamp
        self.results = results
    
    def to_dict(self):
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "results": {k: str(v) for k, v in self.results.items()}
        }

# 全局异步处理器实例
async_processor = AsyncVideoProcessor(max_workers=4, use_process_pool=False)