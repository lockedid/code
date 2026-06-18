from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration, Qwen2VLProcessor
import torch
import cv2
import os
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

class VLMProcessor:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = "/home/wj/code/smart_video_ai/models/Qwen2.5-VL-7B-Instruct"
        self.loaded = False
        self.conversation_history = []
        self.executor = ThreadPoolExecutor(max_workers=1)
    
    def load_model(self):
        if self.loaded:
            return
        
        try:
            print(f"Loading Qwen2.5-VL model from {self.model_path} on {self.device}...")
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"模型目录不存在: {self.model_path}")
            
            self.processor = Qwen2VLProcessor.from_pretrained(self.model_path, trust_remote_code=True, local_files_only=True)
            
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
                use_safetensors=True
            )
            self.model.eval()
            self.loaded = True
            print("Qwen2.5-VL model loaded successfully")
        except FileNotFoundError as e:
            print(f"模型文件未找到: {e}")
            raise
        except Exception as e:
            print(f"Failed to load Qwen2.5-VL model: {e}")
            raise
    
    def describe_frame(self, frame, custom_prompt=None):
        if not self.loaded:
            self.load_model()
        
        try:
            print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")
            
            if frame is None or frame.size == 0:
                return "错误：帧数据为空"
            
            image_pil = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            from PIL import Image
            image_pil = Image.fromarray(image_pil)
            
            print(f"PIL image size: {image_pil.size}, mode: {image_pil.mode}")
            
            prompt = custom_prompt or "请详细描述这张图片中的内容，包括场景、物体和人物活动。"
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            print(f"Prompt text length: {len(text)}")
            
            inputs = self.processor(text=[text], images=[image_pil], return_tensors="pt").to(self.device)
            print(f"Input keys: {list(inputs.keys())}")
            print(f"Input shape: {inputs['input_ids'].shape}")
            
            with torch.no_grad():
                response = self.model.generate(**inputs, max_new_tokens=512, temperature=0.7)
            
            full_response = self.processor.decode(response[0], skip_special_tokens=True)
            print(f"Full response: {full_response[:200]}...")
            
            description = full_response.split("assistant\n")[-1].strip()
            if not description or description.startswith("system"):
                description = full_response.strip()
            
            print(f"Final description: {description[:100]}...")
            return description
        
        except Exception as e:
            print(f"Frame description error: {e}")
            import traceback
            traceback.print_exc()
            return f"描述生成失败: {str(e)}"
    
    def analyze_video_segment(self, video_path, frame_interval=10):
        if not self.loaded:
            self.load_model()
        
        descriptions = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {"error": "无法打开视频文件"}
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        for i in range(0, total_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = i / fps
            description = self.describe_frame(frame)
            descriptions.append({
                "timestamp": f"{timestamp:.2f}秒",
                "frame_index": i,
                "description": description
            })
        
        cap.release()
        return {
            "video_path": video_path,
            "total_frames": total_frames,
            "fps": fps,
            "frame_interval": frame_interval,
            "analysis": descriptions
        }
    
    def generate_event_report(self, event_type, frame=None, context=None):
        if not self.loaded:
            self.load_model()
        
        prompts = {
            "fall": "请分析这张图片，描述是否有人摔倒，以及摔倒的具体情况。",
            "intrusion": "请分析这张图片，描述是否有人员闯入限制区域，以及相关细节。",
            "fire": "请分析这张图片，描述是否存在火灾或烟雾，以及相关细节。",
            "vehicle": "请分析这张图片，描述是否有车辆违规停放，以及相关细节。",
            "crowd": "请分析这张图片，描述人员聚集情况，是否存在安全隐患。",
            "unknown": "请分析这张图片，描述画面中的异常情况。"
        }
        
        prompt = prompts.get(event_type, prompts["unknown"])
        
        if frame is not None:
            try:
                image_pil = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                from PIL import Image
                image_pil = Image.fromarray(image_pil)
                
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
                
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(text=[text], images=[image_pil], return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    response = self.model.generate(**inputs, max_new_tokens=512, temperature=0.7)
                
                full_response = self.processor.decode(response[0], skip_special_tokens=True)
                analysis = full_response.split("assistant\n")[-1].strip()
                if not analysis or analysis.startswith("system"):
                    analysis = full_response.strip()
            except Exception as e:
                analysis = f"图像分析失败: {str(e)}"
        else:
            analysis = "无图像数据可供分析"
        
        report = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "analysis": analysis,
            "context": context or {}
        }
        
        return report
    
    def summarize_video(self, video_path, max_frames=10):
        if not self.loaded:
            self.load_model()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "无法打开视频文件"}
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        
        frame_indices = [int(i * total_frames / max_frames) for i in range(max_frames)]
        frame_indices[-1] = min(frame_indices[-1], total_frames - 1)
        
        descriptions = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                timestamp = idx / fps
                desc = self.describe_frame(frame)
                descriptions.append({
                    "timestamp": f"{timestamp:.2f}秒",
                    "description": desc
                })
        
        cap.release()
        
        combined_text = "\n\n".join([f"{d['timestamp']}: {d['description']}" for d in descriptions])
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"text": f"基于以下视频帧描述，生成一份详细的视频内容总结报告：\n\n{combined_text}\n\n请总结视频中的主要事件、人物活动和场景变化。"}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            response = self.model.generate(**inputs, max_new_tokens=1024, temperature=0.7)
        
        full_summary = self.processor.decode(response[0], skip_special_tokens=True)
        summary = full_summary.split("assistant\n")[-1].strip()
        if not summary or summary.startswith("system"):
            summary = full_summary.strip()
        
        return {
            "video_path": video_path,
            "duration": f"{duration:.2f}秒",
            "fps": fps,
            "total_frames": total_frames,
            "summary": summary,
            "frame_descriptions": descriptions
        }
    
    def chat(self, message, frame=None, max_history=10):
        if not self.loaded:
            self.load_model()
        
        try:
            image_pil = None
            if frame is not None:
                image_pil = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                from PIL import Image
                image_pil = Image.fromarray(image_pil)
            
            self.conversation_history.append({
                "role": "user",
                "content": message
            })
            
            if len(self.conversation_history) > max_history * 2:
                self.conversation_history = self.conversation_history[-max_history * 2:]
            
            messages = []
            for item in self.conversation_history:
                if image_pil is not None and len(messages) == 0:
                    messages.append({
                        "role": item["role"],
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": item["content"]}
                        ]
                    })
                else:
                    messages.append({
                        "role": item["role"],
                        "content": [
                            {"type": "text", "text": item["content"]}
                        ]
                    })
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            if image_pil is not None:
                inputs = self.processor(text=[text], images=[image_pil], return_tensors="pt").to(self.device)
            else:
                inputs = self.processor(text=[text], return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                response = self.model.generate(**inputs, max_new_tokens=512, temperature=0.7)
            
            full_response = self.processor.decode(response[0], skip_special_tokens=True)
            answer = full_response.split("assistant\n")[-1].strip()
            if not answer or answer.startswith("system"):
                answer = full_response.strip()
            
            self.conversation_history.append({
                "role": "assistant",
                "content": answer
            })
            
            return answer
        
        except Exception as e:
            print(f"Chat error: {e}")
            import traceback
            traceback.print_exc()
            return f"对话失败: {str(e)}"
    
    def clear_history(self):
        self.conversation_history = []
        return "对话历史已清空"

vlm_processor = VLMProcessor()