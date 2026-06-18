import cv2
import os
import numpy as np

def create_test_video(output_path: str = "./videos/test.mp4", duration: int = 10):
    os.makedirs("./videos", exist_ok=True)
    
    width, height = 1920, 1080
    fps = 30
    total_frames = duration * fps
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for i in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        cv2.putText(frame, f"Test Video Frame {i}", (100, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        cv2.putText(frame, "Smart Video AI System", (100, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        
        center_x = int(width / 2 + 200 * np.sin(i * 0.05))
        center_y = int(height / 2 + 100 * np.cos(i * 0.03))
        
        cv2.circle(frame, (center_x, center_y), 50, (0, 0, 255), -1)
        cv2.putText(frame, "Person", (center_x - 30, center_y + 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.rectangle(frame, (100, 800), (300, 950), (255, 0, 0), -1)
        cv2.putText(frame, "Car", (150, 880), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.rectangle(frame, (1500, 500), (1700, 700), (0, 255, 255), -1)
        cv2.putText(frame, "Fire", (1550, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        out.write(frame)
    
    out.release()
    print(f"测试视频已生成: {output_path}")

if __name__ == "__main__":
    create_test_video()