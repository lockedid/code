import os
import requests
from tqdm import tqdm

def download_file(url, output_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(output_path)) as pbar:
            for data in response.iter_content(1024*1024):
                f.write(data)
                pbar.update(len(data))

os.makedirs("./models", exist_ok=True)

models = [
    ("yolo11m.pt", "https://hf-mirror.com/ultralytics/yolo11/resolve/main/yolo11m.pt"),
    ("yolo11s.pt", "https://hf-mirror.com/ultralytics/yolo11/resolve/main/yolo11s.pt"),
    ("yolo11m-pose.pt", "https://hf-mirror.com/ultralytics/yolo11/resolve/main/yolo11m-pose.pt")
]

for name, url in models:
    print(f"\n下载 {name}...")
    download_file(url, f"./models/{name}")
    size = os.path.getsize(f"./models/{name}") / (1024*1024)
    print(f"✓ 完成，大小: {size:.2f} MB")
