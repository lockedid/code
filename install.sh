#!/bin/bash

# 配置conda源
cat > ~/.condarc << EOF
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF

# 配置pip源
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << EOF
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF

# 创建并激活环境

conda activate video_ai

# 安装依赖
echo "安装基础依赖..."
pip install numpy==1.26.4 opencv-python==4.10.0.84

echo "安装PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "安装YOLO..."
pip install ultralytics==8.3.26

echo "安装API依赖..."
pip install fastapi==0.110.2 uvicorn==0.27.1 psycopg2-binary==2.9.9

echo "安装大模型依赖..."
pip install transformers==4.40.2 accelerate==0.29.3 sentencepiece==0.1.99 pillow==10.3.0

echo "安装完成！"