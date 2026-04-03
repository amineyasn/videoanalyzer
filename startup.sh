#!/bin/bash
# Azure Web App startup script

echo "=== OxBlue Site Analyzer — Azure Web App Startup ==="

# Install missing system libraries that opencv needs on Azure Linux
echo "Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq \
  libxcb1 \
  libglib2.0-0 \
  libsm6 \
  libxrender1 \
  libxext6 \
  libgl1 \
  2>/dev/null || echo "apt-get failed — may already be installed"

echo "System dependencies done."

# Download YOLO models if not already present
cd /home/site/wwwroot
python3 -c "
from pathlib import Path
import urllib.request

# Standard COCO model
if not Path('yolov8n.pt').exists():
    print('Downloading yolov8n.pt...')
    from ultralytics import YOLO
    YOLO('yolov8n.pt')
    print('yolov8n.pt ready.')
else:
    print('yolov8n.pt already present.')

# Hard hat safety model
hh = Path('hard_hat_yolov8n.pt')
if not hh.exists():
    print('Downloading hard hat model...')
    url = 'https://huggingface.co/keremberke/yolov8n-hard-hat-detection/resolve/main/best.pt'
    urllib.request.urlretrieve(url, str(hh))
    print(f'hard_hat_yolov8n.pt ready ({hh.stat().st_size // 1024} KB).')
else:
    print('hard_hat_yolov8n.pt already present.')
"

echo "=== Starting gunicorn ==="
cd /home/site/wwwroot
gunicorn \
  --bind=0.0.0.0:8000 \
  --timeout=600 \
  --workers=1 \
  --threads=4 \
  --worker-class=gthread \
  server:app
