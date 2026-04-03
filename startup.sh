#!/bin/bash
# Azure Web App startup script
# Runs automatically when the app starts

echo "=== OxBlue Site Analyzer — Azure Web App Startup ==="

# Download YOLO models if not already present (first deploy)
# After first run they persist in the app directory
python3 -c "
from ultralytics import YOLO
from pathlib import Path
import urllib.request

# Standard COCO model
if not Path('yolov8n.pt').exists():
    print('Downloading yolov8n.pt...')
    YOLO('yolov8n.pt')
    print('Done.')
else:
    print('yolov8n.pt already present.')

# Hard hat safety model
hh = Path('hard_hat_yolov8n.pt')
if not hh.exists():
    print('Downloading hard hat model...')
    url = 'https://huggingface.co/keremberke/yolov8n-hard-hat-detection/resolve/main/best.pt'
    urllib.request.urlretrieve(url, str(hh))
    print(f'Done. ({hh.stat().st_size // 1024} KB)')
else:
    print('hard_hat_yolov8n.pt already present.')
"

echo "=== Starting gunicorn ==="
gunicorn \
  --bind=0.0.0.0:8000 \
  --timeout=600 \
  --workers=1 \
  --threads=4 \
  --worker-class=gthread \
  server:app
