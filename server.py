"""
OxBlue Site Analyzer — Python Backend
=======================================
Runs a local web server that accepts a video upload,
samples frames at 1fps with OpenCV, runs YOLOv8 on each frame,
and streams results back to the browser as Server-Sent Events (SSE).

Usage:
    pip install flask flask-cors ultralytics opencv-python
    python server.py

Then open:  http://localhost:5000
"""

import cv2
import json
import os
import tempfile
from pathlib import Path
from collections import defaultdict

from flask import Flask, request, Response, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__, static_folder=".")
CORS(app)

# ── Load YOLO model once at startup ──────────────────────────────────────────
# Swap to yolov8s.pt / yolov8m.pt for higher accuracy
print("[server] Loading YOLOv8 model...")
model = YOLO("yolov8n.pt")
print(f"[server] Model ready — {len(model.names)} classes")

CONSTRUCTION_CLASSES = {
    "person", "car", "truck", "bus", "motorcycle", "bicycle",
    "backpack", "umbrella", "traffic light", "stop sign",
    "hard hat", "safety vest", "crane", "excavator"
}

def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def analyze_video(video_path: str, sample_fps: float = 1.0, confidence: float = 0.35):
    """
    Generator that yields Server-Sent Events (SSE) as each frame is analyzed.
    The browser receives results in real time as they're processed.
    """

    def event(data: dict) -> str:
        return f"data: {json.dumps(data)}\n\n"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        yield event({"type": "error", "message": "Could not open video file"})
        return

    native_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / native_fps
    frame_interval = max(1, int(round(native_fps / sample_fps)))
    frames_to_analyze = total_frames // frame_interval

    # Send video metadata first
    yield event({
        "type": "meta",
        "duration": round(duration_sec, 1),
        "native_fps": round(native_fps, 1),
        "total_frames": total_frames,
        "frames_to_analyze": frames_to_analyze,
        "sample_fps": sample_fps
    })

    frame_idx    = 0
    sampled      = 0
    object_tally = defaultdict(int)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % frame_interval != 0:
            continue

        sampled += 1
        timestamp_sec = frame_idx / native_fps
        timestamp_str = format_timestamp(timestamp_sec)

        # ── Run YOLO ──────────────────────────────────────────────────────────
        results    = model(frame, conf=confidence, verbose=False)[0]
        detections = []

        for box in results.boxes:
            class_id   = int(box.cls[0])
            class_name = model.names[class_id]
            conf_score = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

            detections.append({
                "class":      class_name,
                "confidence": round(conf_score, 3),
                "bbox":       {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "construction_relevant": class_name in CONSTRUCTION_CLASSES
            })
            object_tally[class_name] += 1

        # ── Stream this frame's results to the browser ────────────────────────
        yield event({
            "type":       "frame",
            "frame":      frame_idx,
            "timestamp":  timestamp_str,
            "timestamp_sec": round(timestamp_sec, 2),
            "sampled":    sampled,
            "total":      frames_to_analyze,
            "detections": detections,
            "count":      len(detections)
        })

    cap.release()

    # ── Final summary ─────────────────────────────────────────────────────────
    yield event({
        "type":         "done",
        "frames_analyzed": sampled,
        "object_tally": dict(sorted(object_tally.items(), key=lambda x: -x[1])),
        "construction_relevant": {
            k: v for k, v in object_tally.items()
            if k in CONSTRUCTION_CLASSES
        }
    })


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files:
        return jsonify({"error": "No video file"}), 400

    video_file  = request.files["video"]
    sample_fps  = float(request.form.get("fps", 1.0))
    confidence  = float(request.form.get("confidence", 0.35))

    # Save to temp file
    suffix = Path(video_file.filename).suffix or ".mp4"
    tmp    = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    video_file.save(tmp.name)
    tmp.close()

    def stream_and_cleanup():
        try:
            yield from analyze_video(tmp.name, sample_fps, confidence)
        finally:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass

    return Response(
        stream_and_cleanup(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no"   # needed for nginx proxies
        }
    )


if __name__ == "__main__":
    print("\n" + "="*50)
    print("  OxBlue Site Analyzer")
    print("  Open http://localhost:5000 in your browser")
    print("="*50 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
