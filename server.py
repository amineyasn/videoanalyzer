"""
OxBlue Site Analyzer — Python Backend (v3 — Hybrid YOLO + Claude Vision)
=========================================================================
THREE-STAGE PIPELINE per frame:

  Stage 1 — YOLOv8 Safety Model
            Fast detection of Hardhat / NO-Hardhat / Safety Vest / Person etc.

  Stage 2 — Proximity Check
            If a Person is detected but no Hardhat bounding box overlaps
            their head region → flagged as "unconfirmed / suspicious"

  Stage 3 — Claude Vision (optional, requires ANTHROPIC_API_KEY env var)
            Sends the cropped person region to Claude to distinguish:
            hard hat / beanie / baseball cap / bump cap / no hat
            Only fires on frames where YOLO was uncertain.
            This is the "tiered" cost-control approach.

Usage:
    pip install flask flask-cors ultralytics opencv-python requests anthropic

    # Optional — enables Claude Vision second pass:
    export ANTHROPIC_API_KEY=sk-ant-...

    python server.py
    open http://localhost:5000
"""

import base64
import cv2
import json
import os
import tempfile
import urllib.request
from pathlib import Path
from collections import defaultdict

import anthropic
from flask import Flask, request, Response, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__, static_folder=".")
CORS(app)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

SAFETY_MODEL_URL  = (
    "https://huggingface.co/keremberke/yolov8n-hard-hat-detection"
    "/resolve/main/best.pt"
)
SAFETY_MODEL_PATH = Path("hard_hat_yolov8n.pt")

# Claude Vision second-pass — only fires if API key is set
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
claude_client     = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

CLAUDE_PPE_PROMPT = """You are a construction site safety inspector reviewing a video frame.
Look carefully at each person visible and determine exactly what head protection (if any) they are wearing.

Be very specific — distinguish between:
- "hard_hat": a rigid protective helmet (plastic/fiberglass, typically yellow, white, orange, blue)
- "beanie": a soft knit/wool cap, no rigid shell
- "baseball_cap": a soft cap with a brim/peak, no rigid shell
- "bump_cap": a lightweight soft-shell cap with thin insert, NOT a hard hat
- "no_hat": bare head, no head covering at all
- "other_hat": any other non-protective headwear
- "unclear": cannot determine from this frame

For each person detected, also note if they are wearing a safety vest (high-visibility yellow/orange).

Return ONLY valid JSON, no markdown:
{
  "people": [
    {
      "head_protection": "hard_hat|beanie|baseball_cap|bump_cap|no_hat|other_hat|unclear",
      "is_compliant": true|false,
      "safety_vest": true|false|"unclear",
      "notes": "brief description e.g. yellow hard hat, white beanie, no head protection"
    }
  ],
  "frame_assessment": "one sentence summary",
  "violation_confirmed": true|false
}"""


# ─────────────────────────────────────────────────────────────────────────────
# Download safety model
# ─────────────────────────────────────────────────────────────────────────────

def download_safety_model() -> bool:
    if SAFETY_MODEL_PATH.exists():
        size_kb = SAFETY_MODEL_PATH.stat().st_size // 1024
        print(f"[server] Safety model cached ({size_kb} KB) — skipping download.")
        return True
    print(f"[server] Downloading safety model (~6 MB)…")
    try:
        def _prog(n, bs, total):
            if total > 0:
                print(f"\r[server]   {min(n*bs/total*100,100):.1f}%", end="", flush=True)
        urllib.request.urlretrieve(SAFETY_MODEL_URL, SAFETY_MODEL_PATH, _prog)
        print(f"\n[server] Saved → {SAFETY_MODEL_PATH}")
        return True
    except Exception as e:
        print(f"\n[server] Download failed: {e}")
        if SAFETY_MODEL_PATH.exists():
            SAFETY_MODEL_PATH.unlink()
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────────────────────────────────────

safety_model = None
if download_safety_model():
    try:
        safety_model = YOLO(str(SAFETY_MODEL_PATH))
        print(f"[server] Safety model ready — {list(safety_model.names.values())}")
    except Exception as e:
        print(f"[server] Safety model load error: {e}")

coco_model = YOLO("yolov8n.pt")
print(f"[server] COCO model ready — {len(coco_model.names)} classes")

if claude_client:
    print(f"[server] Claude Vision second-pass ENABLED (ANTHROPIC_API_KEY found)")
else:
    print(f"[server] Claude Vision second-pass DISABLED (set ANTHROPIC_API_KEY to enable)")
print()


# ─────────────────────────────────────────────────────────────────────────────
# Class helpers
# ─────────────────────────────────────────────────────────────────────────────

VIOLATION_CLASSES = {
    "NO-Hardhat", "NO-Safety Vest",
    "no hardhat", "no-hardhat", "no_hardhat",
    "no safety vest", "no-safety vest",
}
SAFE_PPE_CLASSES = {
    "Hardhat", "hardhat", "hard hat",
    "Safety Vest", "safety vest",
}
EQUIPMENT_CLASSES = {
    "truck", "car", "bus", "motorcycle", "bicycle",
    "machinery", "vehicle", "crane", "excavator", "forklift",
    "Safety Cone", "safety cone",
}
COCO_RELEVANT = {
    "truck", "car", "bus", "motorcycle", "bicycle",
    "person", "backpack", "traffic light", "stop sign",
}

NON_HARDHAT_TYPES = {"beanie", "baseball_cap", "bump_cap", "other_hat", "no_hat"}


def classify_detection(cls: str) -> str:
    cn = cls.lower()
    if cls in VIOLATION_CLASSES or cn in {v.lower() for v in VIOLATION_CLASSES}:
        return "violation"
    if cls in SAFE_PPE_CLASSES or cn in {v.lower() for v in SAFE_PPE_CLASSES}:
        return "ppe"
    if cls in EQUIPMENT_CLASSES or cn in {v.lower() for v in EQUIPMENT_CLASSES}:
        return "equipment"
    if cn == "person":
        return "person"
    return "other"


def fmt_ts(sec: float) -> str:
    return f"{int(sec//3600):02d}:{int((sec%3600)//60):02d}:{int(sec%60):02d}"


def box_overlaps(b1, b2, thr=0.3) -> bool:
    ix1, iy1 = max(b1["x1"], b2["x1"]), max(b1["y1"], b2["y1"])
    ix2, iy2 = min(b1["x2"], b2["x2"]), min(b1["y2"], b2["y2"])
    if ix2 <= ix1 or iy2 <= iy1:
        return False
    inter = (ix2 - ix1) * (iy2 - iy1)
    a1 = (b1["x2"]-b1["x1"]) * (b1["y2"]-b1["y1"])
    a2 = (b2["x2"]-b2["x1"]) * (b2["y2"]-b2["y1"])
    union = a1 + a2 - inter
    return (inter / union) > thr if union > 0 else False


def head_region(person_box: dict) -> dict:
    """Return the top ~30% of a person bounding box (the head area)."""
    h = person_box["y2"] - person_box["y1"]
    return {
        "x1": person_box["x1"],
        "y1": person_box["y1"],
        "x2": person_box["x2"],
        "y2": person_box["y1"] + int(h * 0.30),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Claude Vision second pass
# ─────────────────────────────────────────────────────────────────────────────

def claude_ppe_check(frame_bgr, person_boxes: list) -> dict | None:
    """
    Crop the frame to show all detected people and ask Claude to assess PPE.
    Only called when YOLO is uncertain (person detected, hat status unclear).
    Returns the parsed JSON response or None on error.
    """
    if not claude_client or not person_boxes:
        return None

    # Crop to a bounding box that contains all detected people + 20px padding
    x1 = max(0, min(b["x1"] for b in person_boxes) - 20)
    y1 = max(0, min(b["y1"] for b in person_boxes) - 20)
    x2 = min(frame_bgr.shape[1], max(b["x2"] for b in person_boxes) + 20)
    y2 = min(frame_bgr.shape[0], max(b["y2"] for b in person_boxes) + 20)

    crop    = frame_bgr[y1:y2, x1:x2]
    _, buf  = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64     = base64.b64encode(buf).decode("utf-8")

    try:
        resp = claude_client.messages.create(
            model="claude-opus-4-5",
            max_tokens=600,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64",
                                                  "media_type": "image/jpeg",
                                                  "data": b64}},
                    {"type": "text", "text": CLAUDE_PPE_PROMPT}
                ]
            }]
        )
        raw = resp.content[0].text.strip()
        return json.loads(raw.replace("```json", "").replace("```", "").strip())
    except Exception as e:
        print(f"[server] Claude Vision error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline
# ─────────────────────────────────────────────────────────────────────────────

def analyze_video(video_path: str, sample_fps=1.0, confidence=0.35, use_claude=True):

    def event(data): return f"data: {json.dumps(data)}\n\n"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        yield event({"type": "error", "message": "Could not open video"})
        return

    native_fps        = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec      = total_frames / native_fps
    frame_interval    = max(1, int(round(native_fps / sample_fps)))
    frames_to_analyze = max(1, total_frames // frame_interval)

    yield event({
        "type":                 "meta",
        "duration":             round(duration_sec, 1),
        "native_fps":           round(native_fps, 1),
        "total_frames":         total_frames,
        "frames_to_analyze":    frames_to_analyze,
        "sample_fps":           sample_fps,
        "safety_model_active":  safety_model is not None,
        "claude_active":        bool(claude_client and use_claude),
    })

    frame_idx        = 0
    sampled          = 0
    object_tally     = defaultdict(int)
    violation_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % frame_interval != 0:
            continue

        sampled       += 1
        ts_sec         = frame_idx / native_fps
        ts_str         = fmt_ts(ts_sec)

        detections     = []
        seen_boxes     = []
        claude_result  = None

        # ── Stage 1: Safety YOLO ──────────────────────────────────────────────
        if safety_model:
            for box in safety_model(frame, conf=confidence, verbose=False)[0].boxes:
                cls  = safety_model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                bbox = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                cat  = classify_detection(cls)
                detections.append({
                    "class": cls, "confidence": round(conf, 3),
                    "bbox": bbox, "category": cat, "source": "yolo_safety",
                })
                seen_boxes.append(bbox)
                object_tally[cls] += 1

        # COCO vehicles/equipment
        safety_has_people = any(
            d["category"] in ("person","violation","ppe") for d in detections
        )
        for box in coco_model(frame, conf=confidence, verbose=False)[0].boxes:
            cls  = coco_model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            if cls == "person" and safety_has_people:
                continue
            if cls not in COCO_RELEVANT and cls not in EQUIPMENT_CLASSES:
                continue
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            bbox = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            if any(box_overlaps(bbox, sb) for sb in seen_boxes):
                continue
            detections.append({
                "class": cls, "confidence": round(conf, 3),
                "bbox": bbox, "category": classify_detection(cls),
                "source": "yolo_coco",
            })
            seen_boxes.append(bbox)
            object_tally[cls] += 1

        # ── Stage 2: Proximity check ──────────────────────────────────────────
        # Find persons without a Hardhat overlapping their head region
        person_dets  = [d for d in detections if d["category"] == "person"]
        hardhat_dets = [d for d in detections if d["class"] in ("Hardhat","hardhat","hard hat")]
        unconfirmed_persons = []

        for person in person_dets:
            head = head_region(person["bbox"])
            has_hat = any(box_overlaps(head, h["bbox"], thr=0.15) for h in hardhat_dets)
            if not has_hat:
                unconfirmed_persons.append(person)
                # Tentatively flag as suspicious (may be overridden by Stage 3)
                detections.append({
                    "class":      "Unconfirmed-No-Hardhat",
                    "confidence": round(person["confidence"], 3),
                    "bbox":       person["bbox"],
                    "category":   "suspicious",
                    "source":     "proximity_check",
                    "note":       "Person detected — no hard hat in head region. Awaiting Claude review."
                })

        # ── Stage 3: Claude Vision second pass ───────────────────────────────
        # Only triggered when: Claude is enabled AND there are unconfirmed persons
        needs_claude = bool(claude_client and use_claude and unconfirmed_persons)
        if needs_claude:
            person_boxes = [p["bbox"] for p in unconfirmed_persons]
            claude_result = claude_ppe_check(frame, person_boxes)

            if claude_result:
                # Remove the tentative "Unconfirmed" entries — replace with Claude's verdict
                detections = [d for d in detections if d["category"] != "suspicious"]

                for i, person_assessment in enumerate(claude_result.get("people", [])):
                    hat_type  = person_assessment.get("head_protection", "unclear")
                    compliant = person_assessment.get("is_compliant", False)
                    notes     = person_assessment.get("notes", "")
                    vest      = person_assessment.get("safety_vest", "unclear")

                    if hat_type in NON_HARDHAT_TYPES:
                        cat   = "violation"
                        label = f"NO-Hardhat ({hat_type})"
                        # Add to tally
                        object_tally[label] = object_tally.get(label, 0) + 1
                    elif hat_type == "hard_hat":
                        cat   = "ppe"
                        label = "Hardhat (confirmed)"
                    else:
                        cat   = "suspicious"
                        label = f"Unconfirmed-Headwear ({hat_type})"

                    # Use the corresponding person bbox if available
                    bbox = (unconfirmed_persons[i]["bbox"]
                            if i < len(unconfirmed_persons)
                            else unconfirmed_persons[0]["bbox"])

                    detections.append({
                        "class":      label,
                        "confidence": None,
                        "bbox":       bbox,
                        "category":   cat,
                        "source":     "claude_vision",
                        "hat_type":   hat_type,
                        "vest":       vest,
                        "note":       notes,
                    })

                # Also surface Claude's frame-level assessment
                if claude_result.get("frame_assessment"):
                    object_tally["__claude_assessments__"] = (
                        object_tally.get("__claude_assessments__", 0) + 1
                    )

        # ── Risk scoring ──────────────────────────────────────────────────────
        cats = {d["category"] for d in detections}
        if "violation" in cats:
            risk = "high"
            violation_frames += 1
        elif "suspicious" in cats:
            risk = "medium"       # proximity flagged, Claude not available
        elif "person" in cats and "equipment" in cats:
            risk = "medium"
        elif detections:
            risk = "low"
        else:
            risk = "none"

        yield event({
            "type":          "frame",
            "frame":         frame_idx,
            "timestamp":     ts_str,
            "timestamp_sec": round(ts_sec, 2),
            "sampled":       sampled,
            "total":         frames_to_analyze,
            "detections":    detections,
            "count":         len(detections),
            "risk":          risk,
            "violations":    [d for d in detections if d["category"] == "violation"],
            "suspicious":    [d for d in detections if d["category"] == "suspicious"],
            "ppe_compliant": [d for d in detections if d["category"] == "ppe"],
            "people":        [d for d in detections if d["category"] == "person"],
            "equipment":     [d for d in detections if d["category"] == "equipment"],
            "other":         [d for d in detections if d["category"] == "other"],
            "claude_used":   bool(claude_result),
            "claude_result": claude_result,
        })

    cap.release()

    total_violations = sum(
        v for k, v in object_tally.items()
        if k != "__claude_assessments__" and (
            k in VIOLATION_CLASSES
            or k.lower() in {x.lower() for x in VIOLATION_CLASSES}
            or "NO-Hardhat" in k
        )
    )

    yield event({
        "type":             "done",
        "frames_analyzed":  sampled,
        "violation_frames": violation_frames,
        "total_violations": total_violations,
        "object_tally":     {k: v for k, v in
                             sorted(object_tally.items(), key=lambda x: -x[1])
                             if not k.startswith("__")},
    })


# ─────────────────────────────────────────────────────────────────────────────
# Flask routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/status")
def status():
    return jsonify({
        "safety_model": safety_model is not None,
        "claude_enabled": claude_client is not None,
        "safety_classes": list(safety_model.names.values()) if safety_model else [],
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files:
        return jsonify({"error": "No video file"}), 400

    video_file  = request.files["video"]
    sample_fps  = float(request.form.get("fps", 1.0))
    confidence  = float(request.form.get("confidence", 0.35))
    use_claude  = request.form.get("use_claude", "true").lower() == "true"

    suffix = Path(video_file.filename).suffix or ".mp4"
    tmp    = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    video_file.save(tmp.name)
    tmp.close()

    def stream_and_cleanup():
        try:
            yield from analyze_video(tmp.name, sample_fps, confidence, use_claude)
        finally:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass

    return Response(
        stream_and_cleanup(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  OxBlue Site Analyzer v3 — Hybrid YOLO + Claude Vision")
    print("  Open http://localhost:5000")
    print("="*60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
