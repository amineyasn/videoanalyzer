"""
OxBlue Site Analyzer — Python Backend (v4 — Safety + Delivery Detection)
=========================================================================
PIPELINE PER FRAME:

  Stage 1 — YOLO Safety Model
            Hardhat / NO-Hardhat / Safety Vest / Person etc.

  Stage 2 — Proximity Check
            Person with no hardhat nearby → suspicious → Claude PPE review

  Stage 3 — Claude Vision: PPE (only on flagged frames)
            Distinguishes beanie / hard hat / baseball cap / no hat

  Stage 4 — Claude Vision: Delivery (only when truck detected)
            Identifies material on/around truck: rebar, steel beam,
            lumber, concrete, pipe, etc.
            Stitches consecutive truck frames into delivery EVENTS.

Usage:
    pip install flask flask-cors ultralytics opencv-python anthropic
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

app = Flask(__name__, static_folder=str(Path(__file__).parent))
CORS(app)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

SAFETY_MODEL_URL  = (
    "https://huggingface.co/keremberke/yolov8n-hard-hat-detection"
    "/resolve/main/best.pt"
)
# On Azure Web App, use a persistent path within the app directory
# Models are bundled in the deployment zip (see README)
BASE_DIR          = Path(__file__).parent
SAFETY_MODEL_PATH = BASE_DIR / "hard_hat_yolov8n.pt"

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
claude_client     = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

# Seconds of no truck before we consider a delivery event closed
DELIVERY_GAP_SECONDS = 30

# COCO truck-like classes that can carry material
TRUCK_CLASSES = {"truck", "bus", "car"}  # car included for pickups

# How much to expand the truck crop for context (pixels)
CROP_PADDING = 120

# ─────────────────────────────────────────────────────────────────────────────
# Claude prompts
# ─────────────────────────────────────────────────────────────────────────────

CLAUDE_PPE_PROMPT = """You are a construction site safety inspector.
Look at each person and determine their head protection precisely.

Distinguish: hard_hat (rigid helmet) / beanie (soft knit) /
baseball_cap (soft, brim) / bump_cap (thin insert, not protective) /
no_hat / other_hat / unclear

Return ONLY valid JSON, no markdown:
{
  "people": [
    {
      "head_protection": "hard_hat|beanie|baseball_cap|bump_cap|no_hat|other_hat|unclear",
      "is_compliant": true|false,
      "safety_vest": true|false,
      "notes": "e.g. yellow hard hat, red beanie"
    }
  ],
  "frame_assessment": "one sentence",
  "violation_confirmed": true|false
}"""

CLAUDE_DELIVERY_PROMPT = """You are an AI material inspector for a construction site camera system.

Analyze this image carefully. A vehicle (truck, flatbed, or similar) has been detected.

Determine:
1. What material (if any) is being delivered or is visible on/around the vehicle?
2. Is active unloading happening (workers, crane, forklift around vehicle)?
3. How confident are you?

Be very specific about steel types:
- rebar / rebar bundle (long ridged orange-brown steel rods, usually bundled)
- steel_beam / i_beam (grey structural H or I shaped sections)
- steel_plate (flat sheets of steel)
- steel_coil (large circular rolled steel)
- steel_pipe (tubular steel sections)
- steel_other (other steel forms)

Also identify: lumber / plywood / concrete / concrete_blocks /
drywall / conduit / equipment / empty_truck / unclear

Return ONLY valid JSON, no markdown:
{
  "vehicle_type": "flatbed|dump_truck|concrete_mixer|pickup|semi|other|unclear",
  "material_detected": true|false,
  "materials": [
    {
      "type": "rebar|steel_beam|steel_plate|steel_coil|steel_pipe|steel_other|lumber|plywood|concrete|concrete_blocks|drywall|conduit|equipment|empty_truck|unclear",
      "confidence": "high|medium|low",
      "description": "brief visual description e.g. bundle of orange rebar approx 20ft long"
    }
  ],
  "unloading_in_progress": true|false,
  "workers_present": true|false,
  "delivery_assessment": "one sentence summary",
  "is_steel_delivery": true|false
}"""

# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def download_safety_model() -> bool:
    if SAFETY_MODEL_PATH.exists():
        print(f"[server] Safety model cached — skipping download.")
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
    print(f"[server] Claude Vision ENABLED")
else:
    print(f"[server] Claude Vision DISABLED — set ANTHROPIC_API_KEY to enable")
print()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

VIOLATION_CLASSES = {
    "NO-Hardhat", "NO-Safety Vest", "no hardhat",
    "no-hardhat", "no_hardhat", "no safety vest", "no-safety vest",
}
SAFE_PPE_CLASSES = {
    "Hardhat", "hardhat", "hard hat", "Safety Vest", "safety vest",
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
STEEL_TYPES = {
    "rebar", "steel_beam", "i_beam", "steel_plate",
    "steel_coil", "steel_pipe", "steel_other",
}


def classify_det(cls: str) -> str:
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


def compute_risk(detections: list) -> str:
    cats = {d["category"] for d in detections}
    if "violation" in cats:
        return "high"
    if "suspicious" in cats:
        return "medium"
    if "person" in cats and "equipment" in cats:
        return "medium"
    if detections:
        return "low"
    return "none"


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


def head_region(bbox: dict) -> dict:
    h = bbox["y2"] - bbox["y1"]
    return {**bbox, "y2": bbox["y1"] + int(h * 0.30)}


def frame_to_b64(frame_bgr, quality=82) -> str:
    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("utf-8")


def crop_with_padding(frame, bbox: dict, padding: int = CROP_PADDING) -> any:
    h, w = frame.shape[:2]
    x1 = max(0, bbox["x1"] - padding)
    y1 = max(0, bbox["y1"] - padding)
    x2 = min(w, bbox["x2"] + padding)
    y2 = min(h, bbox["y2"] + padding)
    return frame[y1:y2, x1:x2]


# ─────────────────────────────────────────────────────────────────────────────
# Claude calls
# ─────────────────────────────────────────────────────────────────────────────

def claude_call(b64_image: str, prompt: str, max_tokens=700) -> dict | None:
    if not claude_client:
        return None
    try:
        resp = claude_client.messages.create(
            model="claude-opus-4-5",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg", "data": b64_image}},
                {"type": "text", "text": prompt}
            ]}]
        )
        raw = resp.content[0].text.strip().replace("```json","").replace("```","").strip()
        return json.loads(raw)
    except Exception as e:
        print(f"[server] Claude error: {e}")
        return None


def claude_ppe_check(frame_bgr, person_boxes: list) -> dict | None:
    if not person_boxes:
        return None
    x1 = max(0, min(b["x1"] for b in person_boxes) - 20)
    y1 = max(0, min(b["y1"] for b in person_boxes) - 20)
    x2 = min(frame_bgr.shape[1], max(b["x2"] for b in person_boxes) + 20)
    y2 = min(frame_bgr.shape[0], max(b["y2"] for b in person_boxes) + 20)
    crop = frame_bgr[y1:y2, x1:x2]
    return claude_call(frame_to_b64(crop), CLAUDE_PPE_PROMPT)


def claude_delivery_check(frame_bgr, truck_bbox: dict) -> dict | None:
    """Crop around the truck with generous padding and ask Claude what's being delivered."""
    crop = crop_with_padding(frame_bgr, truck_bbox, padding=CROP_PADDING)
    return claude_call(frame_to_b64(crop), CLAUDE_DELIVERY_PROMPT, max_tokens=800)


# ─────────────────────────────────────────────────────────────────────────────
# Delivery event tracker
# ─────────────────────────────────────────────────────────────────────────────

class DeliveryTracker:
    """
    Stitches individual truck-detection frames into delivery events.
    A new event opens when a truck appears.
    An event closes when no truck is seen for DELIVERY_GAP_SECONDS.
    """

    def __init__(self, gap_seconds=DELIVERY_GAP_SECONDS):
        self.gap   = gap_seconds
        self.events: list[dict] = []
        self._open: dict | None = None   # currently open event

    def truck_seen(self, ts_sec: float, ts_str: str, claude_result: dict | None):
        """Call this every time a truck frame is processed."""
        if self._open is None:
            # Start new event
            self._open = {
                "event_id":        len(self.events) + 1,
                "arrival_time":    ts_str,
                "arrival_sec":     ts_sec,
                "last_seen_sec":   ts_sec,
                "departure_time":  None,
                "duration_sec":    None,
                "frames_analyzed": 0,
                "material_votes":  defaultdict(int),   # material_type → count
                "steel_votes":     0,
                "vehicle_types":   defaultdict(int),
                "unloading_seen":  False,
                "workers_seen":    False,
                "frame_results":   [],
            }

        ev = self._open
        ev["last_seen_sec"] = ts_sec
        ev["frames_analyzed"] += 1

        if claude_result:
            ev["frame_results"].append({
                "timestamp": ts_str,
                "result":    claude_result,
            })
            for mat in claude_result.get("materials", []):
                ev["material_votes"][mat["type"]] += 1
            if claude_result.get("is_steel_delivery"):
                ev["steel_votes"] += 1
            if claude_result.get("vehicle_type"):
                ev["vehicle_types"][claude_result["vehicle_type"]] += 1
            if claude_result.get("unloading_in_progress"):
                ev["unloading_seen"] = True
            if claude_result.get("workers_present"):
                ev["workers_seen"] = True

    def tick(self, ts_sec: float):
        """Call on every frame (even non-truck) to check if open event should close."""
        if self._open and (ts_sec - self._open["last_seen_sec"]) > self.gap:
            self._close_event(ts_sec)

    def finish(self, ts_sec: float):
        """Call at end of video to close any still-open event."""
        if self._open:
            self._close_event(ts_sec)

    def _close_event(self, ts_sec: float):
        ev = self._open
        ev["departure_time"] = fmt_ts(ev["last_seen_sec"])
        ev["duration_sec"]   = round(ev["last_seen_sec"] - ev["arrival_sec"])

        # Summarise material votes → ranked list
        mat_votes = dict(ev["material_votes"])
        ev["top_materials"] = sorted(mat_votes.items(), key=lambda x: -x[1])

        # Determine primary material
        if ev["top_materials"]:
            primary = ev["top_materials"][0][0]
        else:
            primary = "unclear"
        ev["primary_material"] = primary
        ev["is_steel_delivery"] = (
            primary in STEEL_TYPES or
            any(m in STEEL_TYPES for m, _ in ev["top_materials"]) or
            ev["steel_votes"] > 0
        )

        # Most-voted vehicle type
        vt = dict(ev["vehicle_types"])
        ev["vehicle_type"] = max(vt, key=vt.get) if vt else "unknown"

        # Clean up non-serialisable defaultdicts
        ev["material_votes"] = mat_votes
        ev["vehicle_types"]  = vt

        self.events.append(ev)
        self._open = None

    def serialisable_events(self) -> list:
        """Return events ready for JSON serialisation."""
        result = []
        for ev in self.events:
            e = dict(ev)
            e["top_materials"] = [{"material": m, "frames": c}
                                   for m, c in ev.get("top_materials", [])]
            result.append(e)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Core analysis generator
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
    delivery_tracker = DeliveryTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % frame_interval != 0:
            continue

        sampled   += 1
        ts_sec     = frame_idx / native_fps
        ts_str     = fmt_ts(ts_sec)

        detections = []
        seen_boxes = []
        claude_ppe_result      = None
        claude_delivery_result = None

        # ── Stage 1: Safety YOLO ──────────────────────────────────────────────
        if safety_model:
            for box in safety_model(frame, conf=confidence, verbose=False)[0].boxes:
                cls  = safety_model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                bbox = {"x1":x1,"y1":y1,"x2":x2,"y2":y2}
                detections.append({
                    "class":cls,"confidence":round(conf,3),
                    "bbox":bbox,"category":classify_det(cls),"source":"yolo_safety"
                })
                seen_boxes.append(bbox)
                object_tally[cls] += 1

        # COCO pass — vehicles + equipment
        safety_has_people = any(
            d["category"] in ("person","violation","ppe") for d in detections
        )
        truck_bboxes = []
        for box in coco_model(frame, conf=confidence, verbose=False)[0].boxes:
            cls  = coco_model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            if cls == "person" and safety_has_people:
                continue
            if cls not in COCO_RELEVANT and cls not in EQUIPMENT_CLASSES:
                continue
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            bbox = {"x1":x1,"y1":y1,"x2":x2,"y2":y2}
            if any(box_overlaps(bbox, sb) for sb in seen_boxes):
                continue
            detections.append({
                "class":cls,"confidence":round(conf,3),
                "bbox":bbox,"category":classify_det(cls),"source":"yolo_coco"
            })
            seen_boxes.append(bbox)
            object_tally[cls] += 1
            if cls in TRUCK_CLASSES:
                truck_bboxes.append(bbox)

        # ── Stage 2: Proximity check (PPE) ────────────────────────────────────
        person_dets  = [d for d in detections if d["category"] == "person"]
        hardhat_dets = [d for d in detections if d["class"] in ("Hardhat","hardhat","hard hat")]
        unconfirmed_persons = []
        for person in person_dets:
            head = head_region(person["bbox"])
            if not any(box_overlaps(head, h["bbox"], thr=0.15) for h in hardhat_dets):
                unconfirmed_persons.append(person)
                detections.append({
                    "class":"Unconfirmed-No-Hardhat","confidence":round(person["confidence"],3),
                    "bbox":person["bbox"],"category":"suspicious","source":"proximity_check",
                    "note":"Person with no hard hat in head region"
                })

        # ── Stage 3: Claude PPE check ─────────────────────────────────────────
        if use_claude and claude_client and unconfirmed_persons:
            claude_ppe_result = claude_ppe_check(frame, [p["bbox"] for p in unconfirmed_persons])
            if claude_ppe_result:
                detections = [d for d in detections if d["category"] != "suspicious"]
                for i, pa in enumerate(claude_ppe_result.get("people", [])):
                    hat_type  = pa.get("head_protection","unclear")
                    compliant = pa.get("is_compliant", False)
                    cat   = "violation" if hat_type in NON_HARDHAT_TYPES else \
                            "ppe"       if hat_type == "hard_hat" else "suspicious"
                    label = f"NO-Hardhat ({hat_type})" if hat_type in NON_HARDHAT_TYPES else \
                            "Hardhat (confirmed)"       if hat_type == "hard_hat" else \
                            f"Unconfirmed ({hat_type})"
                    bbox = unconfirmed_persons[i]["bbox"] if i < len(unconfirmed_persons) \
                           else unconfirmed_persons[0]["bbox"]
                    detections.append({
                        "class":label,"confidence":None,"bbox":bbox,
                        "category":cat,"source":"claude_ppe",
                        "hat_type":hat_type,"vest":pa.get("safety_vest"),
                        "note":pa.get("notes","")
                    })
                    if cat == "violation":
                        object_tally[label] = object_tally.get(label, 0) + 1

        # ── Stage 4: Claude Delivery check ────────────────────────────────────
        if use_claude and claude_client and truck_bboxes:
            # Use the largest truck bounding box for the crop
            largest_truck = max(
                truck_bboxes,
                key=lambda b: (b["x2"]-b["x1"]) * (b["y2"]-b["y1"])
            )
            claude_delivery_result = claude_delivery_check(frame, largest_truck)

            if claude_delivery_result:
                # Annotate the truck detection with delivery info
                for d in detections:
                    if d["class"] in TRUCK_CLASSES:
                        d["delivery"] = claude_delivery_result

                # If material found, add explicit detections for the UI
                for mat in claude_delivery_result.get("materials", []):
                    if mat["type"] != "empty_truck" and mat["type"] != "unclear":
                        is_steel = mat["type"] in STEEL_TYPES
                        detections.append({
                            "class":       mat["type"].replace("_"," ").title(),
                            "confidence":  None,
                            "bbox":        largest_truck,  # approximate to truck location
                            "category":    "steel_delivery" if is_steel else "delivery",
                            "source":      "claude_delivery",
                            "description": mat.get("description",""),
                            "confidence_level": mat.get("confidence",""),
                        })
                        object_tally[mat["type"]] = object_tally.get(mat["type"],0) + 1

            delivery_tracker.truck_seen(ts_sec, ts_str, claude_delivery_result)

        elif truck_bboxes:
            # Truck seen but Claude not available — still track event
            delivery_tracker.truck_seen(ts_sec, ts_str, None)

        # Tick the tracker on every frame (closes stale events)
        delivery_tracker.tick(ts_sec)

        # ── Risk scoring ──────────────────────────────────────────────────────
        cats = {d["category"] for d in detections}
        if "violation" in cats:
            risk = "high"; violation_frames += 1
        elif "suspicious" in cats:
            risk = "medium"
        elif "person" in cats and "equipment" in cats:
            risk = "medium"
        elif detections:
            risk = "low"
        else:
            risk = "none"

        yield event({
            "type":             "frame",
            "frame":            frame_idx,
            "timestamp":        ts_str,
            "timestamp_sec":    round(ts_sec, 2),
            "sampled":          sampled,
            "total":            frames_to_analyze,
            "detections":       detections,
            "count":            len(detections),
            "risk":             risk,
            "violations":       [d for d in detections if d["category"]=="violation"],
            "suspicious":       [d for d in detections if d["category"]=="suspicious"],
            "ppe_compliant":    [d for d in detections if d["category"]=="ppe"],
            "people":           [d for d in detections if d["category"]=="person"],
            "equipment":        [d for d in detections if d["category"]=="equipment"],
            "deliveries":       [d for d in detections if d["category"] in ("delivery","steel_delivery")],
            "other":            [d for d in detections if d["category"]=="other"],
            "truck_detected":   len(truck_bboxes) > 0,
            "claude_ppe_used":      bool(claude_ppe_result),
            "claude_delivery_used": bool(claude_delivery_result),
            "claude_ppe_result":      claude_ppe_result,
            "claude_delivery_result": claude_delivery_result,
        })

    cap.release()
    delivery_tracker.finish(duration_sec)

    total_violations = sum(
        v for k, v in object_tally.items()
        if k in VIOLATION_CLASSES or "NO-Hardhat" in k
        or k.lower() in {x.lower() for x in VIOLATION_CLASSES}
    )
    delivery_events = delivery_tracker.serialisable_events()
    steel_deliveries = [e for e in delivery_events if e.get("is_steel_delivery")]

    yield event({
        "type":              "done",
        "frames_analyzed":   sampled,
        "violation_frames":  violation_frames,
        "total_violations":  total_violations,
        "delivery_events":   delivery_events,
        "steel_deliveries":  len(steel_deliveries),
        "object_tally":      {k:v for k,v in
                              sorted(object_tally.items(), key=lambda x:-x[1])
                              if not k.startswith("__")},
    })


# ─────────────────────────────────────────────────────────────────────────────
# Flask routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(str(Path(__file__).parent), "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    """Serve any static file — logo, etc. Block path traversal."""
    safe_root = Path(__file__).parent.resolve()
    target    = (safe_root / filename).resolve()
    if not str(target).startswith(str(safe_root)):
        return "Forbidden", 403
    if filename in ("analyze", "status"):
        return "Not found", 404
    return send_from_directory(str(safe_root), filename)


@app.route("/status")
def status():
    return jsonify({
        "safety_model":  safety_model is not None,
        "claude_enabled": claude_client is not None,
        "safety_classes": list(safety_model.names.values()) if safety_model else [],
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files:
        return jsonify({"error": "No video file"}), 400
    video_file = request.files["video"]
    sample_fps = float(request.form.get("fps", 1.0))
    confidence = float(request.form.get("confidence", 0.35))
    use_claude = request.form.get("use_claude","true").lower() == "true"
    suffix = Path(video_file.filename).suffix or ".mp4"
    tmp    = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    video_file.save(tmp.name)
    tmp.close()

    def stream_and_cleanup():
        try:
            yield from analyze_video(tmp.name, sample_fps, confidence, use_claude)
        finally:
            try: os.unlink(tmp.name)
            except: pass

    return Response(
        stream_and_cleanup(),
        mimetype="text/event-stream",
        headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"},
    )


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  OxBlue Site Analyzer v4 — Safety + Delivery Detection")
    print("  Local: http://localhost:5000")
    print("="*60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

# Azure Web App / gunicorn entry point — module-level app object is used automatically
