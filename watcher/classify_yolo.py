"""Classify motion events on-device with yolo11n instead of the ollama VLM.

Replaces the llama3.2-vision ollama call (GH#23-related churn: every motion
event cold-loaded ~5 GB into VRAM, spiking GPU power 7W -> ~190W on the RTX
4070 Ti — root cause of "fan cycling" observed 2026-10-14). YOLO11n (5.6 MB)
detects person/vehicle/animal directly; classification becomes local inference
with no model-load churn.

Kept compatible with the existing pipeline:
- writes the same Labeling row shape (labels, description, probabilities),
- decider string stays under the 'ollama:' ML prefix handling in browser.py by
  using 'ollama:yolo11n' so ML-decider filters (/_ml_decider_predicate/)
  recognize it, but the model name makes on-device detection auditable
  where descriptions differ from VLM text.
- honours [classification] MODEL in watcher.cfg if it names a yolo*.pt/.onnx
  weights file (operator can swap weights without code change).

Falls back to the VLM path when weights are missing, so the worker never
hard-fails on a host without the weights file.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from PIL import Image

from . import setup_logging

logger = setup_logging()

DEFAULT_CLASSIFIER_WEIGHTS = "/data/video/watcher/stealthcam/yolo11n.pt"

# COCO person/vehicle/animal classes — keep in sync with stealthcam/detector.py
COCO_NAMES = {
    0: "person",
    1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 6: "train", 7: "truck",
    8: "boat", 9: "traffic light",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe",
}
CLASS_TO_CATEGORY = {
    "person": "person",
    "bicycle": "vehicle", "car": "vehicle", "motorcycle": "vehicle",
    "bus": "vehicle", "train": "vehicle", "truck": "vehicle", "boat": "vehicle",
    "bird": "animal", "cat": "animal", "dog": "animal", "horse": "animal",
    "sheep": "animal", "cow": "animal", "elephant": "animal", "bear": "animal",
    "zebra": "animal", "giraffe": "animal",
}


def _weights_path() -> Path:
    """Resolve detector weights: env WATCHER_CLASSIFICATION_YOLO_WEIGHTS,
    then [classification] YOLO_WEIGHTS in watcher.cfg, then the built-in
    default. (.pt/.onnx in MODEL is honoured as a legacy path override.)"""
    import os
    from .connection import application_config
    env = os.environ.get("WATCHER_CLASSIFICATION_YOLO_WEIGHTS", "")
    candidates = [env]
    for key in ("YOLO_WEIGHTS", "MODEL"):
        candidates.append(application_config("classification", key) or "")
    for configured in candidates:
        if not configured:
            continue
        if not (configured.endswith(".pt") or configured.endswith(".onnx")):
            continue
        p = Path(configured)
        if p.exists():
            return p
    return Path(DEFAULT_CLASSIFIER_WEIGHTS)


def _detector():
    """Laze-loaded ultralytics YOLO; cached across rq jobs in this process."""
    global _DETECTOR, _DETECTOR_WEIGHTS
    weights = _weights_path()
    key = str(weights)
    if _DETECTOR is None or _DETECTOR_WEIGHTS != key:
        try:
            from ultralytics import YOLO  # heavy import, done once
            _DETECTOR = YOLO(key)
            _DETECTOR_WEIGHTS = key
            logger.info("yolo classifier loaded %s", key)
        except Exception as e:
            logger.error("yolo load failed (%s); VLM fallback stays available", e)
            _DETECTOR = None
            _DETECTOR_WEIGHTS = None
            return None
    return _DETECTOR


_DETECTOR = None
_DETECTOR_WEIGHTS = None


def _classify_yolo(img: Image.Image, event_name: str) -> dict:
    """Return the same dict shape _query_ollama produces, from yolo11n."""
    det = _detector()
    if det is None:
        raise RuntimeError("yolo weights unavailable")

    t0 = time.perf_counter()
    results = det.predict(img, verbose=False, device="cpu")
    dt = (time.perf_counter() - t0) * 1000

    r = results[0]
    best, categories = None, set()
    for box in r.boxes:
        cls_id = int(box.cls[0])
        name = det.names.get(cls_id, f"class{cls_id}") if hasattr(det, "names") else f"class{cls_id}"
        category = CLASS_TO_CATEGORY.get(name)
        if category is None:
            continue
        categories.add(category)
        conf = float(box.conf[0])
        if best is None or conf > best[1]:
            best = (category, conf)

    if best is None:
        result = {"category": "unknown", "interesting": False,
                  "confidence": 0.0, "description": "no person/vehicle/animal detected"}
    else:
        category, conf = best
        others = categories - {category}
        desc = f"yolo11n: {category} ({conf:.2f})"
        if others:
            desc += " + " + ", ".join(sorted(others))
        result = {"category": category, "interesting": True,
                  "confidence": conf, "description": desc}

    _log_decision(event_name, "yolo11n", result, dt)
    return result


def _log_decision(event_name: str, model: str, parsed: dict, dt_ms: float):
    entry = {
        "ts":       datetime.now().isoformat(),
        "event":    event_name,
        "request":  {"model": model, "prompt": "yolo11n-detect", "format": {}},
        "response": json.dumps(parsed),
        "parsed":   parsed,
        "dt_ms":    round(dt_ms, 1),
    }
    try:
        from .classify_motion import _ollama_log_path
        with open(_ollama_log_path(), 'a') as f:
            f.write(json.dumps(entry) + '\n')
    except OSError as e:
        logger.warning(f"Could not write yolo log: {e}")