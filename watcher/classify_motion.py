import base64
import io
import json
import subprocess
from datetime import datetime
from pathlib import Path

import requests
import sqlalchemy
from PIL import Image
from rq import Queue, Retry, Worker

from .connection import TunneledConnection, application_config, redis_connection
from .model import EventObservation, Labeling

from . import setup_logging

__all__ = ["task_classify_motion", "run_classify_queue"]

logger = setup_logging()

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "moondream"
MAX_IMAGE_WIDTH = 640

CLASSIFICATION_PROMPT = """Security camera image. Reply with only this JSON, no other text:
{"category": "<person|vehicle|animal|lighting_change|wind_vegetation|shadow|unknown>", "interesting": <true|false>, "confidence": <0.0-1.0>}

interesting=true only if a person, vehicle, or animal is clearly visible."""


def _ollama_host():
    return application_config("classification", "OLLAMA_HOST") or DEFAULT_OLLAMA_HOST


def _ollama_model():
    return application_config("classification", "MODEL") or DEFAULT_MODEL


def _frame_from_event(event) -> Image.Image:
    """Return a PIL Image for the event.

    Uses the saved significant-frame JPEG if it exists on disk.
    Falls back to extracting the indexed frame (or a fixed timestamp)
    directly from the MP4 using ffmpeg — no pre-extracted file required.
    """
    # Try the saved JPEG first
    if event.results:
        ir = event.results[-1]
        from .connection import application_path_for

        saved = application_path_for(ir.file)
        if saved.exists():
            return Image.open(saved)
        # Use the stored frame index for extraction
        frame_idx = (ir.info or {}).get("most_significant_frame")
    else:
        frame_idx = None

    video_path = event.file_path
    if not video_path.exists():
        raise FileNotFoundError(f"video file {video_path} not found")

    return _extract_frame(video_path, frame_idx)


def _extract_frame(video_path: Path, frame_idx=None) -> Image.Image:
    """Extract a single frame from an MP4 via ffmpeg subprocess."""
    if frame_idx is not None:
        # Select the exact frame by index (0-based)
        vf = f"select=eq(n\\,{int(frame_idx)})"
        cmd = ["ffmpeg", "-i", str(video_path), "-vf", vf, "-vframes", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "3", "pipe:1"]
    else:
        # Fall back: grab frame ~2 seconds in
        cmd = ["ffmpeg", "-ss", "2", "-i", str(video_path), "-vframes", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "3", "pipe:1"]

    result = subprocess.run(cmd, capture_output=True, timeout=30)
    if result.returncode != 0 or not result.stdout:
        raise RuntimeError(f"ffmpeg failed extracting frame from {video_path}: {result.stderr.decode()[:300]}")
    return Image.open(io.BytesIO(result.stdout))


def _encode_pil(img: Image.Image) -> str:
    img.thumbnail((MAX_IMAGE_WIDTH, MAX_IMAGE_WIDTH))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _query_ollama(img: Image.Image, model: str = None) -> dict:
    host = _ollama_host()
    model = model or _ollama_model()

    payload = {
        "model": model,
        "prompt": CLASSIFICATION_PROMPT,
        "images": [_encode_pil(img)],
        "stream": False,
        "format": "json",
    }

    try:
        resp = requests.post(f"{host}/api/generate", json=payload, timeout=180)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise ConnectionError(f"Could not reach Ollama at {host}: {e}") from e

    raw = resp.json().get("response", "{}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"Ollama returned non-JSON response: {raw!r}")
        return {"category": "unknown", "interesting": False, "confidence": 0.0, "description": raw}


def task_classify_motion(event_name: str, model: str = None):
    model = model or _ollama_model()
    logger.debug(f"classifying {event_name} with {model}")

    with TunneledConnection() as tc:
        session = sqlalchemy.orm.Session(tc)
        event = EventObservation.by_name(session, event_name)
        if not event:
            raise ValueError(f"event {event_name} not found in database")

        img = _frame_from_event(event)
        result = _query_ollama(img, model=model)

        category = result.get("category", "unknown")
        # moondream sometimes returns interesting as a float — treat >0.5 as True
        raw_interesting = result.get("interesting", False)
        interesting = bool(raw_interesting) if isinstance(raw_interesting, bool) else float(raw_interesting) > 0.5
        confidence = float(result.get("confidence", 0.0))
        description = result.get("description", "") or ""

        logger.info(f"{event_name}: {category} interesting={interesting} conf={confidence:.2f} [{model}] — {description}")

        labels = [category]
        if not interesting:
            labels.append("noise")

        lbl = Labeling(
            event_id=event.id,
            decider=f"ollama:{model}",
            decided_at=datetime.now(),
            labels=labels,
            probabilities=[confidence],
            description=description or None,
            mask=None,
        )
        session.add(lbl)
        session.commit()


def run_classify_queue(queues=["classify_motion"]):
    worker = Worker(queues, connection=redis_connection())
    worker.work(with_scheduler=True)
