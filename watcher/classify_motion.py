import base64
import io
import json
from datetime import datetime
from pathlib import Path

import requests
import sqlalchemy
from PIL import Image
from rq import Queue, Retry, Worker

from .connection import TunneledConnection, application_config, redis_connection, application_path_for
from .model import EventObservation, Labeling

from . import setup_logging

__all__ = ['task_classify_motion', 'run_classify_queue']

logger = setup_logging()

DEFAULT_OLLAMA_HOST = 'http://localhost:11434'
DEFAULT_MODEL = 'moondream'
MAX_IMAGE_WIDTH = 640

CLASSIFICATION_PROMPT = """You are analyzing a security camera image. Classify the motion event.

Respond with ONLY a JSON object, no other text:
{
  "category": "person" or "vehicle" or "animal" or "lighting_change" or "wind_vegetation" or "shadow" or "unknown",
  "interesting": true if a person/vehicle/animal is clearly present, false if motion is environmental,
  "confidence": a number from 0.0 to 1.0,
  "description": "one brief sentence describing what you see"
}

Is there a distinct foreground subject (person, car, animal)? Or is the motion environmental (trees swaying, light changing, shadow moving)?"""


def _ollama_host():
    return application_config('classification', 'OLLAMA_HOST') or DEFAULT_OLLAMA_HOST

def _ollama_model():
    return application_config('classification', 'MODEL') or DEFAULT_MODEL

def _encode_image(img_path: Path) -> str:
    with Image.open(img_path) as img:
        img.thumbnail((MAX_IMAGE_WIDTH, MAX_IMAGE_WIDTH))
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=85)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

def _query_ollama(img_path: Path) -> dict:
    host = _ollama_host()
    model = _ollama_model()

    payload = {
        'model': model,
        'prompt': CLASSIFICATION_PROMPT,
        'images': [_encode_image(img_path)],
        'stream': False,
        'format': 'json',
    }

    try:
        resp = requests.post(f'{host}/api/generate', json=payload, timeout=90)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise ConnectionError(f"Could not reach Ollama at {host}: {e}") from e

    raw = resp.json().get('response', '{}')
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"Ollama returned non-JSON response: {raw!r}")
        return {'category': 'unknown', 'interesting': False, 'confidence': 0.0, 'description': raw}


def task_classify_motion(img_relpath: str, event_name: str):
    img_path = application_path_for(img_relpath)
    logger.debug(f"classifying {img_path} for {event_name}")

    if not img_path.exists():
        raise FileNotFoundError(f"image file {img_path} does not exist")

    result = _query_ollama(img_path)

    category = result.get('category', 'unknown')
    interesting = result.get('interesting', False)
    confidence = float(result.get('confidence', 0.0))
    description = result.get('description', '')

    logger.info(f"{event_name}: {category} interesting={interesting} conf={confidence:.2f} — {description}")

    # Labels follow the existing convention: include 'noise' for environmental events
    labels = [category]
    if not interesting:
        labels.append('noise')

    with TunneledConnection() as tc:
        session = sqlalchemy.orm.Session(tc)
        event = EventObservation.by_name(session, event_name)
        if not event:
            raise ValueError(f"event {event_name} not found in database")

        lbl = Labeling(
            event_id=event.id,
            decider=f'ollama:{_ollama_model()}',
            decided_at=datetime.now(),
            labels=labels,
            probabilities=[confidence],
            mask=None,
        )
        session.add(lbl)
        session.commit()


def run_classify_queue(queues=['classify_motion']):
    worker = Worker(queues, connection=redis_connection())
    worker.work(with_scheduler=True)
