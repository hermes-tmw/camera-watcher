"""VLM describer: describe(crop) -> natural-language description.

This is a *description* task, not a verification task (the eval rejected
VLM-as-verifier). For each YOLO bbox we crop the region and ask a local VLM
(llama3.2-vision) for a short natural-language description.
"""
from __future__ import annotations

import base64
import io

import requests
from PIL import Image

DESCRIBE_PROMPT = (
    "This is a crop from a trail-camera photo. Describe the object in the crop "
    "in a short phrase (e.g. 'person with chainsaw', 'dog', 'parked tractor', "
    "'tree stump', 'deer'). Answer with only the phrase, no preamble."
)


class Describer:
    def __init__(self, model: str, ollama_host: str = "http://localhost:11434",
                 timeout: int = 180):
        self.model = model
        self.ollama_host = ollama_host.rstrip("/")
        self.timeout = timeout

    def describe(self, crop: Image.Image) -> str:
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        r = requests.post(
            f"{self.ollama_host}/api/generate",
            json={
                "model": self.model,
                "prompt": DESCRIBE_PROMPT,
                "images": [b64],
                "stream": False,
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        return (r.json().get("response") or "").strip()
