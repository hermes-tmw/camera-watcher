"""VLM describer: describe(crop) -> natural-language description.

This is a *description* task, not a verification task (the eval rejected
VLM-as-verifier). For each YOLO bbox we crop the region and ask a local VLM
(llama3.2-vision) for a short natural-language description.

Entity-memory addition: ``describe_vehicle`` asks the VLM for structured
human-readable **attributes** (make/model + side lettering) with an explicit
uncertainty signal. These are the *attributes* of an entity, NOT its identity
key — identity comes from the re-id fingerprint (see ``reid.py``). At night
under IR, make/model and lettering are expected to be less reliable (bloom
washes out fine text), so uncertainty is always surfaced rather than guessed.
"""
from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass

import requests
from PIL import Image

DESCRIBE_PROMPT = (
    "This is a crop from a trail-camera photo. Describe the object in the crop "
    "in a short phrase (e.g. 'person with chainsaw', 'dog', 'parked tractor', "
    "'tree stump', 'deer'). Answer with only the phrase, no preamble."
)

VEHICLE_ATTR_PROMPT = (
    "This is a crop from a trail-camera photo that may contain a vehicle. "
    "Extract structured attributes. Reply with ONLY a JSON object, no prose, "
    "with exactly these keys:\n"
    '  "make": the vehicle make or "" if not determinable,\n'
    '  "model": the vehicle model or "" if not determinable,\n'
    '  "lettering": any legible side/door text (e.g. "FedEx", a company name) '
    'or "" if none,\n'
    '  "uncertain": true if you cannot determine make/model with confidence, '
    "else false.\n"
    "Fine lettering often washes out under infrared; prefer \"\" over a guess.\n"
    'Example: {"make": "Chevrolet", "model": "Tahoe", "lettering": "", '
    '"uncertain": false}'
)


@dataclass
class VehicleAttributes:
    make: str = ""
    model: str = ""
    lettering: str = ""
    uncertain: bool = True

    def as_text(self) -> str:
        """Human-readable one-liner (used in alerts + dashboard notes)."""
        parts = []
        if self.make or self.model:
            parts.append(f"{self.make} {self.model}".strip())
        if self.lettering:
            parts.append(f'"{self.lettering}"')
        if not parts:
            return "unidentified vehicle"
        s = " ".join(parts)
        return f"{s} (uncertain)" if self.uncertain else s


class Describer:
    def __init__(self, model: str, ollama_host: str = "http://localhost:11434",
                 timeout: int = 180):
        self.model = model
        self.ollama_host = ollama_host.rstrip("/")
        self.timeout = timeout

    def _generate(self, prompt: str, crop: Image.Image) -> str:
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        r = requests.post(
            f"{self.ollama_host}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "images": [b64],
                "stream": False,
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        return (r.json().get("response") or "").strip()

    def describe(self, crop: Image.Image) -> str:
        return self._generate(DESCRIBE_PROMPT, crop)

    def describe_vehicle(self, crop: Image.Image) -> VehicleAttributes:
        """Return structured make/model/lettering attributes for a vehicle crop.

        Never raises: a VLM/network failure or a non-JSON answer degrades to an
        ``uncertain`` empty attribute set (best-effort attribute extraction —
        the identity decision does not depend on it).
        """
        try:
            raw = self._generate(VEHICLE_ATTR_PROMPT, crop)
        except Exception:
            return VehicleAttributes(uncertain=True)
        try:
            # The VLM may wrap the JSON in a markdown code fence or prose.
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return VehicleAttributes(uncertain=True)
            data = json.loads(raw[start:end + 1])
        except (ValueError, json.JSONDecodeError):
            return VehicleAttributes(uncertain=True)
        if not isinstance(data, dict):
            return VehicleAttributes(uncertain=True)

        make = str(data.get("make", "") or "").strip()
        model = str(data.get("model", "") or "").strip()
        lettering = str(data.get("lettering", "") or "").strip()
        uncertain = bool(data.get("uncertain", True))

        # If the VLM returned nothing but claimed certainty, that's a
        # contradiction — surface as uncertain rather than trusting a blank
        # confident answer.
        if not make and not model and not lettering:
            uncertain = True
        return VehicleAttributes(make=make, model=model, lettering=lettering,
                                  uncertain=uncertain)
