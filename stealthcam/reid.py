"""Vehicle re-identification fingerprint (the entity identity key).

The design decision (option C, settled by Tom + Aime): entity identity is a
**hybrid** — a lightweight numeric re-id embedding is the identity KEY used to
match a vehicle crop across captures, while the describer VLM supplies
human-readable **attributes** (make/model + side lettering) with an explicit
uncertainty signal.

Why not key identity on the YOLO class or the VLM description text:

- The eval showed the parked tractor classified as car/truck/train/sheep across
  captures — class is NOT stable identity.
- Night-IR eval (2026-09-13) showed fine lettering (e.g. FedEx door text)
  washes out under IR bloom, so a text/lettering key would be fragile at night.
  The identity key must come from **silhouette/geometry**, not legible text.

So this module produces a compact numeric fingerprint from the crop's shape and
color distribution, robust to the per-capture jitter and scale variation that
breaks exact-bbox equality. It is deliberately cheap (OpenCV only) and
swappable: the ``Fingerprinter`` protocol lets a dedicated re-id model replace
this default without touching the memory/worker wiring.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image

# A fingerprint is a fixed-size float32 vector serialized to raw bytes. The
# shape is an implementation detail of the backend; only the *distance*
# function between two fingerprints is part of the cross-backend contract.
EMBED_SIZE = 48   # grayscale silhouette side length
HIST_BINS = 16    # per-channel HSV histogram bins (H x S)


class Fingerprinter(Protocol):
    """Produce a stable fingerprint from a vehicle crop (or None if it can't)."""

    model: str

    def fingerprint(self, crop: Image.Image) -> bytes | None:
        """Return a serialized numeric fingerprint for the crop, else None."""
        ...

    def distance(self, a: bytes, b: bytes) -> float:
        """Return a similarity distance in [0, 1] (0 = identical, 1 = unrelated)."""
        ...


def _embedding(crop: Image.Image) -> np.ndarray:
    """Silhouette + color descriptor, normalized for scale/aspect/jitter.

    The descriptor is built from two parts so identity keys on BOTH geometry
    and color, but neither depends on legible text:

    1. A letterboxed, downsampled grayscale map (the silhouette / geometry) —
       letterboxing preserves aspect ratio and downsampling absorbs the few
       pixels of bbox jitter observed across captures.
    2. A normalized HSV (H x S) histogram (the coarse color distribution) —
       coarse enough that IR bloom / minor exposure drift don't flip identity.
    """
    import cv2

    rgb = np.array(crop.convert("RGB"))
    h, w = rgb.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("empty crop")

    # Letterbox to a square preserving aspect ratio (pad with neutral gray).
    side = max(h, w)
    canvas = np.full((side, side, 3), 128, dtype=np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = rgb

    resized = cv2.resize(canvas, (EMBED_SIZE, EMBED_SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [HIST_BINS, HIST_BINS],
                        [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten().astype(np.float32)

    return np.concatenate([gray.flatten(), hist])


@dataclass
class OpenCVFingerprinter:
    """Default fingerprint backend (OpenCV silhouette + color histogram).

    ``model`` names the backend so decisions/entities can record which re-id
    model produced a fingerprint (a different model's vectors are NOT
    comparable — we never mix distances across backends).
    """

    model: str = "cv2-silhouette-v1"
    # Similarity threshold above which two fingerprints are "the same entity".
    # With the normalized-L2 distance below, "similarity" is 1 - distance, so
    # this is a strict bar tuned for silhouette+color robustness. It is
    # configurable and should be re-tuned against a labeled set (the design
    # explicitly calls for a small validation set before relying on it).
    similarity_threshold: float = 0.85

    def fingerprint(self, crop: Image.Image) -> bytes | None:
        try:
            emb = _embedding(crop)
        except Exception:
            # A degenerate/empty crop yields no fingerprint; the caller treats
            # it as "identity unknown" (still captured, just not re-id'd).
            return None
        return emb.astype(np.float32).tobytes()

    def distance(self, a: bytes, b: bytes) -> float:
        va = np.frombuffer(a, dtype=np.float32)
        vb = np.frombuffer(b, dtype=np.float32)
        if va.shape != vb.shape or va.size == 0:
            return 1.0
        # Normalized L2 distance in [0, 1]: 0 = identical, larger = more
        # different. Scale-SENSITIVE (unlike cosine), so a flat red crop and a
        # flat blue crop are correctly distinguished rather than collapsing to
        # parallel unit vectors. This also absorbs the letterbox padding: a
        # constant offset (the gray pad) cancels in the difference.
        diff = float(np.linalg.norm(va - vb))
        mag = float(np.linalg.norm(va) + np.linalg.norm(vb))
        if mag == 0.0:
            return 1.0
        return min(1.0, diff / mag)

    def matches(self, a: bytes, b: bytes) -> bool:
        """True if two fingerprints are the same entity (similarity >= threshold)."""
        return (1.0 - self.distance(a, b)) >= self.similarity_threshold
