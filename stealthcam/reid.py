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

Descriptor (v2, 2026-09-13 — Percy P0-3 fix). The v1 descriptor was a
letterboxed 48x48 grayscale map concatenated with an HSV histogram. Its
normalized-L2 distance was dominated by the low-contrast luminance map (94.5% of
the vector's L2 mass), so different vehicles on similar backgrounds sat within
the 0.85 similarity threshold — the identity key *failed open* (a lookalike
inherited a resident's presence history -> "expected" -> no alert). v2 is
rebalanced so geometry/contour dominates:

1. **Foreground silhouette** — the crop is thresholded against its median to
   isolate the vehicle blob, cropped to the blob's bounding box, and resized to a
   canonical ``MASK_SIDE`` square. This keys on the vehicle's *shape*, is
   shift/scale invariant, and (critically) drops the letterboxed background that
   v1 was matching on.
2. **Gradient-orientation histogram** (dense HOG) over a higher-resolution
   ``EMBED_SIZE`` resize — contour/edge geometry, not coarse luminance.
3. **HSV (H x S) histogram** — the coarse color distribution, kept so color
   still contributes but no longer dominates.

Each part is unit-normalized before concatenation so no one part swamps the
others, and the normalized-L2 distance weights geometry over background.

The ``model`` string names the backend so decisions/entities can record which
re-id model produced a fingerprint — a different model's vectors are NOT
comparable, so we never mix distances across backends (v2 bumps the name, which
cleanly orphans any v1 rows rather than comparing across descriptor layouts).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image

# A fingerprint is a fixed-size float32 vector serialized to raw bytes. The
# shape is an implementation detail of the backend; only the *distance*
# function between two fingerprints is part of the cross-backend contract.
EMBED_SIZE = 96   # resize side for the orientation histogram
MASK_SIDE = 32    # canonical side for the foreground-silhouette map
HIST_BINS = 16    # per-channel HSV histogram bins (H x S)
HOG_CELL = 8      # orientation-histogram cell side (pixels)
HOG_BINS = 9      # orientation bins per cell (20deg each, 0-180)


class Fingerprinter(Protocol):
    """Produce a stable fingerprint from a vehicle crop (or None if it can't)."""

    model: str

    def fingerprint(self, crop: Image.Image) -> bytes | None:
        """Return a serialized numeric fingerprint for the crop, else None."""
        ...

    def distance(self, a: bytes, b: bytes) -> float:
        """Return a similarity distance in [0, 1] (0 = identical, 1 = unrelated)."""
        ...


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n == 0.0 else v / n


def _silhouette(gray: np.ndarray) -> np.ndarray:
    """Foreground-silhouette map: pixels differing from the crop's median.

    The crop is thresholded against its median luminance so the *vehicle* blob
    is isolated from its background, then cropped to the blob's bounding box and
    resized to a canonical square. This is shift/scale invariant and keys on
    shape, not the letterboxed background that the v1 descriptor was matching on.
    Returns a unit-normalized ``MASK_SIDE * MASK_SIDE`` vector (all-zero for a
    flat/uniform crop, where no silhouette can be extracted).
    """
    import cv2

    bg = float(np.median(gray))
    mask = (np.abs(gray.astype(np.float32) - bg) > 15.0).astype(np.uint8) * 255
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return np.zeros(MASK_SIDE * MASK_SIDE, dtype=np.float32)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    blob = mask[y0:y1 + 1, x0:x1 + 1]
    blob = cv2.resize(blob, (MASK_SIDE, MASK_SIDE), interpolation=cv2.INTER_AREA)
    return (blob.astype(np.float32) / 255.0).flatten()


def _orientation_hist(gray: np.ndarray) -> np.ndarray:
    """Dense gradient-orientation histogram (per-cell, magnitude-weighted).

    Contour/edge geometry rather than coarse luminance: each ``HOG_CELL`` cell
    votes its pixel gradients into ``HOG_BINS`` orientation bins. Cell
    histograms are individually L2-normalized so strong edges don't swamp faint
    ones, then concatenated.
    """
    import cv2

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    ang = (cv2.phase(gx, gy, angleInDegrees=True) % 180).astype(np.float32)
    h, w = gray.shape
    feats = []
    for y in range(0, h - HOG_CELL + 1, HOG_CELL):
        for x in range(0, w - HOG_CELL + 1, HOG_CELL):
            m = mag[y:y + HOG_CELL, x:x + HOG_CELL].flatten()
            a = ang[y:y + HOG_CELL, x:x + HOG_CELL].flatten()
            hist = np.zeros(HOG_BINS, dtype=np.float32)
            for i in range(m.size):
                b = int(a[i] // (180.0 / HOG_BINS)) % HOG_BINS
                hist[b] += m[i]
            feats.append(_unit(hist))
    return np.concatenate(feats)


def _embedding(crop: Image.Image) -> np.ndarray:
    """Silhouette + contour-orientation + color descriptor (v2).

    Built from three unit-normalized parts so identity keys on geometry (the
    foreground silhouette + contour orientation) with color as a secondary
    signal — and none of them depends on legible text or the letterboxed
    background:

    1. A foreground-silhouette map (the vehicle's shape, bbox-cropped and
       canonicalized) — shift/scale invariant.
    2. A dense gradient-orientation histogram over a higher-resolution resize
       (contour geometry).
    3. A normalized HSV (H x S) histogram (coarse color distribution), coarse
       enough that IR bloom / minor exposure drift don't flip identity.
    """
    import cv2

    rgb = np.array(crop.convert("RGB"))
    h, w = rgb.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("empty crop")

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(rgb, (EMBED_SIZE, EMBED_SIZE), interpolation=cv2.INTER_AREA)
    resized_gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

    hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [HIST_BINS, HIST_BINS],
                        [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten().astype(np.float32)

    return np.concatenate([
        _unit(_silhouette(gray)),
        _unit(_orientation_hist(resized_gray)),
        _unit(hist),
    ])


@dataclass
class OpenCVFingerprinter:
    """Default fingerprint backend (OpenCV silhouette + orientation + color).

    ``model`` names the backend so decisions/entities can record which re-id
    model produced a fingerprint (a different model's vectors are NOT
    comparable — we never mix distances across backends). v2 bumps the name
    because the descriptor layout changed; any v1 rows are cleanly orphaned
    rather than compared across incompatible vectors.
    """

    model: str = "cv2-silhouette-v2"
    # Similarity threshold above which two fingerprints are "the same entity".
    # With the normalized-L2 distance below, "similarity" is 1 - distance. The
    # v2 descriptor is discriminative enough that this bar separates
    # same-vehicle (rescales ~0.88) from lookalikes (~0.69) on the trust-boundary
    # constructions it was tuned against; it is configurable and should still be
    # re-tuned against a labeled set as one becomes available. When a crop is
    # too jittered/degraded to clear the bar, identity fails CLOSED (a new
    # entity -> alert), never open (wrongly matching a lookalike -> no alert).
    # This MUST match config.py's ``reid_similarity_threshold`` default.
    similarity_threshold: float = 0.80

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
        # parallel unit vectors.
        diff = float(np.linalg.norm(va - vb))
        mag = float(np.linalg.norm(va) + np.linalg.norm(vb))
        if mag == 0.0:
            return 1.0
        return min(1.0, diff / mag)

    def matches(self, a: bytes, b: bytes) -> bool:
        """True if two fingerprints are the same entity (similarity >= threshold)."""
        return (1.0 - self.distance(a, b)) >= self.similarity_threshold
