"""Annotate: draw bbox highlight + description on a directional sub-image."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# BGR colors per category
CATEGORY_COLOR = {
    "person": (0, 0, 255),    # red
    "vehicle": (255, 0, 0),   # blue
    "animal": (0, 200, 0),    # green
    "other": (128, 128, 128),
}


def annotate(image: Image.Image, detections: list, descriptions: dict[int, str] | None = None,
             dest: Path | None = None) -> Image.Image:
    """Draw a bbox + label for each detection; return the annotated PIL image.

    `detections` is a list of Detection. `descriptions` maps detection index ->
    description string (optional). If `dest` is given, also save a JPEG there.
    """
    img = np.array(image.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    descriptions = descriptions or {}

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        color = CATEGORY_COLOR.get(det.category, CATEGORY_COLOR["other"])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{det.cls} {det.confidence:.2f}"
        if i in descriptions and descriptions[i]:
            label += f": {descriptions[i]}"
        cv2.putText(img, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)

    out = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if dest is not None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        out.save(dest, format="JPEG", quality=90)
    return out
