"""YOLO11n detector: detect(image) -> [{class, confidence, bbox}].

Runs all 80 COCO classes; the caller filters to person/vehicle/animal.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

# COCO class names (index = class id)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

# Classes we care about, grouped by alert category.
PERSON_CLASSES = {"person"}
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "train", "boat", "airplane"}
ANIMAL_CLASSES = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
                  "zebra", "giraffe"}


@dataclass
class Detection:
    cls: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2

    @property
    def category(self) -> str:
        if self.cls in PERSON_CLASSES:
            return "person"
        if self.cls in VEHICLE_CLASSES:
            return "vehicle"
        if self.cls in ANIMAL_CLASSES:
            return "animal"
        return "other"


class Detector:
    """Thin wrapper around ultralytics YOLO11n."""

    def __init__(self, weights: str, conf_threshold: float = 0.6, device: str | int = 0):
        from ultralytics import YOLO  # imported lazily (heavy)

        self._model = YOLO(weights)
        self.conf_threshold = conf_threshold
        self.device = device

    def detect(self, image: Image.Image | str | Path) -> list[Detection]:
        """Run detection and return detections above the confidence threshold."""
        r = self._model.predict(image, verbose=False, device=self.device)[0]
        out: list[Detection] = []
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < self.conf_threshold:
                continue
            cls = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"class{cls_id}"
            xyxy = box.xyxy[0].tolist()
            out.append(Detection(cls=cls, confidence=conf, bbox=tuple(xyxy)))
        return out
