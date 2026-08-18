"""Memory: per-direction + bbox persistence -> alert decision.

The unifying mechanism from the design doc. For each direction we track
persistent objects by their bounding box, and decide whether a detection is
"static background" (seen before, persistently) or "novel".

Key design points (from the eval):
  - The parked tractor is misclassified by YOLO as *different* classes
    (car/truck/train/sheep) at *slightly jittering* bboxes across captures.
    Class is therefore NOT a reliable part of an object's identity, and exact
    bbox equality is too strict (a few pixels of jitter). So we match objects
    by **IoU overlap** within a direction: a detection that overlaps an
    existing observation by >= iou_threshold is "the same object".
  - Persistence = same direction + overlapping bbox across >=N captures AND
    >=M hours -> static background -> suppress (any class).

Decision rules:
  - static object (persistent) -> suppress (any class — it's background)
  - person  -> alert (people are transient by nature)
  - vehicle -> alert only if novel (not static — someone drove in, not the
               parked tractor that's always there)
  - animal  -> flag, don't alert
  - other   -> suppress

The store is a SQLite DB owned by the pipeline (single-writer). It also holds
the seen_guids dedup set and the decisions/feedback tables (feedback-loop data).
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Alert decision outcomes
ALERT = "alert"
FLAG = "flag"
SUPPRESS = "suppress"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS seen_guids (
    guid TEXT PRIMARY KEY,
    seen_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_observations (
    id INTEGER PRIMARY KEY,
    direction TEXT NOT NULL,
    bbox TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    capture_count INTEGER NOT NULL DEFAULT 1,
    classes_seen TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY,
    guid TEXT NOT NULL,
    subimage INTEGER NOT NULL,
    direction TEXT NOT NULL,
    class TEXT NOT NULL,
    confidence REAL NOT NULL,
    bbox TEXT NOT NULL,
    description TEXT,
    decision TEXT NOT NULL,
    reason TEXT NOT NULL,
    decided_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY,
    decision_id INTEGER,
    guid TEXT NOT NULL,
    subimage INTEGER NOT NULL,
    human_label TEXT NOT NULL,
    flagged_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse(dt: str) -> datetime:
    return datetime.fromisoformat(dt)


def _iou(a: tuple[float, float, float, float],
         b: tuple[float, float, float, float]) -> float:
    """Intersection-over-union of two bboxes (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Decision:
    decision: str  # alert | flag | suppress
    reason: str
    capture_count: int
    is_static: bool


class Memory:
    def __init__(self, db_path: str | Path, iou_threshold: float = 0.5,
                 static_n_captures: int = 5, static_min_hours: float = 1.0):
        self.db_path = str(db_path)
        self.iou_threshold = iou_threshold
        self.static_n_captures = static_n_captures
        self.static_min_hours = static_min_hours
        self._conn = sqlite3.connect(self.db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- dedup ---------------------------------------------------------------

    def has_seen(self, guid: str) -> bool:
        cur = self._conn.execute("SELECT 1 FROM seen_guids WHERE guid = ?", (guid,))
        return cur.fetchone() is not None

    def mark_seen(self, guid: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO seen_guids (guid, seen_at) VALUES (?, ?)",
            (guid, _now()),
        )
        self._conn.commit()

    # -- object matching -----------------------------------------------------

    def _find_match(self, direction: str, bbox: tuple[float, float, float, float]):
        """Return the best-matching observation row (id, first_seen_at, count,
        classes_seen, bbox) in this direction with IoU >= threshold, else None."""
        rows = self._conn.execute(
            "SELECT id, first_seen_at, capture_count, classes_seen, bbox "
            "FROM memory_observations WHERE direction = ?",
            (direction,),
        ).fetchall()
        best = None
        best_iou = 0.0
        for row in rows:
            other = tuple(json.loads(row[4]))
            iou = _iou(bbox, other)
            if iou > best_iou:
                best_iou = iou
                best = row
        if best is not None and best_iou >= self.iou_threshold:
            return best
        return None

    # -- decision ------------------------------------------------------------

    def decide(self, direction: str, cls: str, bbox: tuple[float, float, float, float],
               now: str | None = None) -> Decision:
        """Record the observation and return the alert decision.

        `cls` is the YOLO class name (e.g. 'person', 'car', 'sheep'). The
        category is derived from it. Static suppression applies to any class.
        """
        now = now or _now()
        category = _category(cls)

        match = self._find_match(direction, bbox)

        if match is None:
            self._conn.execute(
                "INSERT INTO memory_observations "
                "(direction, bbox, first_seen_at, last_seen_at, capture_count, classes_seen) "
                "VALUES (?, ?, ?, ?, 1, ?)",
                (direction, json.dumps(list(bbox)), now, now, json.dumps([cls])),
            )
            capture_count = 1
            first_seen_at = now
        else:
            oid, first_seen_at, capture_count, classes_seen, _ = match
            capture_count += 1
            classes = json.loads(classes_seen)
            if cls not in classes:
                classes.append(cls)
            self._conn.execute(
                "UPDATE memory_observations SET capture_count = ?, last_seen_at = ?, "
                "classes_seen = ? WHERE id = ?",
                (capture_count, now, json.dumps(classes), oid),
            )
        self._conn.commit()

        # static = same direction + overlapping bbox across >=N captures AND >=M hours
        span_hours = (_parse(now) - _parse(first_seen_at)).total_seconds() / 3600.0
        is_static = capture_count >= self.static_n_captures and span_hours >= self.static_min_hours

        if is_static:
            return Decision(SUPPRESS, f"static object ({cls}, {capture_count} captures over {span_hours:.1f}h)",
                           capture_count, True)
        if category == "person":
            return Decision(ALERT, "person", capture_count, False)
        if category == "vehicle":
            return Decision(ALERT, "novel vehicle", capture_count, False)
        if category == "animal":
            return Decision(FLAG, "animal", capture_count, False)
        return Decision(SUPPRESS, f"non-target class {cls}", capture_count, False)

    # -- decisions / feedback (feedback-loop data) ---------------------------

    def record_decision(self, guid: str, subimage: int, direction: str, cls: str,
                        confidence: float, bbox: tuple[float, float, float, float],
                        description: str, decision: Decision) -> int:
        cur = self._conn.execute(
            "INSERT INTO decisions (guid, subimage, direction, class, confidence, bbox, "
            "description, decision, reason, decided_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (guid, subimage, direction, cls, confidence, json.dumps(list(bbox)),
             description, decision.decision, decision.reason, _now()),
        )
        self._conn.commit()
        return cur.lastrowid

    def record_feedback(self, decision_id: int, guid: str, subimage: int,
                        human_label: str) -> None:
        self._conn.execute(
            "INSERT INTO feedback (decision_id, guid, subimage, human_label, flagged_at) "
            "VALUES (?,?,?,?,?)",
            (decision_id, guid, subimage, human_label, _now()),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def _category(cls: str) -> str:
    from .detector import PERSON_CLASSES, VEHICLE_CLASSES, ANIMAL_CLASSES

    if cls in PERSON_CLASSES:
        return "person"
    if cls in VEHICLE_CLASSES:
        return "vehicle"
    if cls in ANIMAL_CLASSES:
        return "animal"
    return "other"
