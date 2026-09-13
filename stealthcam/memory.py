"""Memory: per-direction + bbox persistence -> alert decision.

The unifying mechanism from the design doc. For each direction we track
persistent objects by their bounding box, and decide whether a detection is
"static background" (seen before, persistently) or "novel".

Entity-memory (2026-09-13) extends this with **identity** so "normally here"
is no longer conflated with "static position":

- Each vehicle detection crop gets a numeric re-id **fingerprint** (the
  identity KEY) and human-readable **attributes** (make/model + side lettering)
  from the describer VLM (the human-readable layer).
- An entity is matched by fingerprint similarity within a direction; the
  observation row gains an ``entity_id`` that persists across captures even if
  the vehicle leaves and returns to the same spot.
- Per ``entity_id x direction`` we learn an **expected-presence baseline**: the
  fraction of monitored captures (rolling window) the entity appeared at that
  location. High presence = "typically here" = background note (no alert);
  low/zero prior presence = unusual = alert.

Location policy (per direction):
- ``driveway`` — vehicles get identity-expectedness evaluation (unfamiliar ->
  alert).
- ``road``     — pass-by: dashboard note tagged ``traffic``, no alert.
- unclassified — fall back to the current novel-vehicle policy (static ->
  suppress, else novel vehicle -> alert).

Key design points (from the eval):
  - The parked tractor is misclassified by YOLO as *different* classes
    (car/truck/train/sheep) at *slightly jittering* bboxes across captures.
    Class is therefore NOT a reliable part of an object's identity, and exact
    bbox equality is too strict (a few pixels of jitter). Position persistence
    matches by **IoU overlap** within a direction; *identity* matches by
    fingerprint, which is stable across that class drift.
  - Persistence = same direction + overlapping bbox across >=N captures AND
    >=M hours -> static background -> suppress (non-vehicle classes, and
    vehicles in unclassified regions).

Decision rules:
  - static non-vehicle object -> suppress (background)
  - person  -> alert (people are transient by nature)
  - animal  -> flag, don't alert
  - vehicle (road) -> note (traffic), no alert
  - vehicle (driveway, familiar) -> note (background), no alert
  - vehicle (driveway, unfamiliar) -> alert (with make/model + lettering)
  - vehicle (unclassified) -> static -> suppress, else novel -> alert
  - other   -> suppress

The store is a SQLite DB owned by the pipeline (single-writer). It also holds
the seen_guids dedup set and the decisions/feedback tables (feedback-loop data).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Alert decision outcomes
ALERT = "alert"
FLAG = "flag"
SUPPRESS = "suppress"
NOTE = "note"

# Dashboard note tags (location policy).
TAG_TRAFFIC = "traffic"
TAG_BACKGROUND = "background"

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
CREATE TABLE IF NOT EXISTS deferred_captures (
    guid TEXT PRIMARY KEY,
    deferral_count INTEGER NOT NULL DEFAULT 1,
    alerted INTEGER NOT NULL DEFAULT 0,
    first_deferred_at TEXT NOT NULL,
    last_deferred_at TEXT NOT NULL
);
-- Entity identity + expected-presence baseline (2026-09-13).
CREATE TABLE IF NOT EXISTS entities (
    entity_id TEXT PRIMARY KEY,
    direction TEXT NOT NULL,
    region TEXT,
    fingerprint BLOB NOT NULL,
    model TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    capture_count INTEGER NOT NULL DEFAULT 1,
    attributes TEXT NOT NULL DEFAULT '{}',
    expected INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS entity_presence (
    entity_id TEXT NOT NULL,
    direction TEXT NOT NULL,
    guid TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    ts REAL NOT NULL,
    PRIMARY KEY (entity_id, direction, guid)
);
CREATE TABLE IF NOT EXISTS monitored_captures (
    direction TEXT NOT NULL,
    region TEXT,
    guid TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    ts REAL NOT NULL,
    is_day INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (direction, guid)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse(dt: str) -> datetime:
    return datetime.fromisoformat(dt)


def _ts(dt: str) -> float:
    """Epoch seconds for a capture timestamp, tolerant of naive/aware/Z forms."""
    d = datetime.fromisoformat(dt)
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.timestamp()


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
    decision: str  # alert | flag | suppress | note
    reason: str
    capture_count: int
    is_static: bool
    entity_id: str | None = None
    tag: str | None = None          # "traffic" | "background" | None
    attributes_text: str = ""        # make/model + lettering one-liner

    @property
    def is_alert(self) -> bool:
        return self.decision == ALERT

    @property
    def is_note(self) -> bool:
        return self.decision == NOTE


class Memory:
    def __init__(self, db_path: str | Path, iou_threshold: float = 0.5,
                 static_n_captures: int = 5, static_min_hours: float = 1.0,
                 fingerprinter=None,
                 entity_expected_presence_ratio: float = 0.5,
                 entity_baseline_window_days: float = 7.0,
                 entity_expected_absence_days: float = 2.0):
        self.db_path = str(db_path)
        self.iou_threshold = iou_threshold
        self.static_n_captures = static_n_captures
        self.static_min_hours = static_min_hours
        self.entity_expected_presence_ratio = entity_expected_presence_ratio
        self.entity_baseline_window_days = entity_baseline_window_days
        self.entity_expected_absence_days = entity_expected_absence_days
        if fingerprinter is None:
            from .reid import OpenCVFingerprinter
            fingerprinter = OpenCVFingerprinter()
        self.fingerprinter = fingerprinter
        self._conn = sqlite3.connect(self.db_path)
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Additive schema migrations for pre-existing databases.

        ``CREATE TABLE IF NOT EXISTS`` does not add columns to a table that
        already exists, so column additions are applied here idempotently.
        """
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(entities)")}
        if "expected" not in cols:
            self._conn.execute(
                "ALTER TABLE entities ADD COLUMN expected INTEGER NOT NULL DEFAULT 0"
            )
        mcols = {r[1] for r in self._conn.execute("PRAGMA table_info(monitored_captures)")}
        if "is_day" not in mcols:
            self._conn.execute(
                "ALTER TABLE monitored_captures ADD COLUMN is_day INTEGER NOT NULL DEFAULT 1"
            )

    # -- dedup ---------------------------------------------------------------

    def has_seen(self, guid: str) -> bool:
        cur = self._conn.execute("SELECT 1 FROM seen_guids WHERE guid = ?", (guid,))
        return cur.fetchone() is not None

    def mark_seen(self, guid: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO seen_guids (guid, seen_at) VALUES (?, ?)",
            (guid, _now()),
        )
        # A capture that was previously deferred is now successfully seen —
        # drop its deferral row so it no longer counts toward the alert
        # threshold or the requeue backfill.
        self._conn.execute("DELETE FROM deferred_captures WHERE guid = ?", (guid,))
        self._conn.commit()

    # -- deferred captures ---------------------------------------------------

    def record_deferral(self, guid: str) -> int:
        """Record that a capture's sub-images all failed to download this poll.

        Increments the deferral count (or inserts a new row) and returns the
        new count. A capture that later downloads successfully is marked seen
        and its deferral row is cleared (see mark_seen).
        """
        now = _now()
        cur = self._conn.execute(
            "SELECT deferral_count FROM deferred_captures WHERE guid = ?", (guid,)
        )
        row = cur.fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO deferred_captures "
                "(guid, deferral_count, alerted, first_deferred_at, last_deferred_at) "
                "VALUES (?, 1, 0, ?, ?)",
                (guid, now, now),
            )
            count = 1
        else:
            count = row[0] + 1
            self._conn.execute(
                "UPDATE deferred_captures SET deferral_count = ?, last_deferred_at = ? "
                "WHERE guid = ?",
                (count, now, guid),
            )
        self._conn.commit()
        return count

    def is_deferred_alerted(self, guid: str) -> bool:
        cur = self._conn.execute(
            "SELECT alerted FROM deferred_captures WHERE guid = ?", (guid,)
        )
        row = cur.fetchone()
        return bool(row and row[0])

    def mark_deferred_alerted(self, guid: str) -> None:
        self._conn.execute(
            "UPDATE deferred_captures SET alerted = 1 WHERE guid = ?", (guid,)
        )
        self._conn.commit()

    def clear_deferral(self, guid: str) -> None:
        """Drop a capture's deferral row once it has been successfully seen."""
        self._conn.execute("DELETE FROM deferred_captures WHERE guid = ?", (guid,))
        self._conn.commit()

    def deferred_guids(self) -> list[str]:
        """Return the guids currently tracked as deferred (never marked seen)."""
        rows = self._conn.execute(
            "SELECT guid FROM deferred_captures ORDER BY last_deferred_at"
        ).fetchall()
        return [r[0] for r in rows]

    # -- object matching (position persistence) ------------------------------

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

    # -- entity identity -----------------------------------------------------

    def _match_entity(self, direction: str, fingerprint: bytes) -> str | None:
        """Return the entity_id of a same-model entity in `direction` whose
        stored fingerprint matches `fingerprint`, else None."""
        model = getattr(self.fingerprinter, "model", "")
        rows = self._conn.execute(
            "SELECT entity_id, fingerprint FROM entities "
            "WHERE direction = ? AND model = ?",
            (direction, model),
        ).fetchall()
        for entity_id, stored in rows:
            if self.fingerprinter.matches(fingerprint, stored):
                return entity_id
        return None

    def _new_entity_id(self, direction: str, fingerprint: bytes) -> str:
        # Salt the hash with direction so a byte-identical fingerprint in two
        # mapped directions does NOT collide on the global `entity_id` PK
        # (entity matching is direction-scoped, so a same-bytes crop arriving
        # in a second direction would otherwise INSERT the same hash and raise
        # sqlite3.IntegrityError out of decide()).
        digest = hashlib.sha256(
            direction.encode("utf-8") + b"|" + fingerprint
        ).hexdigest()[:16]
        return f"e-{digest}"

    def _resolve_entity(self, direction: str, fingerprint: bytes,
                        attributes_text: str, attributes_json: str,
                        now: str) -> tuple[str, bool]:
        """Return (entity_id, created) for a fingerprint in `direction`.

        Matches an existing entity by fingerprint similarity (same model); if
        none, creates a new entity keyed on the fingerprint hash. Updates the
        entity's last_seen_at / capture_count / attributes on every sighting.
        """
        model = getattr(self.fingerprinter, "model", "")
        entity_id = self._match_entity(direction, fingerprint)
        created = False
        if entity_id is None:
            entity_id = self._new_entity_id(direction, fingerprint)
            created = True
            self._conn.execute(
                "INSERT INTO entities (entity_id, direction, fingerprint, model, "
                "first_seen_at, last_seen_at, capture_count, attributes) "
                "VALUES (?, ?, ?, ?, ?, ?, 1, ?)",
                (entity_id, direction, fingerprint, model, now, now, attributes_json),
            )
        else:
            self._conn.execute(
                "UPDATE entities SET last_seen_at = ?, capture_count = capture_count + 1 "
                "WHERE entity_id = ?",
                (now, entity_id),
            )
            # Enrich attributes when new info appears (never overwrite a known
            # make/model with a blank one).
            if attributes_text:
                self._merge_attributes(entity_id, attributes_json)
        return entity_id, created

    def _merge_attributes(self, entity_id: str, attributes_json: str) -> None:
        cur = self._conn.execute(
            "SELECT attributes FROM entities WHERE entity_id = ?", (entity_id,)
        )
        row = cur.fetchone()
        if row is None:
            return
        try:
            existing = json.loads(row[0] or "{}")
        except ValueError:
            existing = {}
        try:
            incoming = json.loads(attributes_json or "{}")
        except ValueError:
            incoming = {}
        merged = {}
        for key in ("make", "model", "lettering"):
            merged[key] = incoming.get(key) or existing.get(key) or ""
        # Once confident, stay confident unless the new sighting is explicit.
        merged["uncertain"] = incoming.get("uncertain", existing.get("uncertain", True))
        self._conn.execute(
            "UPDATE entities SET attributes = ? WHERE entity_id = ?",
            (json.dumps(merged), entity_id),
        )

    def record_presence(self, entity_id: str, direction: str, guid: str,
                        captured_at: str) -> None:
        """Record that `entity_id` appeared at `direction` during capture `guid`."""
        ts = _ts(captured_at)
        self._conn.execute(
            "INSERT OR IGNORE INTO entity_presence "
            "(entity_id, direction, guid, captured_at, ts) VALUES (?, ?, ?, ?, ?)",
            (entity_id, direction, guid, captured_at, ts),
        )
        self._conn.commit()

    def record_monitored(self, direction: str, region: str | None, guid: str,
                         captured_at: str, is_day: bool = True) -> None:
        """Record that capture `guid` was monitored at `direction` (denominator
        of the presence baseline), regardless of whether anything was detected.

        ``is_day`` marks whether the capture falls in the configured day window;
        the presence denominator counts DAY captures only, so night captures
        (whose numerator the night confidence gate mostly removes) do not drag
        a familiar vehicle's presence ratio permanently under threshold.
        """
        ts = _ts(captured_at)
        self._conn.execute(
            "INSERT OR IGNORE INTO monitored_captures "
            "(direction, region, guid, captured_at, ts, is_day) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (direction, region, guid, captured_at, ts, 1 if is_day else 0),
        )
        self._conn.commit()

    def presence_ratio(self, entity_id: str, direction: str,
                       now: str) -> float:
        """Fraction of monitored DAY captures (rolling window) the entity appeared
        at `direction`. 0.0 when there is no monitored day history yet."""
        cutoff = _ts(now) - self.entity_baseline_window_days * 86400.0
        num_pres = self._conn.execute(
            "SELECT COUNT(*) FROM entity_presence "
            "WHERE entity_id = ? AND direction = ? AND ts >= ?",
            (entity_id, direction, cutoff),
        ).fetchone()[0]
        num_mon = self._conn.execute(
            "SELECT COUNT(*) FROM monitored_captures "
            "WHERE direction = ? AND ts >= ? AND is_day = 1",
            (direction, cutoff),
        ).fetchone()[0]
        if num_mon == 0:
            return 0.0
        return num_pres / float(num_mon)

    # -- decision ------------------------------------------------------------

    def decide(self, direction: str, cls: str, bbox: tuple[float, float, float, float],
               now: str | None = None, *,
               fingerprint: bytes | None = None,
               region: str | None = None,
               attributes: object | None = None,
               attributes_text: str = "",
               guid: str | None = None) -> Decision:
        """Record the observation and return the alert decision.

        ``cls`` is the YOLO class name (e.g. 'person', 'car', 'sheep'). The
        category is derived from it. Static suppression applies to non-vehicle
        classes and to vehicles in unclassified regions.

        Entity-memory additions (all keyword-only, backward compatible):

        - ``fingerprint`` — the re-id bytes for a vehicle crop (identity key).
        - ``region``     — the resolved spatial region ("driveway"/"road"/None).
        - ``attributes`` — a VehicleAttributes (make/model/lettering).
        - ``guid``       — the capture guid, used to dedupe presence/monitored
          rows (falls back to ``now`` when absent).
        """
        now = now or _now()
        category = _category(cls)
        attr_text = attributes_text
        if attr_text == "" and attributes is not None and hasattr(attributes, "as_text"):
            attr_text = attributes.as_text()
        attr_json = _attributes_json(attributes)

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

        # --- vehicles: identity + location policy (settled decisions 1-3) ---
        if category == "vehicle":
            return self._decide_vehicle(
                direction, cls, now, fingerprint, region, attr_text, attr_json,
                guid, capture_count, is_static, span_hours,
            )

        # --- non-vehicles: existing behavior, unchanged --------------------
        if is_static:
            return Decision(SUPPRESS, f"static object ({cls}, {capture_count} captures over {span_hours:.1f}h)",
                           capture_count, True)
        if category == "person":
            return Decision(ALERT, "person", capture_count, False)
        if category == "animal":
            return Decision(FLAG, "animal", capture_count, False)
        return Decision(SUPPRESS, f"non-target class {cls}", capture_count, False)

    def _decide_vehicle(self, direction: str, cls: str, now: str,
                        fingerprint: bytes | None, region: str | None,
                        attr_text: str, attr_json: str, guid: str | None,
                        capture_count: int, is_static: bool,
                        span_hours: float) -> Decision:
        if region == "road":
            # Pass-by: note only, no alert (still capture make/model + lettering).
            return Decision(NOTE, f"vehicle pass-by ({cls})", capture_count, False,
                            tag=TAG_TRAFFIC, attributes_text=attr_text)

        if region == "driveway":
            # Identity-expectedness evaluation.
            if fingerprint is None:
                # Cannot identify — conservative: treat as novel.
                return Decision(ALERT, "unidentified vehicle (no fingerprint)",
                                capture_count, False, attributes_text=attr_text)
            entity_id, _ = self._resolve_entity(
                direction, fingerprint, attr_text, attr_json, now)
            key = guid or now
            # Capture the PRIOR last-presence time before recording this
            # sighting, so the hysteresis absence-gap measures the time since
            # the *previous* sighting (not zero, since we just recorded now).
            prior_last_ts = self._last_presence_ts(entity_id, direction)
            self.record_presence(entity_id, direction, key, now)
            return self._evaluate_expectedness(
                entity_id, direction, cls, now, attr_text, capture_count,
                prior_last_ts)

        # Unclassified region -> current novel-vehicle policy.
        if is_static:
            return Decision(SUPPRESS, f"static object ({cls}, {capture_count} captures over {span_hours:.1f}h)",
                           capture_count, True)
        return Decision(ALERT, "novel vehicle", capture_count, False,
                        attributes_text=attr_text)

    def _evaluate_expectedness(self, entity_id: str, direction: str, cls: str,
                               now: str, attr_text: str,
                               capture_count: int,
                               prior_last_ts: float | None = None) -> Decision:
        """Apply presence-ratio + hysteresis to decide expected vs unfamiliar.

        Hysteresis (P1-1): once an entity is "expected" it STAYS expected across
        ordinary detection gaps (occlusion, weather, a few missed captures) and
        only reverts to unfamiliar after being absent for >=
        ``entity_expected_absence_days``. Without this, a familiar vehicle at
        ~threshold presence flips between alert/note on every minor miss.
        """
        already_expected = self._entity_expected(entity_id)

        if already_expected:
            last_ts = (prior_last_ts
                       if prior_last_ts is not None
                       else self._last_presence_ts(entity_id, direction))
            absent_days = ((_ts(now) - last_ts) / 86400.0) if last_ts is not None else 0.0
            if absent_days < self.entity_expected_absence_days:
                return Decision(
                    NOTE,
                    f"expected vehicle ({cls}, presence {self.presence_ratio(entity_id, direction, now):.0%})",
                    capture_count, False, entity_id=entity_id,
                    tag=TAG_BACKGROUND, attributes_text=attr_text,
                )
            # Absent long enough -> drop expected, fall through to ratio check.
            self._set_entity_expected(entity_id, False)

        ratio = self.presence_ratio(entity_id, direction, now)
        if ratio >= self.entity_expected_presence_ratio:
            self._set_entity_expected(entity_id, True)
            return Decision(
                NOTE,
                f"expected vehicle ({cls}, presence {ratio:.0%})",
                capture_count, False, entity_id=entity_id,
                tag=TAG_BACKGROUND, attributes_text=attr_text,
            )
        label = attr_text or "vehicle"
        return Decision(
            ALERT,
            f"unfamiliar vehicle ({label}; presence {ratio:.0%})",
            capture_count, False, entity_id=entity_id,
            attributes_text=attr_text,
        )

    def _entity_expected(self, entity_id: str) -> bool:
        row = self._conn.execute(
            "SELECT expected FROM entities WHERE entity_id = ?", (entity_id,)
        ).fetchone()
        return bool(row and row[0])

    def _set_entity_expected(self, entity_id: str, expected: bool) -> None:
        self._conn.execute(
            "UPDATE entities SET expected = ? WHERE entity_id = ?",
            (1 if expected else 0, entity_id),
        )
        self._conn.commit()

    def _last_presence_ts(self, entity_id: str, direction: str) -> float | None:
        row = self._conn.execute(
            "SELECT MAX(ts) FROM entity_presence WHERE entity_id = ? AND direction = ?",
            (entity_id, direction),
        ).fetchone()
        return row[0] if row and row[0] is not None else None

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


def _attributes_json(attributes: object | None) -> str:
    if attributes is None:
        return "{}"
    if hasattr(attributes, "__dict__"):
        d = {k: v for k, v in attributes.__dict__.items()}
        return json.dumps(d)
    return "{}"


def _category(cls: str) -> str:
    from .detector import PERSON_CLASSES, VEHICLE_CLASSES, ANIMAL_CLASSES

    if cls in PERSON_CLASSES:
        return "person"
    if cls in VEHICLE_CLASSES:
        return "vehicle"
    if cls in ANIMAL_CLASSES:
        return "animal"
    return "other"
