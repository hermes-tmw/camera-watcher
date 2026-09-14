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

Per-location scope (2026-09-13 refactor): the engine is location-agnostic. One
codebase + one DB serve multiple sites (Taylor Hill Stealthcam, Wichita house,
...); every memory table carries a ``scope`` column and every composite PK
includes it, so identity/state/config can never leak across sites. A Memory
instance is bound to ONE scope at construction (``Memory(..., scope=...)`` —
required, no default) and all of its reads/writes are scope-filtered. Identity
hashing salts the scope into the entity_id, so byte-identical fingerprints in
two scopes are distinct entities by construction.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("stealthcam.memory")

# Alert decision outcomes
ALERT = "alert"
FLAG = "flag"
SUPPRESS = "suppress"
NOTE = "note"

# Dashboard note tags (location policy).
TAG_TRAFFIC = "traffic"
TAG_BACKGROUND = "background"

_MEMORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS seen_guids (
    scope TEXT NOT NULL,
    guid TEXT NOT NULL,
    seen_at TEXT NOT NULL,
    PRIMARY KEY (scope, guid)
);
CREATE TABLE IF NOT EXISTS memory_observations (
    scope TEXT NOT NULL,
    id INTEGER PRIMARY KEY,
    direction TEXT NOT NULL,
    bbox TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    capture_count INTEGER NOT NULL DEFAULT 1,
    classes_seen TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS decisions (
    scope TEXT NOT NULL,
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
    scope TEXT NOT NULL,
    id INTEGER PRIMARY KEY,
    decision_id INTEGER,
    guid TEXT NOT NULL,
    subimage INTEGER NOT NULL,
    human_label TEXT NOT NULL,
    flagged_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS deferred_captures (
    scope TEXT NOT NULL,
    guid TEXT NOT NULL,
    deferral_count INTEGER NOT NULL DEFAULT 1,
    alerted INTEGER NOT NULL DEFAULT 0,
    first_deferred_at TEXT NOT NULL,
    last_deferred_at TEXT NOT NULL,
    PRIMARY KEY (scope, guid)
);
-- Entity identity + expected-presence baseline (2026-09-13).
-- entity_id is salted with the scope (see _new_entity_id): byte-identical
-- fingerprints in two scopes hash to different ids, so no cross-scope
-- collision or surfacing is possible even before the PK is consulted.
CREATE TABLE IF NOT EXISTS entities (
    scope TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    direction TEXT NOT NULL,
    region TEXT,
    fingerprint BLOB NOT NULL,
    model TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    capture_count INTEGER NOT NULL DEFAULT 1,
    attributes TEXT NOT NULL DEFAULT '{}',
    expected INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (scope, entity_id)
);
CREATE TABLE IF NOT EXISTS entity_presence (
    scope TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    direction TEXT NOT NULL,
    guid TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    ts REAL NOT NULL,
    PRIMARY KEY (scope, entity_id, direction, guid)
);
CREATE TABLE IF NOT EXISTS monitored_captures (
    scope TEXT NOT NULL,
    direction TEXT NOT NULL,
    region TEXT,
    guid TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    ts REAL NOT NULL,
    is_day INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (scope, direction, guid)
);
CREATE INDEX IF NOT EXISTS idx_decisions_scope_guid
    ON decisions (scope, guid);
CREATE INDEX IF NOT EXISTS idx_entity_presence_scope_ts
    ON entity_presence (scope, entity_id, direction, ts);
CREATE INDEX IF NOT EXISTS idx_monitored_scope_ts
    ON monitored_captures (scope, ts, is_day);
CREATE INDEX IF NOT EXISTS idx_memory_obs_scope_dir
    ON memory_observations (scope, direction);
"""

# Rebuilt-by-migration tables carry ``DEFAULT 'taylor_hill'`` on scope so the
# OLD worker binary (INSERTs without scope) can keep writing during the deploy
# cutover window; fresh tables above have NO default (loud on omission).
# migrate.py owns the difference; Memory just executes the schema.
MEMORY_SCHEMA = _MEMORY_SCHEMA

_SCHEMA = _MEMORY_SCHEMA


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
                 entity_expected_absence_days: float = 2.0,
                 scope: str | None = None):
        if not scope:
            raise ValueError(
                "Memory requires an explicit scope (location name, e.g. "
                "'taylor_hill') — a scopeless Memory would blur identity "
                "state across sites; pass scope= from the caller's config.")
        self.db_path = str(db_path)
        self.scope = scope
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
        # Did THIS open create the file? Captured before connect (which
        # materializes a 0-byte file) and passed to migrate_db_open — with
        # WAL enabled at open, the file is no longer 0-byte by migration
        # time, so the size heuristic alone can't distinguish fresh from
        # legacy anymore (the caller's knowledge can).
        db_is_new = not Path(self.db_path).is_file()
        self._conn = sqlite3.connect(self.db_path)
        # Multi-writer readiness (2026-09-13, wichita ingestor): two scope
        # processes (taylor_hill worker + wichita motion ingestor) share this
        # one DB file. DELETE-journal mode serializes badly and fails readers
        # during every writer's exclusive lock; WAL gives writer/reader
        # concurrency with SQLITE_BUSY backoff via busy_timeout. Constraint:
        # the file lives on NFS, where WAL is only safe when ALL accessors go
        # through ONE NFS client host — true for every deployed accessor
        # (all containers run on mira), and the deploy convention keeps the
        # CLI/backfill on mira too. Refuse to run if WAL can't be enabled:
        # silently staying in DELETE mode would re-introduce the "database is
        # locked" alert-storm exactly when the second scope goes live.
        # ':memory:' DBs are exempt BY CONSTRUCTION — they are
        # single-connection (no second process can attach), so the multi-
        # writer hazard WAL guards against does not exist, and SQLite has no
        # WAL mode for them anyway.
        self._conn.execute("PRAGMA busy_timeout = 10000")
        if self.db_path != ":memory:":
            try:
                mode = self._conn.execute(
                    "PRAGMA journal_mode=WAL").fetchone()[0]
            except sqlite3.OperationalError as e:
                self._conn.close()
                raise RuntimeError(
                    f"{self.db_path}: cannot enable WAL journal mode needed "
                    f"for multi-scope concurrent access: {e}") from e
            if str(mode).lower() != "wal":
                self._conn.close()
                raise RuntimeError(
                    f"{self.db_path}: journal_mode=WAL returned {mode!r} — the "
                    f"filesystem does not support WAL. Multi-scope concurrent "
                    f"access requires WAL (or a move off this filesystem).")
        # Migrate BEFORE executing the schema: on a legacy DB the scoped
        # indexes in _SCHEMA reference a scope column the old tables don't
        # have yet. migrate_db_open takes the one-time pre-migration backup
        # (deploy path: the DB must not be mutated in place without a
        # rollback copy), then rebuilds legacy tables (armor defaults),
        # creates a fresh DB's scoped schema, and no-ops on scoped DBs.
        # Every per-table rebuild is one explicit transaction, so a crash
        # mid-rebuild rolls back instead of stranding the table half-done.
        from .migrate import (migrate_db_open, assert_no_orphan_rebuild_tables,
                              SCHEMA_EPOCH_USER_VERSION)

        # Fast path (P2): an already-scoped DB stamped with the current
        # epoch skips re-scanning sqlite_master on every boot. The epoch is
        # bumped whenever the scoped schema/migration changes shape; a DB
        # stamped with a DIFFERENT epoch still gets the full check.
        cur_epoch = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if cur_epoch != SCHEMA_EPOCH_USER_VERSION:
            report = migrate_db_open(self.db_path, self._conn,
                                     db_is_new=db_is_new)
            if report["status"] != "already-scoped":
                log.info("pipeline DB migrated at open: %s (%d tables)",
                         report["status"], len(report.get("tables", {})))
            self._conn.execute(
                f"PRAGMA user_version = {SCHEMA_EPOCH_USER_VERSION}")
        else:
            # Fast path still guards stranded rebuild artifacts (unconditional
            # — a user_version-stamped DB can still be stranded by an old
            # crash AFTER it was stamped, and the guard is one query).
            assert_no_orphan_rebuild_tables(self._conn)
        self._conn.executescript(_SCHEMA)
        self._backfill_columns()
        self._conn.commit()

    def _backfill_columns(self) -> None:
        """Column additions for tables that predate a column (idempotent).

        ``CREATE TABLE IF NOT EXISTS`` does not add columns to a table that
        already exists, so column additions are applied here idempotently.
        Legacy (pre-scope) tables are rebuilt with the scope dimension by
        migrate.migrate_scoped_db (called from __init__ before this).
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
        cur = self._conn.execute(
            "SELECT 1 FROM seen_guids WHERE scope = ? AND guid = ?",
            (self.scope, guid))
        return cur.fetchone() is not None

    def mark_seen(self, guid: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO seen_guids (scope, guid, seen_at) VALUES (?, ?, ?)",
            (self.scope, guid, _now()),
        )
        # A capture that was previously deferred is now successfully seen —
        # drop its deferral row so it no longer counts toward the alert
        # threshold or the requeue backfill.
        self._conn.execute(
            "DELETE FROM deferred_captures WHERE scope = ? AND guid = ?",
            (self.scope, guid))
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
            "SELECT deferral_count FROM deferred_captures "
            "WHERE scope = ? AND guid = ?", (self.scope, guid)
        )
        row = cur.fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO deferred_captures "
                "(scope, guid, deferral_count, alerted, first_deferred_at, last_deferred_at) "
                "VALUES (?, ?, 1, 0, ?, ?)",
                (self.scope, guid, now, now),
            )
            count = 1
        else:
            count = row[0] + 1
            self._conn.execute(
                "UPDATE deferred_captures SET deferral_count = ?, last_deferred_at = ? "
                "WHERE scope = ? AND guid = ?",
                (count, now, self.scope, guid),
            )
        self._conn.commit()
        return count

    def is_deferred_alerted(self, guid: str) -> bool:
        cur = self._conn.execute(
            "SELECT alerted FROM deferred_captures WHERE scope = ? AND guid = ?",
            (self.scope, guid)
        )
        row = cur.fetchone()
        return bool(row and row[0])

    def mark_deferred_alerted(self, guid: str) -> None:
        self._conn.execute(
            "UPDATE deferred_captures SET alerted = 1 "
            "WHERE scope = ? AND guid = ?",
            (self.scope, guid)
        )
        self._conn.commit()

    def clear_deferral(self, guid: str) -> None:
        """Drop a capture's deferral row once it has been successfully seen."""
        self._conn.execute(
            "DELETE FROM deferred_captures WHERE scope = ? AND guid = ?",
            (self.scope, guid))
        self._conn.commit()

    def deferred_guids(self) -> list[str]:
        """Return the guids currently tracked as deferred (never marked seen)."""
        rows = self._conn.execute(
            "SELECT guid FROM deferred_captures WHERE scope = ? "
            "ORDER BY last_deferred_at",
            (self.scope,),
        ).fetchall()
        return [r[0] for r in rows]

    # -- object matching (position persistence) ------------------------------

    def _find_match(self, direction: str, bbox: tuple[float, float, float, float]):
        """Return the best-matching observation row (id, first_seen_at, count,
        classes_seen, bbox) in this scope+direction with IoU >= threshold,
        else None."""
        rows = self._conn.execute(
            "SELECT id, first_seen_at, capture_count, classes_seen, bbox "
            "FROM memory_observations WHERE scope = ? AND direction = ?",
            (self.scope, direction),
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
        """Return the entity_id of a same-model entity in this scope's
        `direction` whose stored fingerprint matches `fingerprint`, else None."""
        model = getattr(self.fingerprinter, "model", "")
        rows = self._conn.execute(
            "SELECT entity_id, fingerprint FROM entities "
            "WHERE scope = ? AND direction = ? AND model = ?",
            (self.scope, direction, model),
        ).fetchall()
        for entity_id, stored in rows:
            if self.fingerprinter.matches(fingerprint, stored):
                return entity_id
        return None

    def _new_entity_id(self, direction: str, fingerprint: bytes) -> str:
        # Salt the hash with SCOPE and direction so a byte-identical
        # fingerprint never collides across scopes (Tom's core guarantee:
        # Taylor Hill and Wichita identity can never leak) nor across mapped
        # directions (entity matching is direction-scoped; a same-bytes crop
        # arriving in a second direction would otherwise INSERT the same hash
        # and raise sqlite3.IntegrityError out of decide()).
        digest = hashlib.sha256(
            self.scope.encode("utf-8") + b"|"
            + direction.encode("utf-8") + b"|" + fingerprint
        ).hexdigest()[:16]
        return f"e-{digest}"

    def _resolve_entity(self, direction: str, fingerprint: bytes,
                        attributes_text: str, attributes_json: str,
                        now: str) -> tuple[str, bool]:
        """Return (entity_id, created) for a fingerprint in this scope's
        `direction`.

        Matches an existing entity by fingerprint similarity (same model); if
        none, creates a new entity keyed on the scope-salted fingerprint hash.
        Updates the entity's last_seen_at / capture_count / attributes on
        every sighting.
        """
        model = getattr(self.fingerprinter, "model", "")
        entity_id = self._match_entity(direction, fingerprint)
        created = False
        if entity_id is None:
            entity_id = self._new_entity_id(direction, fingerprint)
            created = True
            self._conn.execute(
                "INSERT INTO entities (scope, entity_id, direction, fingerprint, model, "
                "first_seen_at, last_seen_at, capture_count, attributes) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)",
                (self.scope, entity_id, direction, fingerprint, model, now, now,
                 attributes_json),
            )
        else:
            self._conn.execute(
                "UPDATE entities SET last_seen_at = ?, capture_count = capture_count + 1 "
                "WHERE scope = ? AND entity_id = ?",
                (now, self.scope, entity_id),
            )
            # Enrich attributes when new info appears (never overwrite a known
            # make/model with a blank one).
            if attributes_text:
                self._merge_attributes(entity_id, attributes_json)
        return entity_id, created

    def _merge_attributes(self, entity_id: str, attributes_json: str) -> None:
        cur = self._conn.execute(
            "SELECT attributes FROM entities WHERE scope = ? AND entity_id = ?",
            (self.scope, entity_id)
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
            "UPDATE entities SET attributes = ? WHERE scope = ? AND entity_id = ?",
            (json.dumps(merged), self.scope, entity_id),
        )

    def record_presence(self, entity_id: str, direction: str, guid: str,
                        captured_at: str) -> None:
        """Record that `entity_id` appeared at `direction` during capture `guid`
        (within this Memory's scope)."""
        ts = _ts(captured_at)
        self._conn.execute(
            "INSERT OR IGNORE INTO entity_presence "
            "(scope, entity_id, direction, guid, captured_at, ts) VALUES (?, ?, ?, ?, ?, ?)",
            (self.scope, entity_id, direction, guid, captured_at, ts),
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
            "(scope, direction, region, guid, captured_at, ts, is_day) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (self.scope, direction, region, guid, captured_at, ts,
             1 if is_day else 0),
        )
        self._conn.commit()

    def presence_ratio(self, entity_id: str, direction: str,
                       now: str) -> float:
        """Fraction of monitored DAY captures (rolling window) the entity appeared
        at `direction`. 0.0 when there is no monitored day history yet."""
        cutoff = _ts(now) - self.entity_baseline_window_days * 86400.0
        num_pres = self._conn.execute(
            "SELECT COUNT(*) FROM entity_presence "
            "WHERE scope = ? AND entity_id = ? AND direction = ? AND ts >= ?",
            (self.scope, entity_id, direction, cutoff),
        ).fetchone()[0]
        num_mon = self._conn.execute(
            "SELECT COUNT(*) FROM monitored_captures "
            "WHERE scope = ? AND direction = ? AND ts >= ? AND is_day = 1",
            (self.scope, direction, cutoff),
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
        - ``region``     — the resolved spatial region ("driveway"/"road"/
          "road_edge"/None; road_edge = near the road-band edge, FLAGged).
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
                "(scope, direction, bbox, first_seen_at, last_seen_at, capture_count, classes_seen) "
                "VALUES (?, ?, ?, ?, ?, 1, ?)",
                (self.scope, direction, json.dumps(list(bbox)), now, now,
                 json.dumps([cls])),
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
                "classes_seen = ? WHERE scope = ? AND id = ?",
                (capture_count, now, json.dumps(classes), self.scope, oid),
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

        if region == "road_edge":
            # Near-band (Percy PR#43 P2-3): the geometry is not confident the
            # detection is road vs property — a firm road NOTE risks a silent
            # missed visitor (driveway->road), a firm driveway ALERT risks a
            # noisy pass-by alert (road->driveway). FLAG for human review
            # instead; dashboard-only (FLAG never posts to Mattermost).
            return Decision(FLAG, f"vehicle near road band ({cls})", capture_count,
                            False, tag=TAG_TRAFFIC, attributes_text=attr_text)

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
            "SELECT expected FROM entities WHERE scope = ? AND entity_id = ?",
            (self.scope, entity_id)
        ).fetchone()
        return bool(row and row[0])

    def _set_entity_expected(self, entity_id: str, expected: bool) -> None:
        self._conn.execute(
            "UPDATE entities SET expected = ? WHERE scope = ? AND entity_id = ?",
            (1 if expected else 0, self.scope, entity_id),
        )
        self._conn.commit()

    def _last_presence_ts(self, entity_id: str, direction: str) -> float | None:
        row = self._conn.execute(
            "SELECT MAX(ts) FROM entity_presence "
            "WHERE scope = ? AND entity_id = ? AND direction = ?",
            (self.scope, entity_id, direction),
        ).fetchone()
        return row[0] if row and row[0] is not None else None

    def count_entities(self) -> int:
        """Number of entities in this scope (baseline-seeding sanity check)."""
        return int(self._conn.execute(
            "SELECT COUNT(*) FROM entities WHERE scope = ?",
            (self.scope,)).fetchone()[0])

    def count_monitored(self) -> int:
        """Number of monitored-capture rows in this scope (same check)."""
        return int(self._conn.execute(
            "SELECT COUNT(*) FROM monitored_captures WHERE scope = ?",
            (self.scope,)).fetchone()[0])

    # -- decisions / feedback (feedback-loop data) ---------------------------

    def record_decision(self, guid: str, subimage: int, direction: str, cls: str,
                        confidence: float, bbox: tuple[float, float, float, float],
                        description: str, decision: Decision) -> int:
        cur = self._conn.execute(
            "INSERT INTO decisions (scope, guid, subimage, direction, class, confidence, "
            "bbox, description, decision, reason, decided_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (self.scope, guid, subimage, direction, cls, confidence,
             json.dumps(list(bbox)), description, decision.decision,
             decision.reason, _now()),
        )
        self._conn.commit()
        return cur.lastrowid

    def record_feedback(self, decision_id: int, guid: str, subimage: int,
                        human_label: str) -> None:
        self._conn.execute(
            "INSERT INTO feedback (scope, decision_id, guid, subimage, human_label, flagged_at) "
            "VALUES (?,?,?,?,?,?)",
            (self.scope, decision_id, guid, subimage, human_label, _now()),
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
