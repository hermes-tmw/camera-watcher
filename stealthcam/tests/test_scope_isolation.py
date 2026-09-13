"""Tests for the per-location scope dimension (trust-boundary guarantee).

The refactor's core promise (Tom, 2026-09-13): Taylor Hill and Wichita house
are SEPARATE scopes sharing one engine and one DB, with NO cross-scope leak of
identity, state, or config — enforced structurally, not by discipline:

1. Scope isolation: the same fingerprint under two scopes must NOT collide,
   must not surface as the same entity, and both must persist independently
   (entity rows, presence, seen-set, decisions).
2. Migration: legacy (pre-scope) tables are rebuilt losslessly and additively
   (every row stamped scope='taylor_hill', counts preserved, PKs now include
   scope), and re-running is a no-op (idempotent).
3. Cutover armor: rebuilt tables keep ``scope DEFAULT '<stamp>'`` so the OLD
   worker binary (INSERTs that omit scope) keeps writing into the right scope
   during the deploy window.
4. Per-scope config: scope_setting() resolves scopes.<scope>.<key> over the
   legacy flat attrs, and an unknown key in a scopes block raises at load.
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from stealthcam.memory import Memory, ALERT, NOTE
from stealthcam.migrate import migrate_scoped_db, migrate_file, DEFAULT_STAMP
from stealthcam.config import Config, load_config, _validate_scopes


def _iso(dt: datetime) -> str:
    return dt.isoformat()


_T0 = datetime(2026, 9, 10, 12, 0, 0, tzinfo=timezone.utc)


class _TrivialFingerprinter:
    model = "trivial"

    def fingerprint(self, crop):
        return b"x"

    def matches(self, a: bytes, b: bytes) -> bool:
        return a == b


def _mk(db: str, scope: str) -> Memory:
    return Memory(db, fingerprinter=_TrivialFingerprinter(),
                  entity_expected_presence_ratio=0.5,
                  entity_baseline_window_days=7.0,
                  scope=scope)


def _same_fingerprint_two_scopes(db_path: Path):
    """Feed the identical detection (guid + fingerprint) into both scopes.

    Returns both Decision objects. The guid is shared deliberately: the
    seen-guid dedup is per-scope, so the SAME capture processed by two
    location engines must be seen by each independently.
    """
    bbox = (100.0, 100.0, 300.0, 300.0)
    guid = "019f-same-capture-guid"
    th = _mk(str(db_path), "taylor_hill")
    wi = _mk(str(db_path), "wichita")
    d_th = th.decide("East", "car", bbox, now=_iso(_T0),
                     fingerprint=b"same-bytes", region="driveway", guid=guid)
    d_wi = wi.decide("East", "car", bbox, now=_iso(_T0),
                     fingerprint=b"same-bytes", region="driveway", guid=guid)
    # The worker marks a capture seen after processing it (decide() itself
    # does not); both engines processed the SAME capture guid.
    th.mark_seen(guid)
    wi.mark_seen(guid)
    # And the worker records a decision row per detection, per scope.
    for mem, d in ((th, d_th), (wi, d_wi)):
        mem.record_decision(guid, 1, "East", "car", 0.9, bbox, "", d)
    return th, wi, d_th, d_wi, guid


class TestScopeIsolation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "pipeline.db"

    def tearDown(self):
        self.tmp.cleanup()

    def test_shared_engine_instance_requires_scope(self):
        with self.assertRaises(ValueError):
            Memory(str(self.db), fingerprinter=_TrivialFingerprinter())

    def test_same_fingerprint_two_scopes_no_collision_no_surfacing(self):
        th, wi, d_th, d_wi, _ = _same_fingerprint_two_scopes(self.db)

        # Distinct entities: no PK collision, no identity surfacing.
        self.assertEqual(d_th.decision, ALERT)
        self.assertEqual(d_wi.decision, ALERT)
        self.assertIsNotNone(d_th.entity_id)
        self.assertIsNotNone(d_wi.entity_id)
        self.assertNotEqual(d_th.entity_id, d_wi.entity_id,
                            "byte-identical fingerprints in two scopes must "
                            "hash to different entity ids (scope-salted)")

        # Each scope persists exactly one entity and one seen-guid; querying
        # one scope never returns the other's rows.
        conn = sqlite3.connect(str(self.db))
        for scope, mem in (("taylor_hill", th), ("wichita", wi)):
            ents = conn.execute(
                "SELECT entity_id FROM entities WHERE scope = ?",
                (scope,)).fetchall()
            self.assertEqual([e[0] for e in ents], [d_th.entity_id if scope == "taylor_hill" else d_wi.entity_id])
            self.assertEqual(conn.execute(
                "SELECT COUNT(*) FROM entity_presence WHERE scope = ?",
                (scope,)).fetchone()[0], 1)
            self.assertEqual(conn.execute(
                "SELECT COUNT(*) FROM seen_guids WHERE scope = ?",
                (scope,)).fetchone()[0], 1)
            self.assertEqual(conn.execute(
                "SELECT COUNT(*) FROM decisions WHERE scope = ?",
                (scope,)).fetchone()[0], 1)
            # Engine-level reads are scope-filtered too.
            self.assertTrue(mem.has_seen("019f-same-capture-guid"))
            self.assertIsNone(
                conn.execute(
                    "SELECT 1 FROM entities WHERE scope = ? AND entity_id = ?",
                    (scope, d_wi.entity_id if scope == "taylor_hill"
                     else d_th.entity_id)).fetchone())
        conn.close()
        th.close()
        wi.close()

    def test_same_guid_independent_seen_sets(self):
        # A shared capture guid is deduped per scope: marking it seen in one
        # scope does not mark it seen in the other (the other location's
        # engine must still process the capture).
        th, wi, _, _, guid = _same_fingerprint_two_scopes(self.db)
        self.assertTrue(th.has_seen(guid))
        self.assertTrue(wi.has_seen(guid))
        wi.clear_deferral("any-guid")  # scope-filtered no-op must not raise
        # Reverse check with a guid only seen in taylor_hill.
        th.mark_seen("only-th-guid")
        self.assertTrue(th.has_seen("only-th-guid"))
        self.assertFalse(wi.has_seen("only-th-guid"))
        th.close()
        wi.close()

    def test_presence_baselines_are_per_scope(self):
        # Seed an 8/10 presence ratio for the same fingerprint in taylor_hill
        # only; wichita must still evaluate as unfamiliar (ratio 0) for its
        # own entity.
        th, wi, d_th, _, _ = _same_fingerprint_two_scopes(self.db)
        for k in range(1, 8):
            t = _T0 + timedelta(hours=k)
            th.decide("East", "car", (100.0, 100.0, 300.0, 300.0),
                      now=_iso(t), fingerprint=b"same-bytes",
                      region="driveway", guid=f"th-guid-{k}")
        for k in range(10):
            t = _T0 + timedelta(hours=k)
            th.record_monitored("East", "driveway", f"th-mon-{k}", _iso(t))
        # taylor_hill: ratio 8/10 -> expected -> note.
        d = th.decide("East", "car", (100.0, 100.0, 300.0, 300.0),
                      now=_iso(_T0 + timedelta(hours=8)),
                      fingerprint=b"same-bytes", region="driveway",
                      guid="th-guid-8")
        self.assertEqual(d.decision, NOTE)
        # wichita: same fingerprint, zero baseline -> alert (its own entity).
        d2 = wi.decide("East", "car", (100.0, 100.0, 300.0, 300.0),
                       now=_iso(_T0 + timedelta(hours=8)),
                       fingerprint=b"same-bytes", region="driveway",
                       guid="wi-guid-1")
        self.assertEqual(d2.decision, ALERT)
        # Sanity: the two scopes' entities are distinct ids (scope-salted).
        self.assertNotEqual(d2.entity_id, d_th.entity_id)
        th.close()
        wi.close()

    def test_expectedness_lifecycle_is_per_scope(self):
        # expected=1 set in one scope must not leak to the other scope's
        # same-bytes entity (manual lever UPDATEs are scope-filtered).
        th, wi, d_th, d_wi, _ = _same_fingerprint_two_scopes(self.db)
        conn = sqlite3.connect(str(self.db))
        conn.execute(
            "UPDATE entities SET expected = 1 WHERE scope = 'taylor_hill'")
        conn.commit()
        self.assertTrue(th._entity_expected(d_th.entity_id))
        self.assertFalse(wi._entity_expected(d_wi.entity_id))
        conn.close()
        th.close()
        wi.close()


class TestMigration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _legacy_db(self) -> Path:
        """Build a DB with the EXACT pre-scope live schema + row counts."""
        path = self.dir / "legacy.db"
        conn = sqlite3.connect(str(path))
        conn.executescript("""
CREATE TABLE seen_guids (guid TEXT PRIMARY KEY, seen_at TEXT NOT NULL);
CREATE TABLE memory_observations (id INTEGER PRIMARY KEY, direction TEXT NOT NULL,
    bbox TEXT NOT NULL, first_seen_at TEXT NOT NULL, last_seen_at TEXT NOT NULL,
    capture_count INTEGER NOT NULL DEFAULT 1, classes_seen TEXT NOT NULL DEFAULT '[]');
CREATE TABLE decisions (id INTEGER PRIMARY KEY, guid TEXT NOT NULL, subimage INTEGER NOT NULL,
    direction TEXT NOT NULL, class TEXT NOT NULL, confidence REAL NOT NULL, bbox TEXT NOT NULL,
    description TEXT, decision TEXT NOT NULL, reason TEXT NOT NULL, decided_at TEXT NOT NULL);
CREATE TABLE feedback (id INTEGER PRIMARY KEY, decision_id INTEGER, guid TEXT NOT NULL,
    subimage INTEGER NOT NULL, human_label TEXT NOT NULL, flagged_at TEXT NOT NULL);
CREATE TABLE deferred_captures (guid TEXT PRIMARY KEY,
    deferral_count INTEGER NOT NULL DEFAULT 1, alerted INTEGER NOT NULL DEFAULT 0,
    first_deferred_at TEXT NOT NULL, last_deferred_at TEXT NOT NULL);
CREATE TABLE entities (entity_id TEXT PRIMARY KEY, direction TEXT NOT NULL, region TEXT,
    fingerprint BLOB NOT NULL, model TEXT NOT NULL, first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL, capture_count INTEGER NOT NULL DEFAULT 1,
    attributes TEXT NOT NULL DEFAULT '{}', expected INTEGER NOT NULL DEFAULT 0);
CREATE TABLE entity_presence (entity_id TEXT NOT NULL, direction TEXT NOT NULL,
    guid TEXT NOT NULL, captured_at TEXT NOT NULL, ts REAL NOT NULL,
    PRIMARY KEY (entity_id, direction, guid));
CREATE TABLE monitored_captures (direction TEXT NOT NULL, region TEXT, guid TEXT NOT NULL,
    captured_at TEXT NOT NULL, ts REAL NOT NULL, is_day INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (direction, guid));
""")
        counts = {"seen_guids": 37, "memory_observations": 12, "decisions": 44,
                  "feedback": 2, "deferred_captures": 3, "entities": 4,
                  "entity_presence": 9, "monitored_captures": 21}
        for i in range(counts["seen_guids"]):
            conn.execute("INSERT INTO seen_guids VALUES (?, ?)",
                         (f"g{i}", _iso(_T0)))
        for i in range(counts["memory_observations"]):
            conn.execute(
                "INSERT INTO memory_observations (direction, bbox, first_seen_at, "
                "last_seen_at) VALUES (?, ?, ?, ?)",
                ("East", f"[{i},1,2,3]", _iso(_T0), _iso(_T0)))
        for i in range(counts["decisions"]):
            conn.execute(
                "INSERT INTO decisions (guid, subimage, direction, class, confidence, "
                "bbox, decision, reason, decided_at) VALUES (?, 1, 'East', 'car', "
                "0.9, '[1,2,3,4]', 'alert', 'r', ?)", (f"g{i % 37}", _iso(_T0)))
        for i in range(counts["feedback"]):
            conn.execute(
                "INSERT INTO feedback (decision_id, guid, subimage, human_label, "
                "flagged_at) VALUES (1, 'g0', 1, 'good', ?)", (_iso(_T0),))
        for i in range(counts["deferred_captures"]):
            conn.execute(
                "INSERT INTO deferred_captures VALUES (?, 1, 0, ?, ?)",
                (f"d{i}", _iso(_T0), _iso(_T0)))
        for i in range(counts["entities"]):
            conn.execute(
                "INSERT INTO entities (entity_id, direction, fingerprint, model, "
                "first_seen_at, last_seen_at) VALUES (?, 'East', ?, 'm', ?, ?)",
                (f"e{i}", f"fp{i}".encode(), _iso(_T0), _iso(_T0)))
        for i in range(counts["entity_presence"]):
            conn.execute(
                "INSERT INTO entity_presence VALUES (?, 'East', ?, ?, ?)",
                (f"e{i % 4}", f"g{i}", _iso(_T0), 1.0))
        for i in range(counts["monitored_captures"]):
            conn.execute(
                "INSERT INTO monitored_captures (direction, guid, captured_at, ts) "
                "VALUES ('East', ?, ?, ?)", (f"m{i}", _iso(_T0), 1.0))
        conn.commit()
        conn.close()
        return path, counts

    def _counts(self, conn) -> dict:
        return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in ("seen_guids", "memory_observations", "decisions",
                          "feedback", "deferred_captures", "entities",
                          "entity_presence", "monitored_captures")}

    def test_legacy_migration_lossless_and_stamped(self):
        path, counts = self._legacy_db()
        conn = sqlite3.connect(str(path))
        report = migrate_scoped_db(conn)
        self.assertEqual(report["status"], "migrated")
        conn.commit()
        # Every row preserved.
        self.assertEqual(self._counts(conn), counts)
        # Every row stamped with the Taylor Hill scope.
        for t in counts:
            self.assertEqual(
                conn.execute(
                    f"SELECT COUNT(*) FROM {t} WHERE scope = ?",
                    (DEFAULT_STAMP,)).fetchone()[0],
                counts[t],
                f"{t}: all legacy rows must carry scope='{DEFAULT_STAMP}'")
            self.assertEqual(
                conn.execute(f"SELECT COUNT(*) FROM {t} WHERE scope IS NULL")
                .fetchone()[0], 0)
        # Composite PKs include scope, scope-first.
        for t, pk_len in (("seen_guids", 2), ("entities", 2),
                          ("entity_presence", 4), ("monitored_captures", 3),
                          ("deferred_captures", 2)):
            info = conn.execute(f"PRAGMA table_info({t})").fetchall()
            pk = [r[1] for r in sorted((r for r in info if r[5] > 0),
                                       key=lambda r: r[5])]
            self.assertEqual(pk[0], "scope", f"{t}: PK must start with scope")
            self.assertEqual(len(pk), pk_len, f"{t}: PK shape")
        conn.close()

    def test_migration_idempotent(self):
        path, counts = self._legacy_db()
        conn = sqlite3.connect(str(path))
        r1 = migrate_scoped_db(conn)
        conn.commit()
        self.assertEqual(r1["status"], "migrated")
        r2 = migrate_scoped_db(conn)
        conn.commit()
        self.assertEqual(r2["status"], "already-scoped")
        self.assertEqual(self._counts(conn), counts, "re-run must not change data")
        conn.close()

    def test_migrated_db_still_has_default_scope_armor(self):
        # The rebuilt tables keep DEFAULT '<stamp>' on scope so the OLD worker
        # (INSERTs omitting scope) survives the cutover window.
        path, _ = self._legacy_db()
        migrate_file(path, backup=False)
        conn = sqlite3.connect(str(path))
        for t in ("seen_guids", "entities", "entity_presence",
                  "monitored_captures", "deferred_captures"):
            dflt = next((r[4] for r in conn.execute(f"PRAGMA table_info({t})")
                         if r[1] == "scope"), None)
            # PRAGMA returns the default as the raw SQL literal (quoted).
            self.assertEqual(
                (dflt or "").strip("'"), DEFAULT_STAMP,
                f"{t}: rebuilt table must keep scope DEFAULT armor")
        # And the legacy INSERT shape really works.
        conn.execute("INSERT INTO seen_guids (guid, seen_at) VALUES ('x', 'now')")
        conn.commit()
        self.assertEqual(conn.execute(
            "SELECT scope FROM seen_guids WHERE guid = 'x'").fetchone()[0],
            DEFAULT_STAMP)
        conn.close()

    def test_fresh_db_builds_clean_empty_scopes(self):
        # A brand-new DB gets the scoped schema with NO legacy default: new
        # code must supply scope explicitly (omission is loud).
        path = self.dir / "fresh.db"
        Memory(str(path), fingerprinter=_TrivialFingerprinter(),
               scope="wichita").close()
        conn = sqlite3.connect(str(path))
        self.assertTrue(conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='entities'").fetchone())
        dflt = next((r[4] for r in conn.execute("PRAGMA table_info(entities)")
                     if r[1] == "scope"), None)
        self.assertIsNone(dflt, "fresh schema must NOT carry the legacy default")
        conn.close()

    def test_legacy_rows_reachable_through_new_engine(self):
        # After migration, the new engine (scope=taylor_hill) sees the legacy
        # rows: an already-seen guid stays seen; a legacy entity matches.
        path, _ = self._legacy_db()
        migrate_file(path, backup=False)
        # Live entities carry the model of the fingerprinter that produced
        # them; use the trivial fingerprinter's model for the fixture.
        conn = sqlite3.connect(str(path))
        conn.execute("UPDATE entities SET model = 'trivial'")
        conn.commit()
        conn.close()
        mem = _mk(str(path), "taylor_hill")
        self.assertTrue(mem.has_seen("g0"))
        self.assertTrue(mem.has_seen("g36"))
        # The legacy entity e0 lives in East with fingerprint fp0; a matching
        # sighting resolves to THAT entity (not a new one).
        d = mem.decide("East", "car", (5.0, 5.0, 50.0, 50.0),
                       now=_iso(_T0 + timedelta(hours=1)),
                       fingerprint=b"fp0", region="driveway", guid="new-g")
        self.assertEqual(d.entity_id, "e0")
        mem.close()

    def test_memory_open_migrates_legacy_db_in_place(self):
        # The Memory constructor itself performs the migration (deploy path:
        # first boot after upgrade migrates on open).
        path, counts = self._legacy_db()
        mem = _mk(str(path), "taylor_hill")
        self.assertTrue(mem.has_seen("g1"))
        mem.close()
        conn = sqlite3.connect(str(path))
        self.assertEqual(self._counts(conn), counts)
        conn.close()


class TestScopeConfig(unittest.TestCase):
    def test_default_scope_is_taylor_hill(self):
        cfg = Config()
        self.assertEqual(cfg.memory_scope, "taylor_hill")
        self.assertEqual(cfg.scopes, {})

    def test_scope_setting_falls_back_to_flat_attrs(self):
        cfg = Config()
        cfg.direction_regions = "west=driveway"
        cfg.detector_night_conf_threshold = 0.75
        self.assertEqual(cfg.scope_setting("taylor_hill", "direction_regions", ""),
                         "west=driveway")
        self.assertEqual(
            cfg.scope_setting("taylor_hill", "detector_night_conf_threshold", 0.6),
            0.75)
        # A scope with no block falls back too.
        self.assertEqual(cfg.scope_setting("wichita", "direction_regions", ""),
                         "west=driveway")

    def test_scope_block_overrides_without_touching_other_scopes(self):
        cfg = Config()
        cfg.direction_regions = "west=driveway"  # Taylor live value
        cfg.scopes = {
            "taylor_hill": {"entity_expected_presence_ratio": 0.5},
            "wichita": {"direction_regions": "north=road,east=driveway",
                        "entity_expected_presence_ratio": 0.6},
        }
        self.assertEqual(
            cfg.scope_setting("wichita", "direction_regions", ""), "north=road,east=driveway")
        self.assertEqual(cfg.scope_setting("wichita", "entity_expected_presence_ratio", 0.5), 0.6)
        # taylor_hill unaffected: keeps the flat (live) mapping + ratio.
        self.assertEqual(cfg.scope_setting("taylor_hill", "direction_regions", ""), "west=driveway")
        self.assertEqual(cfg.scope_setting("taylor_hill", "entity_expected_presence_ratio", 0.5), 0.5)

    def test_unknown_scope_param_raises(self):
        with self.assertRaises(ValueError):
            _validate_scopes({"wichita": {"diretcion_regions": "x"}},
                             Config.SCOPE_KEYS)

    def test_scopes_env_json_roundtrip(self):
        import os
        from unittest import mock
        env = {
            "STEALTHCAM_MEMORY_SCOPE": "taylor_hill",
            "STEALTHCAM_SCOPES": json.dumps(
                {"wichita": {"direction_regions": "north=road"}}),
        }
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = load_config("/nonexistent/config.yaml")
        self.assertEqual(cfg.memory_scope, "taylor_hill")
        self.assertEqual(cfg.scopes["wichita"]["direction_regions"], "north=road")

    def test_scopes_env_invalid_json_raises(self):
        import os
        from unittest import mock
        with mock.patch.dict(os.environ, {"STEALTHCAM_SCOPES": "{oops"},
                              clear=False):
            with self.assertRaises(ValueError):
                load_config("/nonexistent/config.yaml")


if __name__ == "__main__":
    unittest.main()