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
   legacy flat attrs, an unknown key in a scopes block raises at load, a YAML
   `scopes:` block both applies and validates (typo -> raise), and scope
   overrides actually reach the live path (run_once -> process_capture,
   _make_memory -> Memory) rather than being silently dropped for flat attrs.
5. Migration safety: the Memory-open auto-migration path takes the same
   one-time backup as the CLI; a crash mid-rebuild is transactional (rolls
   back) and stranded orphan tables from an older crash refuse to run loudly.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from stealthcam.memory import Memory, ALERT, NOTE
from stealthcam.migrate import (migrate_scoped_db, migrate_file,
                                DEFAULT_STAMP, SCHEMA_EPOCH_USER_VERSION)
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
        # Composite PKs include scope, scope-first. monitored_captures is
        # epoch-2 (issue #44): PK (scope, direction, region, guid).
        for t, pk_len in (("seen_guids", 2), ("entities", 2),
                          ("entity_presence", 4), ("monitored_captures", 4),
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

    def test_memory_open_migration_takes_backup(self):
        # P0-1 regression: the deploy path is Memory-open auto-migration —
        # it must take the same one-time .pre-scope-migration.bak snapshot
        # the CLI path takes. Without it, a bad migration on the only
        # production DB has no rollback copy.
        path, _ = self._legacy_db()
        mem = _mk(str(path), "taylor_hill")
        mem.close()
        bak = Path(str(path) + ".pre-scope-migration.bak")
        self.assertTrue(bak.is_file(),
                        "Memory-open auto-migration must leave a backup")
        # The backup holds the PRE-migration (legacy) shape, not the
        # migrated one — a snapshot of migrated data is not a rollback copy.
        bak_conn = sqlite3.connect(str(bak))
        legacy_cols = {r[1] for r in bak_conn.execute(
            "PRAGMA table_info(seen_guids)")}
        bak_conn.close()
        self.assertNotIn("scope", legacy_cols,
                         "backup must hold the pre-migration schema")
        # Idempotent: a second open must NOT overwrite the original snapshot
        # (the pre-migration bytes are the artifact).
        mem2 = _mk(str(path), "taylor_hill")
        mem2.close()
        self.assertTrue(bak.is_file())
        bak_conn = sqlite3.connect(str(bak))
        self.assertNotIn(
            "scope",
            {r[1] for r in bak_conn.execute("PRAGMA table_info(seen_guids)")})
        bak_conn.close()

    def test_memory_open_fresh_db_no_backup_junk(self):
        # A fresh DB (created by Memory) has no pre-migration data — no
        # backup file should be created for it (nothing to roll back).
        path = self.dir / "fresh-open.db"
        Memory(str(path), fingerprinter=_TrivialFingerprinter(),
               scope="wichita").close()
        self.assertFalse(
            Path(str(path) + ".pre-scope-migration.bak").is_file())


class TestMigrationCrashSafety(unittest.TestCase):
    """P0-2: a crash mid-rebuild must not strand data silently.

    The per-table rebuild used to run in Python sqlite3's legacy autocommit
    mode, where every DDL commits immediately — a SIGKILL between RENAME and
    RENAME-back left ``<table>_legacy``/``<table>_new`` orphaned and the
    retry path recreated ``<table>`` EMPTY (silent data wipe). Now the
    rebuild is one explicit transaction (SQLite DDL is transactional), and
    stranded orphan tables from any older crash make the next open REFUSE
    loudly instead of skipping them.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _legacy_db(self) -> Path:
        path = self.dir / "legacy.db"
        conn = sqlite3.connect(str(path))
        conn.executescript("""
CREATE TABLE seen_guids (guid TEXT PRIMARY KEY, seen_at TEXT NOT NULL);
CREATE TABLE entities (entity_id TEXT PRIMARY KEY, direction TEXT NOT NULL,
    region TEXT, fingerprint BLOB NOT NULL, model TEXT NOT NULL,
    first_seen_at TEXT NOT NULL, last_seen_at TEXT NOT NULL,
    capture_count INTEGER NOT NULL DEFAULT 1,
    attributes TEXT NOT NULL DEFAULT '{}', expected INTEGER NOT NULL DEFAULT 0);
""")
        for i in range(500):
            conn.execute("INSERT INTO seen_guids VALUES (?, ?)",
                         (f"g{i}", _iso(_T0)))
        for i in range(4):
            conn.execute(
                "INSERT INTO entities (entity_id, direction, fingerprint, "
                "model, first_seen_at, last_seen_at) VALUES (?, 'East', ?, "
                "'m', ?, ?)", (f"e{i}", f"fp{i}".encode(), _iso(_T0), _iso(_T0)))
        conn.commit()
        conn.close()
        return path

    def _table_shape(self, path: Path, table: str) -> tuple[list[str], int]:
        conn = sqlite3.connect(str(path))
        try:
            cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
            n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            return cols, n
        finally:
            conn.close()

    def _orphan_names(self, path: Path) -> list[str]:
        conn = sqlite3.connect(str(path))
        try:
            return [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND (name LIKE '%_legacy' OR name LIKE '%_new')")]
        finally:
            conn.close()

    def _kill_mid_migration(self, path: Path) -> str:
        """Run migrate_scoped_db in a child process and SIGKILL it once a
        rebuild artifact table appears (real process death, Percy probe 6
        shape). Returns the orphan names observed at kill time.

        With the transactional rebuild the artifact tables are uncommitted
        until the table's txn commits, so a ro-connection poll usually never
        sees them — that is itself the guarantee (the crash window is
        invisible). When the poll does catch one (e.g. a slow table commit
        boundary), the kill lands mid-flight for real.
        """
        repo_root = Path(__file__).resolve().parents[2]
        child = self.dir / "child.py"
        child.write_text(
            "import sys\n"
            f"sys.path.insert(0, {str(repo_root)!r})\n"
            "import sqlite3\n"
            "from stealthcam.migrate import migrate_scoped_db\n"
            "conn = sqlite3.connect(sys.argv[1])\n"
            "migrate_scoped_db(conn)\n"
            "conn.commit()\n"
            "print('COMPLETED')\n")
        proc = subprocess.Popen(
            [sys.executable, str(child), str(path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        deadline = time.time() + 30
        orphans: list[str] = []
        while time.time() < deadline:
            try:
                probe = sqlite3.connect(f"file:{path}?mode=ro", uri=True,
                                        timeout=0.3)
                orphans = [r[0] for r in probe.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND (name LIKE '%_legacy' OR name LIKE '%_new')")]
                probe.close()
            except sqlite3.OperationalError:
                pass  # locked — migration in flight, keep polling
            if orphans:
                proc.kill()  # SIGKILL mid-rebuild, no cleanup, no commit
                break
            if proc.poll() is not None:
                break  # finished before we could catch the window
            time.sleep(0.01)
        proc.communicate(timeout=30)
        return ",".join(orphans)

    def test_sigkill_mid_rebuild_crash_safe_on_retry(self):
        # The regression contract (Percy probe 6): kill the migrating process
        # with a real SIGKILL at any point during the rebuild, then run the
        # retry path (= Memory open). The retry must either restore the data
        # or refuse loudly — NEVER silently recreate the table empty.
        #
        # Under the transactional rebuild the orphan artifact tables are
        # uncommitted until each table's txn commits, so the ro-poll usually
        # cannot catch a mid-rebuild window (that IS the fix: the dangerous
        # window is invisible from outside). The pre-fix behavior would
        # strand visible committed orphans here (probe_6 saw seen_guids_
        # legacy/_new) and the retry would report success with the table
        # EMPTY. Assert the invariant either way:
        path = self._legacy_db()
        killed_during = self._kill_mid_migration(path)
        orphans = self._orphan_names(path)
        if not killed_during:
            # No committed orphans observed while the migration ran — with
            # the OLD autocommit rebuild this poll saw them within ~ms.
            self.assertEqual(
                orphans, [],
                "orphan tables committed outside any transaction are "
                "visible to a concurrent reader — that is the pre-fix "
                "stranding shape; the rebuild must be transactional")
        # Retry = the deploy's own recovery: next boot opens Memory again.
        mem = _mk(str(path), "taylor_hill")
        try:
            for i in range(500):
                self.assertTrue(mem.has_seen(f"g{i}"),
                                f"seen_guids g{i} lost after crash+retry")
            conn = sqlite3.connect(str(path))
            n_entities = conn.execute(
                "SELECT COUNT(*) FROM entities").fetchone()[0]
            conn.close()
            self.assertEqual(n_entities, 4,
                             "entities lost after crash+retry")
        finally:
            mem.close()
        # No stranded orphans, no wiped tables, no silent anything.
        self.assertEqual(self._orphan_names(path), [])

    def test_stranded_orphan_tables_refuse_to_run(self):
        # A DB stranded by an OLD (pre-fix) migration run — legacy/new
        # artifacts present, live table empty — must make the next open
        # RAISE, never silently "migrate" around the stranded rows.
        path = self._legacy_db()
        conn = sqlite3.connect(str(path))
        # Reproduce the pre-fix stranding shape (what probe_6 produced):
        conn.execute("ALTER TABLE seen_guids RENAME TO seen_guids_legacy")
        conn.execute(
            "CREATE TABLE seen_guids_new (scope TEXT NOT NULL, "
            "guid TEXT NOT NULL, seen_at TEXT NOT NULL, "
            "PRIMARY KEY (scope, guid))")
        conn.commit()
        conn.close()
        with self.assertRaises(RuntimeError) as ctx:
            _mk(str(path), "taylor_hill")
        self.assertIn("seen_guids_legacy", str(ctx.exception))
        # The stranded rows are still there for manual recovery — the
        # refusal must not have destroyed them.
        conn = sqlite3.connect(str(path))
        n = conn.execute(
            "SELECT COUNT(*) FROM seen_guids_legacy").fetchone()[0]
        conn.close()
        self.assertEqual(n, 500)

    def test_stranded_orphans_refuse_even_on_epoch_fast_path(self):
        # The user_version fast path must NOT skip the stranded-orphan guard:
        # a DB stamped with the epoch can still be stranded by a later
        # legacy-crash artifact planted on disk (or restored from a partial
        # backup). The guard is unconditional.
        path = self._legacy_db()
        _mk(str(path), "taylor_hill").close()  # migrates + stamps epoch
        conn = sqlite3.connect(str(path))
        self.assertEqual(
            conn.execute("PRAGMA user_version").fetchone()[0],
            SCHEMA_EPOCH_USER_VERSION)
        conn.execute("ALTER TABLE entities RENAME TO entities_legacy")
        conn.execute(
            "CREATE TABLE entities_new (scope TEXT NOT NULL)")
        conn.commit()
        conn.close()
        with self.assertRaises(RuntimeError) as ctx:
            _mk(str(path), "taylor_hill")
        self.assertIn("entities_legacy", str(ctx.exception))

    def test_orphan_check_in_migrate_scoped_db_too(self):
        # The guard lives at the migrate_scoped_db level, so the CLI path
        # (migrate_file) refuses stranded DBs as loudly as Memory-open.
        path = self._legacy_db()
        conn = sqlite3.connect(str(path))
        conn.execute("ALTER TABLE seen_guids RENAME TO seen_guids_legacy")
        conn.execute("CREATE TABLE seen_guids_new (scope TEXT NOT NULL)")
        conn.commit()
        with self.assertRaises(RuntimeError):
            migrate_scoped_db(conn)
        conn.close()

    def test_migrated_epoch_fast_path_skips_rescan(self):
        # P2: after a successful migration/open, the DB is stamped with the
        # schema epoch and the NEXT open skips the sqlite_master scan.
        path = self._legacy_db()
        conn = sqlite3.connect(str(path))
        self.assertEqual(
            conn.execute("PRAGMA user_version").fetchone()[0], 0)
        conn.close()
        mem = _mk(str(path), "taylor_hill")
        mem.close()
        conn = sqlite3.connect(str(path))
        self.assertEqual(
            conn.execute("PRAGMA user_version").fetchone()[0],
            SCHEMA_EPOCH_USER_VERSION,
            "Memory open must stamp the schema epoch")
        conn.close()
        # A fresh DB gets the epoch stamped too (same fast path).
        fresh = self.dir / "fresh-epoch.db"
        Memory(str(fresh), fingerprinter=_TrivialFingerprinter(),
               scope="taylor_hill").close()
        conn = sqlite3.connect(str(fresh))
        self.assertEqual(
            conn.execute("PRAGMA user_version").fetchone()[0],
            SCHEMA_EPOCH_USER_VERSION)
        conn.close()


class TestYamlScopeConfig(unittest.TestCase):
    """P1-1: YAML `scopes:` must apply AND validate (fail loudly on typos).

    The flat-key path used to flatten `scopes.<scope>.<param>` into a
    nonexistent `scopes_<scope>_<param>` attr that the hasattr() filter
    silently dropped — the override never applied and a typo'd param never
    raised, so a YAML-configured site silently ran another site's config.
    """

    def _load(self, yaml_text: str, env=None):
        import os
        from unittest import mock
        d = tempfile.mkdtemp()
        cfg_path = Path(d) / "config.yaml"
        cfg_path.write_text(yaml_text)
        with mock.patch.dict(os.environ, env or {}, clear=False):
            cfg = load_config(str(cfg_path))
        return cfg

    def test_yaml_scopes_block_applies(self):
        cfg = self._load(
            "memory_scope: wichita\n"
            "scopes:\n"
            "  wichita:\n"
            "    direction_regions: north=road,east=driveway\n"
            "    entity_expected_presence_ratio: 0.6\n")
        self.assertEqual(cfg.memory_scope, "wichita")
        self.assertEqual(
            cfg.scope_setting("wichita", "direction_regions", ""),
            "north=road,east=driveway")
        self.assertEqual(
            cfg.scope_setting("wichita", "entity_expected_presence_ratio", 0.5),
            0.6)
        # Other scopes untouched by wichita's block.
        self.assertEqual(
            cfg.scope_setting("taylor_hill", "entity_expected_presence_ratio", 0.5),
            0.5)

    def test_yaml_scopes_typo_raises(self):
        # A typo'd param in a YAML scopes block must fail at config load —
        # not silently no-op until the first decision.
        with self.assertRaises(ValueError) as ctx:
            self._load(
                "scopes:\n"
                "  wichita:\n"
                "    diretcion_regions: north=road\n")
        self.assertIn("diretcion_regions", str(ctx.exception))

    def test_yaml_scopes_non_mapping_raises(self):
        with self.assertRaises(ValueError):
            self._load("scopes: not-a-mapping\n")

    def test_yaml_dotted_scopes_key_refused(self):
        # The literal flat form `scopes.wichita.direction_regions: ...` used
        # to be silently dropped; it must now be refused (write the nested
        # form), so no path drops a scopes config silently.
        with self.assertRaises(ValueError) as ctx:
            self._load("scopes.wichita.direction_regions: north=road\n")
        self.assertIn("scopes.wichita", str(ctx.exception))

    def test_env_scopes_and_yaml_scopes_merge(self):
        # Both sources feed cfg.scopes; the env (compose-injected) wins on
        # per-key conflicts since it is applied after the file.
        import json
        import os
        from unittest import mock
        d = tempfile.mkdtemp()
        cfg_path = Path(d) / "config.yaml"
        cfg_path.write_text(
            "scopes:\n"
            "  wichita:\n"
            "    direction_regions: north=road\n"
            "  taylor_hill:\n"
            "    entity_expected_presence_ratio: 0.55\n")
        env = {"STEALTHCAM_SCOPES": json.dumps(
            {"wichita": {"direction_regions": "north=driveway"}})}
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = load_config(str(cfg_path))
        self.assertEqual(cfg.scopes["taylor_hill"]
                         ["entity_expected_presence_ratio"], 0.55)
        self.assertEqual(cfg.scopes["wichita"]["direction_regions"],
                         "north=driveway")


class TestScopeWiringLivePath(unittest.TestCase):
    """P1-2: scope overrides must reach the LIVE path, not just scope_setting.

    run_once used to pass night/day flat attrs to process_capture and
    _make_memory read the five entity/re-id params flat — so a scope block
    differing from the flat values was silently ignored. These tests assert
    the values that actually ARRIVE at process_capture / Memory.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _cfg(self, scope_block: dict, flat_overrides: dict) -> Config:
        cfg = Config()
        cfg.scopes = {"taylor_hill": scope_block}
        cfg.__post_init__()
        for k, v in flat_overrides.items():
            setattr(cfg, k, v)
        cfg.memory_scope = "taylor_hill"
        cfg.pipeline_db = str(Path(self.tmp.name) / "wiring.db")
        return cfg

    def test_make_memory_receives_scope_params(self):
        from stealthcam.worker import _make_memory
        cfg = self._cfg(
            {"entity_expected_presence_ratio": 0.9,
             "entity_baseline_window_days": 3.0,
             "entity_expected_absence_days": 5.0,
             "reid_similarity_threshold": 0.42,
             "reid_model": "cv2-silhouette-v2"},
            # Flat values deliberately DIFFERENT from the scope block.
            {"entity_expected_presence_ratio": 0.5,
             "entity_baseline_window_days": 7.0,
             "entity_expected_absence_days": 2.0,
             "reid_similarity_threshold": 0.55})
        mem = _make_memory(cfg)
        try:
            self.assertEqual(mem.entity_expected_presence_ratio, 0.9)
            self.assertEqual(mem.entity_baseline_window_days, 3.0)
            self.assertEqual(mem.entity_expected_absence_days, 5.0)
            self.assertEqual(mem.fingerprinter.similarity_threshold, 0.42)
        finally:
            mem.close()

    def test_make_memory_flat_fallback_when_no_scope_block(self):
        # No scope block -> flat attrs (the Taylor live shape) still win.
        from stealthcam.worker import _make_memory
        cfg = self._cfg({}, {"entity_expected_presence_ratio": 0.7})
        mem = _make_memory(cfg)
        try:
            self.assertEqual(mem.entity_expected_presence_ratio, 0.7)
        finally:
            mem.close()

    def test_run_once_passes_scope_params_to_process_capture(self):
        # Behavior, not reflection: stub run_once's externals, record the
        # kwargs process_capture actually receives (the Percy probe shape).
        from stealthcam import worker
        recorded = {}

        def fake_process_capture(capture, client, detector, describer,
                                 memory, notifier, dashboard, subimage_dir,
                                 **kw):
            recorded.update(kw)
            return {"alerts": 0, "flags": 0, "suppressed": 0, "detections": 0}

        class _Stub:
            def __init__(self, *a, **k):
                pass

            def login(self):
                pass

            def list_images(self, take_count=None):
                return [{"imageGuid": "fake-guid-1"}]

            def get_device_status(self):
                return {}

            def close(self):
                pass

        class _RecMemory:
            """Records the wiring that reaches has_seen/mark_seen."""
            def has_seen(self, guid):
                return False

            def mark_seen(self, guid):
                pass

            def close(self):
                pass

        stubs = {
            "StealthcamClient": _Stub,
            "Detector": _Stub,
            "Describer": _Stub,
            "Notifier": _Stub,
            "DashboardWriter": _Stub,
            "process_capture": fake_process_capture,
            "_make_memory": lambda cfg: _RecMemory(),
        }
        saved = {k: getattr(worker, k) for k in stubs}
        for k, v in stubs.items():
            setattr(worker, k, v)
        try:
            cfg = self._cfg(
                {"detector_night_conf_threshold": 0.9,
                 "day_start_hour": 6,
                 "day_end_hour": 21,
                 "direction_regions": "west=driveway"},
                {"detector_night_conf_threshold": 0.75,
                 "day_start_hour": 7,
                 "day_end_hour": 20,
                 "direction_regions": "south=road",
                 "dashboard_db_url": ""})
            cfg.subimage_dir = str(Path(self.tmp.name))
            worker.run_once(cfg)
        finally:
            for k, v in saved.items():
                setattr(worker, k, v)
        # The SCOPE values must be what reached process_capture — flat
        # values arriving here is exactly the P1-2 bug.
        self.assertEqual(recorded.get("night_conf_threshold"), 0.9)
        self.assertEqual(recorded.get("day_start_hour"), 6)
        self.assertEqual(recorded.get("day_end_hour"), 21)
        self.assertEqual(recorded.get("region_mapping"),
                         {"west": "driveway"})

    def test_run_once_flat_fallback_without_scope_block(self):
        # No scope block -> flat attrs reach process_capture unchanged.
        from stealthcam import worker
        recorded = {}

        def fake_process_capture(capture, client, detector, describer,
                                 memory, notifier, dashboard, subimage_dir,
                                 **kw):
            recorded.update(kw)
            return {"alerts": 0, "flags": 0, "suppressed": 0, "detections": 0}

        class _Stub:
            def __init__(self, *a, **k):
                pass

            def login(self):
                pass

            def list_images(self, take_count=None):
                return [{"imageGuid": "fake-guid-1"}]

            def get_device_status(self):
                return {}

            def close(self):
                pass

        class _RecMemory:
            def has_seen(self, guid):
                return False

            def mark_seen(self, guid):
                pass

            def close(self):
                pass

        stubs = {
            "StealthcamClient": _Stub,
            "Detector": _Stub,
            "Describer": _Stub,
            "Notifier": _Stub,
            "DashboardWriter": _Stub,
            "process_capture": fake_process_capture,
            "_make_memory": lambda cfg: _RecMemory(),
        }
        saved = {k: getattr(worker, k) for k in stubs}
        for k, v in stubs.items():
            setattr(worker, k, v)
        try:
            cfg = self._cfg({}, {"detector_night_conf_threshold": 0.77,
                                 "day_start_hour": 5,
                                 "day_end_hour": 22,
                                 "dashboard_db_url": ""})
            cfg.subimage_dir = str(Path(self.tmp.name))
            worker.run_once(cfg)
        finally:
            for k, v in saved.items():
                setattr(worker, k, v)
        self.assertEqual(recorded.get("night_conf_threshold"), 0.77)
        self.assertEqual(recorded.get("day_start_hour"), 5)
        self.assertEqual(recorded.get("day_end_hour"), 22)


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


class TestEpoch2PkUpgrade(unittest.TestCase):
    """Issue #44 (Percy PR#43 P2-4): the monitored_captures PK widens from
    (scope, direction, guid) to (scope, direction, region, guid) so one frame
    can contribute one denominator row PER REGION. Epoch-1 (already-scoped)
    DBs must upgrade in place, row- and scope-preserving, with a one-time
    epoch backup on both the auto and CLI paths."""

    def _epoch1_db(self) -> tuple[Path, int]:
        """Build an epoch-1-shaped DB (region-blind PK) with real rows."""
        path = self.dir / "epoch1.db"
        conn = sqlite3.connect(str(path))
        conn.executescript("""
CREATE TABLE monitored_captures (
    scope TEXT NOT NULL DEFAULT 'taylor_hill',
    direction TEXT NOT NULL, region TEXT, guid TEXT NOT NULL,
    captured_at TEXT NOT NULL, ts REAL NOT NULL,
    is_day INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (scope, direction, guid));
""")
        rows = [
            ("taylor_hill", "West", "driveway", "f1", _iso(_T0), 1.0, 1),
            ("taylor_hill", "West", "road", "f1-road", _iso(_T0), 1.0, 1),
            ("taylor_hill", "West", "driveway", "f2", _iso(_T0), 1.0, 1),
            ("wichita", "WichitaDriveway", "road", "m1", _iso(_T0), 1.0, 0),
        ]
        for r in rows:
            conn.execute(
                "INSERT INTO monitored_captures (scope, direction, region, "
                "guid, captured_at, ts, is_day) VALUES (?, ?, ?, ?, ?, ?, ?)",
                r)
        conn.commit()
        conn.close()
        return path, 4

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="epoch2-"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.dir, ignore_errors=True)

    def _pk(self, conn, table):
        info = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return [r[1] for r in sorted((r for r in info if r[5] > 0),
                                     key=lambda r: r[5])]

    def test_pk_upgrade_preserves_rows_and_scope(self):
        path, n = self._epoch1_db()
        conn = sqlite3.connect(str(path))
        report = migrate_scoped_db(conn)
        self.assertEqual(report["status"], "upgraded")
        self.assertEqual(report["pk_upgraded"], {"monitored_captures": 4})
        conn.commit()
        self.assertEqual(
            conn.execute("SELECT COUNT(*) FROM monitored_captures")
            .fetchone()[0], n, "every row preserved")
        self.assertEqual(
            conn.execute("SELECT COUNT(*) FROM monitored_captures "
                         "WHERE scope IS NULL OR scope = ''").fetchone()[0], 0)
        self.assertEqual(
            self._pk(conn, "monitored_captures"),
            ["scope", "direction", "region", "guid"],
            "PK must be the region-aware shape after upgrade")
        # Both scope values survived (never re-stamped).
        self.assertEqual(
            conn.execute("SELECT COUNT(*) FROM monitored_captures "
                         "WHERE scope='wichita'").fetchone()[0], 1)
        conn.close()

    def test_pk_upgrade_idempotent(self):
        path, n = self._epoch1_db()
        conn = sqlite3.connect(str(path))
        r1 = migrate_scoped_db(conn)
        self.assertEqual(r1["status"], "upgraded")
        conn.commit()
        r2 = migrate_scoped_db(conn)
        conn.commit()
        self.assertEqual(r2["status"], "already-scoped")
        self.assertEqual(
            conn.execute("SELECT COUNT(*) FROM monitored_captures")
            .fetchone()[0], n)
        conn.close()

    def test_epoch1_and_legacy_dbs_reject_unknown_pk_shape(self):
        # A DB whose PK matches NEITHER the epoch-1 nor the current shape is
        # a manual-reconciliation situation, never a silent guess.
        path = self.dir / "weird.db"
        conn = sqlite3.connect(str(path))
        conn.executescript("""
CREATE TABLE monitored_captures (
    scope TEXT NOT NULL, direction TEXT NOT NULL, region TEXT,
    guid TEXT NOT NULL, captured_at TEXT NOT NULL, ts REAL NOT NULL,
    is_day INTEGER NOT NULL DEFAULT 1, extra TEXT,
    PRIMARY KEY (scope, direction, guid, extra));
""")
        conn.commit()
        with self.assertRaises(RuntimeError):
            migrate_scoped_db(conn)
        conn.close()

    def test_auto_open_takes_epoch_backup_before_upgrade(self):
        path, n = self._epoch1_db()
        mem = Memory(str(path), scope="taylor_hill",
                     fingerprinter=_TrivialFingerprinter())
        try:
            bak = path.with_name("epoch1.db.pre-epoch2-pk-upgrade.bak")
            self.assertTrue(bak.is_file(),
                            "epoch-upgrade must snapshot before rebuilding")
            # The backup holds the PRE-upgrade PK shape.
            bak_conn = sqlite3.connect(str(bak))
            pk = [r[1] for r in sorted(
                (r for r in bak_conn.execute(
                    "PRAGMA table_info(monitored_captures)") if r[5] > 0),
                key=lambda r: r[5])]
            bak_conn.close()
            self.assertEqual(pk, ["scope", "direction", "guid"])
            # And the live DB is upgraded and functional.
            mem.record_monitored("West", "driveway", "g-new", _iso(_T0))
            self.assertEqual(
                mem._conn.execute(
                    "SELECT COUNT(*) FROM monitored_captures "
                    "WHERE region='driveway'").fetchone()[0], 3)
        finally:
            mem.close()

    def test_auto_open_upgraded_db_stamps_current_epoch(self):
        path, _ = self._epoch1_db()
        mem = Memory(str(path), scope="taylor_hill",
                     fingerprinter=_TrivialFingerprinter())
        try:
            conn = sqlite3.connect(str(path))
            self.assertEqual(
                conn.execute("PRAGMA user_version").fetchone()[0],
                SCHEMA_EPOCH_USER_VERSION)
            conn.close()
        finally:
            mem.close()

    def test_fresh_memory_schema_has_region_pk(self):
        m = Memory(":memory:", scope="taylor_hill",
                   fingerprinter=_TrivialFingerprinter())
        try:
            pk = [r[1] for r in sorted(
                (r for r in m._conn.execute(
                    "PRAGMA table_info(monitored_captures)") if r[5] > 0),
                key=lambda r: r[5])]
            self.assertEqual(pk, ["scope", "direction", "region", "guid"])
        finally:
            m.close()

    def test_cli_migrate_reports_pk_upgrade(self):
        path, _ = self._epoch1_db()
        from stealthcam.migrate import main as migrate_main
        rc = migrate_main([str(path), "--no-backup"])
        self.assertEqual(rc, 0)
        conn = sqlite3.connect(str(path))
        try:
            self.assertEqual(
                self._pk(conn, "monitored_captures"),
                ["scope", "direction", "region", "guid"])
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()