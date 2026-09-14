"""Location-scope migration for the entity-memory SQLite pipeline DB.

The entity-memory engine (memory.py) was first deployed Taylor-Hill/Stealthcam
-shaped: every memory table keyed identity on direction/guid alone. This
migration generalizes the store to a **per-location scope dimension** so one
codebase + one DB serve multiple sites (e.g. ``taylor_hill`` and ``wichita``)
with no cross-scope leakage:

- every memory table gains a ``scope`` column and a composite PK that includes
  it (entities PK(scope, entity_id), entity_presence
  PK(scope, entity_id, direction, guid), seen_guids PK(scope, guid), ...);
- existing rows are stamped ``scope='taylor_hill'`` (this DB's only live scope
  as of 2026-09-13) — the migration is additive: no row is dropped or zeroed;
- rebuilt tables keep ``scope TEXT NOT NULL DEFAULT 'taylor_hill'`` so the
  OLD worker binary (whose INSERTs omit scope) keeps writing into the right
  scope during the cutover window; the NEW code always writes scope
  explicitly, so the default is belt-and-braces, never load-bearing;
- a fresh database simply gets the scoped schema from _SCHEMA (nothing to
  migrate, and no legacy-default column baggage);
- re-running on an already-migrated DB is a no-op (idempotent).

Run standalone (after copying the DB — never work on the live file without a
staging pass first):

    python3 -m stealthcam.migrate /path/to/stealthcam.db \
        [--stamp taylor_hill] [--backup-dir DIR] [--dry-run]

The Memory class calls ``migrate_scoped_db()`` at open, so a deployment that
ships this code migrates on first use. That auto-migration path is wrapped by
``migrate_db_open()``, which takes the same one-time ``.pre-scope-migration.bak``
snapshot the CLI path takes before any write (deploy-path safety: the backup
must not exist only on the manual path). Each table rebuild is additionally
wrapped in one explicit transaction, so a crash mid-rebuild (SIGKILL, OOM,
container restart) rolls back to the pre-rebuild state instead of stranding
the table half-migrated.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

# The only live scope on this DB as of the 2026-09-13 refactor. Kept explicit
# (not a bare constant sprinkled through the code) so the next location can
# grep for every place a default stamp is applied.
DEFAULT_STAMP = "taylor_hill"

# Schema epoch stamped into ``PRAGMA user_version`` once a DB reaches the
# scoped shape. Memory-open uses it as a fast path: an epoch match proves the
# DB passed this module's checks, so it skips the per-boot sqlite_master
# scan. BUMP THIS whenever the scoped schema or migration changes shape —
# DBs stamped with an older epoch re-run the full check on next open.
#
# Epoch history:
#   1 = per-location scope dimension (2026-09-13, PR #42).
#   2 = region-aware monitored_captures PK (2026-09-13, issue #44 / Percy
#       PR#43 P2-4): PK widened (scope, direction, region, guid) so one
#       frame can contribute one denominator row PER REGION; epoch-1 DBs
#       re-run the full check on next open and get the in-place PK upgrade.
#   3 = verified epoch stamp (2026-09-14, issue #47 / Percy PR#46 P2): the
#       stamp moved from Memory.__init__ into migrate_scoped_db itself and
#       is now applied only AFTER the PK shape of every _PK_UPGRADES table
#       is verified against the spec — a mixed-shape DB (e.g. epoch-1 PK
#       alongside an unscoped table) refuses loudly instead of being
#       attested as epoch 2 with a region-blind PK. Bumped so any DB the
#       buggy code already stamped with epoch 2 re-runs the full check
#       once on next open.
SCHEMA_EPOCH_USER_VERSION = 3

# Tables that must be rebuilt with a scope column + composite PK. Each entry:
# (table, pk-without-scope, all-columns-with-types, not-null-columns).
# ``region``/``attributes`` keep their nullable/DEFAULT semantics from the
# original schema; ``NOT NULL`` below lists what the old schema enforced.
_SCOPE_TABLES: dict[str, dict] = {
    "seen_guids": {
        "pk": ["guid"],
        "cols": {
            "guid": "TEXT",
            "seen_at": "TEXT NOT NULL",
        },
    },
    "memory_observations": {
        "pk": [],  # autoincrement rowid preserved via INTEGER PRIMARY KEY
        "cols": {
            "id": "INTEGER PRIMARY KEY",
            "direction": "TEXT NOT NULL",
            "bbox": "TEXT NOT NULL",
            "first_seen_at": "TEXT NOT NULL",
            "last_seen_at": "TEXT NOT NULL",
            "capture_count": "INTEGER NOT NULL DEFAULT 1",
            "classes_seen": "TEXT NOT NULL DEFAULT '[]'",
        },
    },
    "decisions": {
        "pk": [],  # id INTEGER PRIMARY KEY below
        "cols": {
            "id": "INTEGER PRIMARY KEY",
            "guid": "TEXT NOT NULL",
            "subimage": "INTEGER NOT NULL",
            "direction": "TEXT NOT NULL",
            "class": "TEXT NOT NULL",
            "confidence": "REAL NOT NULL",
            "bbox": "TEXT NOT NULL",
            "description": "TEXT",
            "decision": "TEXT NOT NULL",
            "reason": "TEXT NOT NULL",
            "decided_at": "TEXT NOT NULL",
        },
    },
    "feedback": {
        "pk": [],
        "cols": {
            "id": "INTEGER PRIMARY KEY",
            "decision_id": "INTEGER",
            "guid": "TEXT NOT NULL",
            "subimage": "INTEGER NOT NULL",
            "human_label": "TEXT NOT NULL",
            "flagged_at": "TEXT NOT NULL",
        },
    },
    "deferred_captures": {
        "pk": ["guid"],
        "cols": {
            "guid": "TEXT",
            "deferral_count": "INTEGER NOT NULL DEFAULT 1",
            "alerted": "INTEGER NOT NULL DEFAULT 0",
            "first_deferred_at": "TEXT NOT NULL",
            "last_deferred_at": "TEXT NOT NULL",
        },
    },
    "entities": {
        "pk": ["entity_id"],
        "cols": {
            "entity_id": "TEXT",
            "direction": "TEXT NOT NULL",
            "region": "TEXT",
            "fingerprint": "BLOB NOT NULL",
            "model": "TEXT NOT NULL",
            "first_seen_at": "TEXT NOT NULL",
            "last_seen_at": "TEXT NOT NULL",
            "capture_count": "INTEGER NOT NULL DEFAULT 1",
            "attributes": "TEXT NOT NULL DEFAULT '{}'",
            "expected": "INTEGER NOT NULL DEFAULT 0",
        },
    },
    "entity_presence": {
        "pk": ["entity_id", "direction", "guid"],
        "cols": {
            "entity_id": "TEXT",
            "direction": "TEXT NOT NULL",
            "guid": "TEXT",
            "captured_at": "TEXT NOT NULL",
            "ts": "REAL NOT NULL",
        },
    },
    "monitored_captures": {
        "pk": ["direction", "region", "guid"],
        "cols": {
            "direction": "TEXT NOT NULL",
            "region": "TEXT",
            "guid": "TEXT",
            "captured_at": "TEXT NOT NULL",
            "ts": "REAL NOT NULL",
            "is_day": "INTEGER NOT NULL DEFAULT 1",
        },
    },
}

# Indexes on the scoped schema (mirror the pre-scope layout; scope-first).
# Values are LISTS of CREATE INDEX statements (a table may need several —
# issue #47 P3: the CLI/rebuild path must re-create BOTH monitored_captures
# indexes, not just idx_monitored_scope_ts). Keep in lockstep with the
# CREATE INDEX statements in memory.py _MEMORY_SCHEMA: anything only in
# _MEMORY_SCHEMA self-heals on the next Memory open, but a CLI-migrated DB
# lacks it until then.
_SCOPE_INDEXES: dict[str, list[str]] = {
    "decisions": [
        "CREATE INDEX IF NOT EXISTS idx_decisions_scope_guid "
        "ON decisions (scope, guid)",
    ],
    "entity_presence": [
        "CREATE INDEX IF NOT EXISTS idx_entity_presence_scope_ts "
        "ON entity_presence (scope, entity_id, direction, ts)",
    ],
    "monitored_captures": [
        "CREATE INDEX IF NOT EXISTS idx_monitored_scope_ts "
        "ON monitored_captures (scope, ts, is_day)",
        # Region-aware denominator index (issue #44 / PR #46); was missing
        # here so a CLI-migrated DB lacked it until the next Memory open.
        "CREATE INDEX IF NOT EXISTS idx_monitored_scope_dir_region_ts "
        "ON monitored_captures (scope, direction, region, ts, is_day)",
    ],
    "memory_observations": [
        "CREATE INDEX IF NOT EXISTS idx_memory_obs_scope_dir "
        "ON memory_observations (scope, direction)",
    ],
}

# PK-shape fixes applied IN PLACE to already-scoped DBs (epoch upgrades).
# Epoch-1 monitored_captures PKed on (scope, direction, guid); issue #44
# widened it to (scope, direction, region, guid) so a frame evidencing both
# road and driveway contributes one denominator row PER REGION. Each entry:
# (old-pk-without-scope, new-pk-without-scope, shape-epoch). ``shape-epoch``
# is the schema generation whose PK shape this upgrade INSTALLS (the
# region-aware shape shipped as epoch 2, PR #46) and names the one-time
# rollback snapshot (.pre-epoch<shape-epoch>-pk-upgrade.bak). It must NOT
# track SCHEMA_EPOCH_USER_VERSION: a stamping-protocol fix that bumps the
# epoch (issue #47, epoch 3) changes no PK shape, and renaming the rollback
# artifact an operator already knows would defeat its idempotence.
_PK_UPGRADES: dict[str, tuple[list[str], list[str], int]] = {
    "monitored_captures": (["direction", "guid"],
                           ["direction", "region", "guid"], 2),
}


def _upgrade_pk_shapes(conn: sqlite3.Connection, stamp: str) -> dict:
    """Rebuild already-scoped tables whose PK shape predates the current
    spec (epoch upgrades), preserving every row and its scope value.

    Runs inside one explicit transaction per table (SQLite DDL is
    transactional — a crash mid-rebuild rolls the table back whole). Rows
    keep their ORIGINAL scope (never re-stamped). A PK collision under the
    widened key means the legacy data was already region-ambiguous in a way
    the old PK silently collapsed — fail loudly rather than drop rows.
    """
    upgraded: dict[str, int] = {}
    for table, (old_pk, new_pk, _shape_epoch) in _PK_UPGRADES.items():
        if not _table_exists(conn, table):
            continue
        if not _is_scoped(conn, table):
            continue  # legacy shape: handled by the rebuild path below
        cur_pk = _pk_cols(conn, table)
        expected_old = ["scope"] + old_pk
        if cur_pk == ["scope"] + new_pk:
            continue  # already upgraded
        if cur_pk != expected_old:
            raise RuntimeError(
                f"{table}: PK {cur_pk} matches neither the epoch-1 shape "
                f"{expected_old} nor the current shape "
                f"{['scope'] + new_pk}; refusing to guess (manual "
                f"reconciliation required)")
        spec = _SCOPE_TABLES[table]
        cols = list(spec["cols"].keys())
        sel_cols = ", ".join(["scope"] + cols)
        ins_cols = sel_cols
        placeholders = ", ".join(["?"] * (len(cols) + 1))
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(f"ALTER TABLE {table} RENAME TO {table}_legacy")
            _create_scoped(conn, table, spec, stamp)
            rows = conn.execute(
                f"SELECT {sel_cols} FROM {table}_legacy").fetchall()
            conn.executemany(
                f"INSERT INTO {table}_new ({ins_cols}) VALUES ({placeholders})",
                rows,
            )
            conn.execute(f"DROP TABLE {table}_legacy")
            conn.execute(f"ALTER TABLE {table}_new RENAME TO {table}")
            for stmt in _SCOPE_INDEXES.get(table, []):
                conn.execute(stmt)
        except BaseException:
            conn.rollback()
            raise
        conn.commit()
        upgraded[table] = len(rows)
    return upgraded


def _is_scoped(conn: sqlite3.Connection, table: str) -> bool:
    cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    return "scope" in cols


def _pk_cols(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    pk = [r[1] for r in sorted((r for r in rows if r[5] > 0), key=lambda r: r[5])]
    return pk


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
        (table,),
    ).fetchone() is not None


def assert_no_orphan_rebuild_tables(conn: sqlite3.Connection) -> None:
    """Refuse to run over stranded ``*_legacy``/``*_new`` rebuild artifacts.

    A crash during an OLD (pre-fix) migration run can leave ``<table>_legacy``
    / ``<table>_new`` behind with the real ``<table>`` empty. Silently
    "migrating" around that would strand the rows forever and hand the engine
    an EMPTY table (silent data wipe). The DB is recoverable by hand from
    those tables — make the operator do that loudly rather than guess.

    Deliberately UNCONDITIONAL (runs on every Memory open, even on the
    user_version fast path): it is a single sqlite_master query, and the
    fast path must never skip the stranded-data guard.
    """
    orphans = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND (name LIKE '%_legacy' OR name LIKE '%_new')")]
    if orphans:
        raise RuntimeError(
            f"stranded rebuild table(s) from an interrupted migration: "
            f"{sorted(orphans)} — a previous migration crashed mid-rebuild; "
            f"refusing to open. Recover by hand: for each affected table, "
            f"copy rows from the _legacy/_new table back into the live one "
            f"(from the .pre-scope-migration.bak snapshot if needed), drop "
            f"the orphan tables, and retry. Do NOT delete the orphans "
            f"blindly — they may hold the only copy of the rows.")


def _create_scoped(conn: sqlite3.Connection, table: str, spec: dict,
                   default_scope: str) -> None:
    """Create the scoped version of `table`.

    ``scope`` carries ``DEFAULT '<default_scope>'``: the OLD worker binary
    (whose INSERTs omit scope) keeps writing into the stamped scope during
    the cutover window instead of failing a NOT NULL check. The NEW code
    always supplies scope explicitly, so the default is armor, never
    load-bearing. (A FRESH database's schema has NO default — see _SCHEMA in
    memory.py — so new-code omissions stay loud there.)
    """
    cols = spec["cols"]
    pk = spec["pk"]
    col_defs = [f"scope TEXT NOT NULL DEFAULT '{default_scope}'"]
    # scope goes FIRST in every composite PK (see module docstring).
    for name, decl in cols.items():
        col_defs.append(f"{name} {decl}")
    if pk:
        col_defs.append(f"PRIMARY KEY (scope, {', '.join(pk)})")
    conn.execute(f"CREATE TABLE {table}_new ({', '.join(col_defs)})")


def _verify_pk_shapes(conn: sqlite3.Connection) -> None:
    """Verify every _PK_UPGRADES table carries the current (scoped) PK.

    Issue #47: the epoch stamp ATTESTS this shape, so it must be verified,
    never assumed. The historical defect: the mixed-shape (rebuild) branch
    returned without running _upgrade_pk_shapes while Memory.__init__
    stamped the epoch unconditionally — a user_version=2 attest over a
    region-blind monitored_captures PK, and the fast path trusted the lying
    stamp on every later boot (issue #44's silent road-row drop reborn,
    zero errors anywhere). A mismatch here refuses loudly (same contract as
    assert_no_orphan_rebuild_tables), leaves the stamp unwritten, and the
    DB keeps its old epoch so the full check re-runs after reconciliation.
    """
    for table, (_old_pk, new_pk, _shape_epoch) in _PK_UPGRADES.items():
        if not _table_exists(conn, table):
            continue
        cur = _pk_cols(conn, table)
        if cur != ["scope"] + new_pk:
            raise RuntimeError(
                f"{table}: PK {cur} is not the required scoped shape "
                f"{['scope'] + new_pk}; refusing to stamp schema epoch "
                f"{SCHEMA_EPOCH_USER_VERSION} over an unverified shape "
                f"(manual reconciliation required)")


def _stamp_epoch(conn: sqlite3.Connection) -> None:
    """Stamp the schema epoch. Called ONLY from migrate_scoped_db, only on
    exits whose PK verification passed (issue #47)."""
    conn.execute(f"PRAGMA user_version = {SCHEMA_EPOCH_USER_VERSION}")


def migrate_scoped_db(conn: sqlite3.Connection,
                      stamp: str = DEFAULT_STAMP) -> dict:
    """Bring a (possibly pre-scope) pipeline DB to the scoped schema.

    Returns a report dict {table: rows_migrated} plus ``status``:
    ``already-scoped`` (no-op), ``migrated`` (work done), ``fresh``
    (created empty scoped tables).

    Idempotent. Additive: existing rows are preserved with scope=stamp.

    Epoch stamping (issue #47): this function — not Memory.__init__ —
    writes ``PRAGMA user_version = SCHEMA_EPOCH_USER_VERSION``, and only
    after verifying every _PK_UPGRADES table's PK matches the current
    scoped shape (``_verify_pk_shapes``). A mismatch raises RuntimeError
    naming the table and shape, consistent with
    assert_no_orphan_rebuild_tables; the stamp is never written over an
    unverified shape, so the next open re-runs the full check.

    Crash safety: each table's rebuild (RENAME -> CREATE _new -> INSERT ->
    DROP legacy -> RENAME back) runs inside ONE explicit BEGIN IMMEDIATE /
    COMMIT transaction — SQLite DDL is transactional, so a process killed
    mid-rebuild rolls back to the pre-rebuild state and the retry re-runs
    that table cleanly. In Python sqlite3's legacy autocommit mode the DDL
    would otherwise commit immediately, stranding the rebuild half-done.

    The caller controls the outer transaction for the non-rebuild paths
    (fresh-DB schema creation, index creation); those commit via the
    caller's ``conn.commit()`` as before.
    """
    assert_no_orphan_rebuild_tables(conn)

    report: dict = {"status": "already-scoped", "tables": {}}
    existing = {t for (t,) in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}

    scoped_targets = [t for t in _SCOPE_TABLES if t in existing]
    if not scoped_targets:
        # Fresh DB: create the scoped schema directly (no legacy default).
        from .memory import MEMORY_SCHEMA

        conn.executescript(MEMORY_SCHEMA)
        report["status"] = "fresh"
        # _MEMORY_SCHEMA builds every _PK_UPGRADES table in the current
        # shape, so the stamp below is preceded by the same verification
        # every other exit runs (defense in depth, issue #47).
        _verify_pk_shapes(conn)
        _stamp_epoch(conn)
        return report

    not_scoped = [t for t in scoped_targets if not _is_scoped(conn, t)]
    if not not_scoped:
        # All present tables already carry scope — verify PKs include it,
        # then apply any epoch PK-shape upgrades (issue #44: monitored_captures
        # (scope, direction, guid) -> (scope, direction, region, guid)).
        # Row-preserving, scope-preserving, one transaction per table.
        upgraded = _upgrade_pk_shapes(conn, stamp)
        if upgraded:
            report["status"] = "upgraded"
            report["pk_upgraded"] = upgraded
        for t in scoped_targets:
            pk = _pk_cols(conn, t)
            if pk[:1] != ["scope"] and _SCOPE_TABLES[t]["pk"]:
                raise RuntimeError(
                    f"{t} has a scope column but PK {pk} does not start with "
                    f"scope — schema is neither legacy nor scoped; refusing "
                    f"to guess (manual reconciliation required)")
        _verify_pk_shapes(conn)
        _stamp_epoch(conn)
        return report

    # -- rebuild path: every not-yet-scoped table is copied + stamped --------
    report["status"] = "migrated"
    for table in not_scoped:
        spec = _SCOPE_TABLES[table]
        cols = list(spec["cols"].keys())
        sel_cols = ", ".join(cols)
        ins_cols = ", ".join(["scope"] + cols)
        placeholders = ", ".join(["?"] * (len(cols) + 1))

        # One explicit transaction per table rebuild: RENAME -> CREATE _new
        # -> INSERT -> DROP legacy -> RENAME back -> index. SQLite DDL is
        # transactional, so a crash anywhere in here rolls the whole table
        # back to its pre-rebuild (legacy) state; the retry re-runs it.
        # (executescript would COMMIT any open txn implicitly — individual
        # execute() calls keep this atomic.)
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(f"ALTER TABLE {table} RENAME TO {table}_legacy")
            _create_scoped(conn, table, spec, stamp)
            rows = conn.execute(
                f"SELECT {sel_cols} FROM {table}_legacy").fetchall()
            # Plain INSERT: the new PK only adds a constant (the stamp) to the
            # old one, so a collision here means legacy data was already
            # corrupt — fail loudly rather than silently drop rows.
            conn.executemany(
                f"INSERT INTO {table}_new ({ins_cols}) VALUES ({placeholders})",
                [(stamp, *row) for row in rows],
            )
            conn.execute(f"DROP TABLE {table}_legacy")
            conn.execute(f"ALTER TABLE {table}_new RENAME TO {table}")
            for stmt in _SCOPE_INDEXES.get(table, []):
                conn.execute(stmt)
        except BaseException:
            conn.rollback()
            raise
        conn.commit()
        report["tables"][table] = len(rows)

    # Tables that existed but were already scoped (mixed-shape guard).
    for table in scoped_targets:
        if table not in not_scoped:
            report["tables"][table] = 0
    # Issue #47: the rebuild branch previously returned WITHOUT running
    # _upgrade_pk_shapes, so a mixed-shape DB (e.g. epoch-1 monitored_captures
    # PK + an unscoped seen_guids) got the epoch stamp attesting a PK shape
    # it never had or verified — and the fast path trusted that stamp
    # forever. Verify before stamping: a mismatch raises here, the stamp is
    # never written, and Memory-open surfaces the refusal loudly.
    _verify_pk_shapes(conn)
    _stamp_epoch(conn)
    return report


def _backup_path(db_path: Path) -> Path:
    return db_path.with_name(db_path.name + ".pre-scope-migration.bak")


def _snapshot(db_path: Path, target: Path) -> None:
    """Consistent snapshot of a possibly-WAL SQLite DB via the backup API.

    A plain file copy of a DB in WAL mode can be torn (main file without its
    -wal tail); sqlite3's online backup produces a self-consistent copy even
    while a writer holds the live DB.
    """
    src = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        dst = sqlite3.connect(str(target))
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()


def _ensure_backup(db_path: Path, backup_dir: str | None = None) -> Path | None:
    """Take the one-time pre-migration snapshot if it doesn't already exist.

    Idempotent: re-runs never overwrite the original snapshot (the original
    pre-migration bytes are the artifact — a second snapshot of an already
    migrated DB would silently replace the rollback copy with migrated data).
    Returns the backup path, or None when there is nothing to snapshot
    (non-file-backed, or a zero-byte file — an empty DB has no data to roll
    back; sqlite3.connect materializes a 0-byte file the moment Memory opens
    a not-yet-existing path, so this is what a "fresh DB" looks like here).
    """
    if not db_path.is_file():
        return None
    if db_path.stat().st_size == 0:
        return None
    if backup_dir:
        target = Path(backup_dir).resolve() / _backup_path(db_path).name
    else:
        target = _backup_path(db_path)
    if not target.is_file():
        _snapshot(db_path, target)
    return target


def _pk_upgrade_pending(conn: sqlite3.Connection) -> int | None:
    """Shape-epoch of the first pending in-place PK upgrade, or None.

    The returned epoch names the rollback snapshot
    (.pre-epoch<shape-epoch>-pk-upgrade.bak): the schema generation whose PK
    shape the pending upgrade INSTALLS, not necessarily the current
    SCHEMA_EPOCH_USER_VERSION (issue #47 — a stamping-protocol bump changes
    no PK shape and must not rename operator-facing rollback artifacts).
    """
    for table, (old_pk, _new_pk, shape_epoch) in _PK_UPGRADES.items():
        if not _table_exists(conn, table) or not _is_scoped(conn, table):
            continue
        if _pk_cols(conn, table) == ["scope"] + old_pk:
            return shape_epoch
    return None


def _ensure_epoch_backup(db_path: Path, epoch: int) -> Path | None:
    """One-time snapshot before an in-place PK-shape upgrade (epoch bump).

    Same idempotence rule as _ensure_backup: an existing snapshot is never
    overwritten (the pre-upgrade bytes are the artifact). A separate file
    from the epoch-1 pre-scope snapshot — that one holds pre-scope data and
    is consumed by a different rollback path; the PK upgrade needs ITS OWN
    rollback point. Skipped for non-file / zero-byte DBs (nothing to roll
    back).
    """
    if not db_path.is_file() or db_path.stat().st_size == 0:
        return None
    target = db_path.with_name(
        db_path.name + f".pre-epoch{epoch}-pk-upgrade.bak")
    if not target.is_file():
        _snapshot(db_path, target)
    return target


def migrate_db_open(db_path: str | Path, conn: sqlite3.Connection,
                    stamp: str = DEFAULT_STAMP,
                    backup_dir: str | None = None,
                    db_is_new: bool = False) -> dict:
    """Migration entry point for the Memory-open (deploy auto-migration) path.

    Takes the same one-time ``.pre-scope-migration.bak`` snapshot the CLI
    path takes BEFORE migrating, then runs ``migrate_scoped_db``. The caller
    owns the connection (Memory keeps using it) and commits as usual.

    Backup is skipped when the DB file does not exist yet (Memory creating a
    fresh DB — nothing pre-migration exists to roll back) or is ``:memory:``
    (dies with the process anyway). ``db_is_new`` is the caller's direct
    knowledge that THIS open created the file: Memory enables WAL before
    migrating (multi-writer readiness), and the WAL pragma materializes a
    real header on the previously-not-existing file, which would defeat the
    0-byte freshness heuristic — so the constructor states "new" explicitly.
    """
    path = Path(db_path)
    if str(path) != ":memory:" and not db_is_new:
        _ensure_backup(path, backup_dir)
    # Epoch-upgrade path (issue #44): an already-scoped DB whose
    # monitored_captures PK predates the region-aware shape gets its own
    # one-time rollback snapshot before the in-place rebuild — the epoch-1
    # pre-scope .bak holds pre-scope data and is the WRONG rollback point
    # for this upgrade. Same idempotence rule: never overwrite an existing
    # snapshot. (The scoped-shape check needs the caller's connection; the
    # snapshot itself uses the backup API against the file.)
    pending_epoch = _pk_upgrade_pending(conn)
    if str(path) != ":memory:" and pending_epoch is not None:
        _ensure_epoch_backup(path, pending_epoch)
    return migrate_scoped_db(conn, stamp=stamp)


def migrate_file(db_path: str | Path, stamp: str = DEFAULT_STAMP,
                 backup_dir: str | None = None,
                 backup: bool = True) -> dict:
    """Migrate a DB file in place, taking a one-time backup first.

    The backup is skipped if one already exists (idempotent re-runs never
    overwrite the original snapshot). Returns the migrate report.
    """
    path = Path(db_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"no such database: {path}")
    if backup:
        _ensure_backup(path, backup_dir)
    conn = sqlite3.connect(str(path))
    try:
        # Epoch-upgrade snapshot (issue #44) — same policy as the auto path:
        # one-time, never overwrites, taken only when a PK rebuild is pending.
        pending_epoch = _pk_upgrade_pending(conn)
        if backup and pending_epoch is not None:
            _ensure_epoch_backup(path, pending_epoch)
        report = migrate_scoped_db(conn, stamp=stamp)
        conn.commit()
        # WAL may hold the migration in -wal; checkpoint so the main file
        # alone is self-consistent for later copy/inspect.
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return report
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Add the per-location scope dimension to an "
                    "entity-memory pipeline DB.")
    ap.add_argument("db", help="path to stealthcam.db (stage a COPY first)")
    ap.add_argument("--stamp", default=DEFAULT_STAMP,
                    help=f"scope stamped onto legacy rows (default: {DEFAULT_STAMP})")
    ap.add_argument("--backup-dir", default=None,
                    help="directory for the pre-migration backup (default: beside the DB)")
    ap.add_argument("--no-backup", action="store_true",
                    help="skip the pre-migration backup copy")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be migrated, change nothing")
    args = ap.parse_args(argv)

    path = Path(args.db).resolve()
    if not path.is_file():
        print(f"error: no such database: {path}", file=sys.stderr)
        return 2
    if args.dry_run:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            existing = {t for (t,) in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            targets = [t for t in _SCOPE_TABLES if t in existing]
            pending = [t for t in targets if not _is_scoped(conn, t)]
            print("dry-run: tables needing migration:", pending or "none")
            # Epoch PK-shape upgrades (issue #44): already-scoped tables whose
            # PK predates the current spec (rebuilt in place, rows preserved).
            pk_pending = {}
            for t, (old_pk, new_pk, _shape_epoch) in _PK_UPGRADES.items():
                if t not in existing or not _is_scoped(conn, t):
                    continue
                cur = _pk_cols(conn, t)
                if cur == ["scope"] + new_pk or cur != ["scope"] + old_pk:
                    continue
                pk_pending[t] = cur
            print("dry-run: scoped tables needing PK upgrade:",
                  pk_pending or "none")
            for t in targets:
                n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                print(f"  {t}: {n} rows")
            return 0
        finally:
            conn.close()
    report = migrate_file(path, stamp=args.stamp,
                          backup=not args.no_backup, backup_dir=args.backup_dir)
    print(f"migration: {report['status']}")
    for table, n in report["tables"].items():
        print(f"  {table}: {n} rows stamped scope='{args.stamp}'")
    for table, n in report.get("pk_upgraded", {}).items():
        print(f"  {table}: PK upgraded in place, {n} rows preserved")
    return 0


if __name__ == "__main__":
    sys.exit(main())