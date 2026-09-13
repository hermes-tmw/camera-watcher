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

    python3 -m stealthcam.migrate_scope /path/to/stealthcam.db \
        [--stamp taylor_hill] [--backup-dir DIR] [--dry-run]

The Memory class calls ``migrate_scoped_db()`` at open, so a deployment that
ships this code migrates on first use. A ``.pre-scope-migration.bak`` copy is
taken once (first successful run) before any write.
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
        "pk": ["direction", "guid"],
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
_SCOPE_INDEXES = {
    "decisions": "CREATE INDEX IF NOT EXISTS idx_decisions_scope_guid "
                 "ON decisions (scope, guid)",
    "entity_presence": "CREATE INDEX IF NOT EXISTS idx_entity_presence_scope_ts "
                       "ON entity_presence (scope, entity_id, direction, ts)",
    "monitored_captures": "CREATE INDEX IF NOT EXISTS idx_monitored_scope_ts "
                          "ON monitored_captures (scope, ts, is_day)",
    "memory_observations": "CREATE INDEX IF NOT EXISTS idx_memory_obs_scope_dir "
                           "ON memory_observations (scope, direction)",
}


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


def migrate_scoped_db(conn: sqlite3.Connection,
                      stamp: str = DEFAULT_STAMP) -> dict:
    """Bring a (possibly pre-scope) pipeline DB to the scoped schema.

    Returns a report dict {table: rows_migrated} plus ``status``:
    ``already-scoped`` (no-op), ``migrated`` (work done), ``fresh``
    (created empty scoped tables).

    Idempotent. Additive: existing rows are preserved with scope=stamp.
    Single connection/transaction per table rebuild; a failure mid-way
    aborts before any commit (caller controls the final commit).
    """
    report: dict = {"status": "already-scoped", "tables": {}}
    existing = {t for (t,) in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}

    scoped_targets = [t for t in _SCOPE_TABLES if t in existing]
    if not scoped_targets:
        # Fresh DB: create the scoped schema directly (no legacy default).
        from .memory import MEMORY_SCHEMA

        conn.executescript(MEMORY_SCHEMA)
        report["status"] = "fresh"
        return report

    not_scoped = [t for t in scoped_targets if not _is_scoped(conn, t)]
    if not not_scoped:
        # All present tables already carry scope — verify PKs include it.
        for t in scoped_targets:
            pk = _pk_cols(conn, t)
            if pk[:1] != ["scope"] and _SCOPE_TABLES[t]["pk"]:
                raise RuntimeError(
                    f"{t} has a scope column but PK {pk} does not start with "
                    f"scope — schema is neither legacy nor scoped; refusing "
                    f"to guess (manual reconciliation required)")
        return report

    # -- rebuild path: every not-yet-scoped table is copied + stamped --------
    report["status"] = "migrated"
    for table in not_scoped:
        spec = _SCOPE_TABLES[table]
        cols = list(spec["cols"].keys())
        sel_cols = ", ".join(cols)
        ins_cols = ", ".join(["scope"] + cols)
        placeholders = ", ".join(["?"] * (len(cols) + 1))

        conn.execute(f"ALTER TABLE {table} RENAME TO {table}_legacy")
        _create_scoped(conn, table, spec, stamp)
        rows = conn.execute(
            f"SELECT {sel_cols} FROM {table}_legacy").fetchall()
        # Plain INSERT: the new PK only adds a constant (the stamp) to the old
        # one, so a collision here means legacy data was already corrupt —
        # fail loudly rather than silently drop rows.
        conn.executemany(
            f"INSERT INTO {table}_new ({ins_cols}) VALUES ({placeholders})",
            [(stamp, *row) for row in rows],
        )
        report["tables"][table] = len(rows)
        conn.execute(f"DROP TABLE {table}_legacy")
        conn.execute(f"ALTER TABLE {table}_new RENAME TO {table}")
        if table in _SCOPE_INDEXES:
            conn.execute(_SCOPE_INDEXES[table])

    # Tables that existed but were already scoped (mixed-shape guard).
    for table in scoped_targets:
        if table not in not_scoped:
            report["tables"][table] = 0
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
        target = (Path(backup_dir).resolve() if backup_dir else path.parent) / \
            _backup_path(path).name
        if not target.is_file():
            _snapshot(path, target)
    conn = sqlite3.connect(str(path))
    try:
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
    return 0


if __name__ == "__main__":
    sys.exit(main())