"""One-shot backfill: convert stealthcam capture_time from naive MDT to naive UTC.

Before the timezone fix, ``stealthcam/dashboard.py`` stored ``capture_time`` as
naive US/Mountain wall-clock (the value burned into the image). The browser's
``_fmt_time`` treats the naive column as container-local UTC, so those rows
render 6h early. After the fix, new rows store naive UTC.

This script converts the already-stored rows in place, DST-safe: each stored
naive value is localized as US/Mountain wall time (pytz picks the correct
offset for that date), converted to UTC, and stripped of tzinfo. It also fixes
``stealthcam_telemetry.last_sync_at`` (same convention).

NOT idempotent. Running twice double-shifts every row +6h (silent, committed).
Run exactly once. Two structural guards protect against that:

1. A marker row in ``_backfill_markers`` records a completed run; the script
   refuses to run again while the marker is present (pass ``--force`` to
   override, e.g. after restoring a pre-fix snapshot).
2. ``--max-event-id`` / ``--max-telemetry-id`` bound the conversion to rows
   that existed before the fix deployed, so post-deploy rows written by the
   new (naive-UTC) writer are never double-shifted. The telemetry writer
   inserts a row every ~5 minutes, so without a cutoff the deploy/backfill
   ordering is impossible for telemetry.

Usage:
    python3 stealthcam/backfill_tz.py <db_url> --yes \\
        --max-event-id <N> --max-telemetry-id <N>
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime

import psycopg2
import pytz

CAMERA_TZ = "US/Mountain"
MARKER_KEY = "tz_backfill_v1"

_MARKER_SCHEMA = """
CREATE TABLE IF NOT EXISTS _backfill_markers (
    key TEXT PRIMARY KEY,
    completed_at TIMESTAMP NOT NULL,
    event_rows INTEGER NOT NULL,
    telemetry_rows INTEGER NOT NULL
);
"""


def _mdt_naive_to_utc_naive(value: datetime) -> datetime:
    """Localize a naive MDT wall-clock value and return naive UTC (DST-safe)."""
    mountain = pytz.timezone(CAMERA_TZ)
    aware = mountain.localize(value)
    return aware.astimezone(pytz.UTC).replace(tzinfo=None)


def _marker_exists(cur) -> bool:
    cur.execute("SELECT 1 FROM _backfill_markers WHERE key = %s", (MARKER_KEY,))
    return cur.fetchone() is not None


def backfill(db_url: str, max_event_id: int | None = None,
             max_telemetry_id: int | None = None,
             force: bool = False) -> tuple[int, int]:
    conn = psycopg2.connect(db_url)
    conn.autocommit = False
    cur = conn.cursor()
    try:
        cur.execute(_MARKER_SCHEMA)
        if _marker_exists(cur) and not force:
            raise RuntimeError(
                "backfill already ran (marker present). Refusing to re-run: "
                "this script is NOT idempotent and would double-shift all rows. "
                "Pass --force to override (only after restoring a pre-fix snapshot)."
            )

        # 1. event_observations.capture_time for stealthcam rows.
        query = (
            "SELECT id, capture_time FROM event_observations "
            "WHERE camera = 'stealthcam' AND capture_time IS NOT NULL"
        )
        params: list = []
        if max_event_id is not None:
            query += " AND id <= %s"
            params.append(max_event_id)
        query += " ORDER BY id"
        cur.execute(query, params)
        rows = cur.fetchall()
        converted = 0
        for row_id, capture_time in rows:
            new_val = _mdt_naive_to_utc_naive(capture_time)
            cur.execute(
                "UPDATE event_observations SET capture_time = %s WHERE id = %s",
                (new_val, row_id),
            )
            converted += 1

        # 2. stealthcam_telemetry.last_sync_at.
        query = (
            "SELECT id, last_sync_at FROM stealthcam_telemetry "
            "WHERE last_sync_at IS NOT NULL"
        )
        params = []
        if max_telemetry_id is not None:
            query += " AND id <= %s"
            params.append(max_telemetry_id)
        query += " ORDER BY id"
        cur.execute(query, params)
        sync_rows = cur.fetchall()
        sync_converted = 0
        for row_id, last_sync_at in sync_rows:
            new_val = _mdt_naive_to_utc_naive(last_sync_at)
            cur.execute(
                "UPDATE stealthcam_telemetry SET last_sync_at = %s WHERE id = %s",
                (new_val, row_id),
            )
            sync_converted += 1

        cur.execute(
            "INSERT INTO _backfill_markers "
            "(key, completed_at, event_rows, telemetry_rows) "
            "VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (key) DO UPDATE SET "
            "completed_at = EXCLUDED.completed_at, "
            "event_rows = EXCLUDED.event_rows, "
            "telemetry_rows = EXCLUDED.telemetry_rows",
            (MARKER_KEY, datetime.now(), converted, sync_converted),
        )
        conn.commit()
        return converted, sync_converted
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-shot MDT->UTC backfill for stealthcam capture_time "
                    "and telemetry last_sync_at. NOT idempotent — run exactly once."
    )
    parser.add_argument("db_url", help="Postgres connection URL")
    parser.add_argument("--yes", action="store_true",
                        help="Confirm the destructive one-shot run.")
    parser.add_argument("--max-event-id", type=int, default=None,
                        help="Only convert event_observations rows with id <= N "
                             "(pre-deploy cutoff).")
    parser.add_argument("--max-telemetry-id", type=int, default=None,
                        help="Only convert stealthcam_telemetry rows with id <= N "
                             "(pre-deploy cutoff).")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if the completion marker is present "
                             "(only after restoring a pre-fix snapshot).")
    args = parser.parse_args()

    if not args.yes:
        print("Refusing to run without --yes: this is a destructive, "
              "non-idempotent one-shot backfill.", file=sys.stderr)
        sys.exit(2)

    n_capture, n_sync = backfill(
        args.db_url,
        max_event_id=args.max_event_id,
        max_telemetry_id=args.max_telemetry_id,
        force=args.force,
    )
    print(f"backfill complete: {n_capture} capture_time rows, "
          f"{n_sync} last_sync_at rows converted (MDT -> UTC)")


if __name__ == "__main__":
    main()
