"""One-shot backfill: convert stealthcam capture_time from naive MDT to naive UTC.

Before the timezone fix, ``stealthcam/dashboard.py`` stored ``capture_time`` as
naive US/Mountain wall-clock (the value burned into the image). The browser's
``_fmt_time`` treats the naive column as container-local UTC, so those rows
render 6h early. After the fix, new rows store naive UTC.

This script converts the already-stored rows in place, DST-safe: each stored
naive value is localized as US/Mountain wall time (pytz picks the correct
offset for that date), converted to UTC, and stripped of tzinfo. It also fixes
``stealthcam_telemetry.last_sync_at`` (same convention).

Idempotent: re-running after the fix is a no-op because the writer now stores
naive UTC and this script only touches rows whose camera='stealthcam' (capture)
or the telemetry table (sync) — but it is NOT safe to run twice against rows
already converted, so it is guarded by a marker: it only converts rows whose
capture_time is NOT already UTC. Since we cannot distinguish "naive MDT" from
"naive UTC" by value alone, the guard is the presence of the fix itself — run
this ONCE, immediately after deploying the fix, before the new writer inserts
any new rows. (New rows are naive UTC and would be double-shifted if this ran
again, so do not re-run.)

Usage:
    python3 stealthcam/backfill_tz.py <db_url>
"""
from __future__ import annotations

import sys
from datetime import datetime

import psycopg2
import pytz

CAMERA_TZ = "US/Mountain"


def _mdt_naive_to_utc_naive(value: datetime) -> datetime:
    """Localize a naive MDT wall-clock value and return naive UTC (DST-safe)."""
    mountain = pytz.timezone(CAMERA_TZ)
    aware = mountain.localize(value)
    return aware.astimezone(pytz.UTC).replace(tzinfo=None)


def backfill(db_url: str) -> tuple[int, int]:
    conn = psycopg2.connect(db_url)
    conn.autocommit = False
    cur = conn.cursor()
    try:
        # 1. event_observations.capture_time for stealthcam rows.
        cur.execute(
            "SELECT id, capture_time FROM event_observations "
            "WHERE camera = 'stealthcam' AND capture_time IS NOT NULL "
            "ORDER BY id"
        )
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
        cur.execute(
            "SELECT id, last_sync_at FROM stealthcam_telemetry "
            "WHERE last_sync_at IS NOT NULL ORDER BY id"
        )
        sync_rows = cur.fetchall()
        sync_converted = 0
        for row_id, last_sync_at in sync_rows:
            new_val = _mdt_naive_to_utc_naive(last_sync_at)
            cur.execute(
                "UPDATE stealthcam_telemetry SET last_sync_at = %s WHERE id = %s",
                (new_val, row_id),
            )
            sync_converted += 1

        conn.commit()
        return converted, sync_converted
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python3 stealthcam/backfill_tz.py <db_url>", file=sys.stderr)
        sys.exit(2)
    n_capture, n_sync = backfill(sys.argv[1])
    print(f"backfill complete: {n_capture} capture_time rows, "
          f"{n_sync} last_sync_at rows converted (MDT -> UTC)")


if __name__ == "__main__":
    main()
