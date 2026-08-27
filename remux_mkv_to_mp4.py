#!/usr/bin/env python3
"""Backfill: remux motion-5 .mkv clips to browser-playable .mp4 (GH#29).

Motion 5 writes WichitaDriveway clips as Matroska (.mkv) with pcm_mulaw
audio. Browsers cannot decode that container+audio combination, so the
dashboard's <video> player does nothing. The h264 video stream itself is
fine, so a lossless stream-copy into an mp4 container (dropping the audio
track) is sufficient.

For each event_observations row whose video_file ends in .mkv:
  1. locate the file at /data/video/watcher/<video_location>/<video_file>
  2. remux to <same-basename>.mp4  (ffmpeg -c:v copy -an)
  3. update the row's video_file to the .mp4 name
  4. delete the original .mkv only after the remux + DB update succeed

The DB update and the file remux are ordered so a crash never leaves a row
pointing at a deleted file: the .mkv is removed last, and only after the row
has been repointed at the verified .mp4.

Usage:
  python3 remux_mkv_to_mp4.py            # dry run — report only
  python3 remux_mkv_to_mp4.py --run      # actually remux + update + delete
  python3 remux_mkv_to_mp4.py --limit N  # process at most N rows (either mode)

DB connection is read from etc/watcher.cfg [database] (DB_HOST/DB_USER/
DB_PASS/DB_NAME). Run from the repo root.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import psycopg2

REPO_ROOT = Path(__file__).resolve().parent
LOCAL_DATA_DIR = Path('/data/video/watcher')

FFMPEG_TIMEOUT_SECONDS = 600


def db_config():
    """Read [database] from etc/watcher.cfg into a dict."""
    import configparser
    cfg = configparser.ConfigParser()
    cfg.read(REPO_ROOT / 'etc' / 'watcher.cfg')
    db = cfg['database']
    return {
        'host': db.get('DB_HOST'),
        'user': db.get('DB_USER'),
        'password': db.get('DB_PASS'),
        'dbname': db.get('DB_NAME'),
    }


def fetch_mkv_rows(conn, limit):
    cur = conn.cursor()
    if limit:
        cur.execute(
            "SELECT id, video_file, video_location FROM event_observations "
            "WHERE video_file LIKE '%%.mkv' ORDER BY id LIMIT %s", (limit,))
    else:
        cur.execute(
            "SELECT id, video_file, video_location FROM event_observations "
            "WHERE video_file LIKE '%%.mkv' ORDER BY id")
    return cur.fetchall()


def remux(src: Path, dst: Path) -> None:
    """Stream-copy h264 into mp4, dropping audio. Raises on failure."""
    cmd = [
        'ffmpeg', '-y', '-i', str(src),
        '-c:v', 'copy', '-an',
        '-movflags', '+faststart',
        str(dst),
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_SECONDS)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed ({proc.returncode}) for {src}:\n"
            f"STDERR: {proc.stderr[-2000:]}")
    if not dst.exists() or dst.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg produced no output for {src}")


def update_row(conn, row_id, new_name):
    cur = conn.cursor()
    cur.execute(
        "UPDATE event_observations SET video_file = %s WHERE id = %s",
        (new_name, row_id))
    if cur.rowcount != 1:
        raise RuntimeError(
            f"expected 1 row updated for id {row_id}, got {cur.rowcount}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true',
                        help='actually remux + update + delete (default dry-run)')
    parser.add_argument('--limit', type=int, default=None,
                        help='process at most N rows')
    args = parser.parse_args()

    cfg = db_config()
    conn = psycopg2.connect(**cfg)
    conn.autocommit = False

    rows = fetch_mkv_rows(conn, args.limit)
    print(f"{'DRY RUN' if not args.run else 'RUNNING'}: {len(rows)} .mkv rows "
          f"in event_observations")

    remuxed = skipped_missing = failed = 0
    for row_id, video_file, video_location in rows:
        src = LOCAL_DATA_DIR / (video_location or '') / video_file
        dst = src.with_suffix('.mp4')
        new_name = dst.name

        if not src.exists():
            print(f"  [skip] missing file: {src}")
            skipped_missing += 1
            continue

        if dst.exists():
            # Already remuxed (idempotent re-run). Just repoint the row.
            print(f"  [exists] {src.name} -> {new_name} (mp4 already present)")
        else:
            print(f"  [remux] {src.name} -> {new_name}")
            if args.run:
                try:
                    remux(src, dst)
                except Exception as e:
                    print(f"  [FAIL] {src}: {e}", file=sys.stderr)
                    failed += 1
                    continue

        if args.run:
            try:
                update_row(conn, row_id, new_name)
                conn.commit()
                # Delete the source only after the row is safely repointed.
                if src.exists():
                    src.unlink()
                remuxed += 1
            except Exception as e:
                conn.rollback()
                print(f"  [FAIL] DB update for id {row_id}: {e}", file=sys.stderr)
                failed += 1

    conn.close()
    print(f"\nDone. remuxed={remuxed} skipped_missing={skipped_missing} "
          f"failed={failed}")
    if not args.run:
        print("Dry run — pass --run to apply.")


if __name__ == '__main__':
    main()
