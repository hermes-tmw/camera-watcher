#!/usr/bin/env python3
"""
Consolidate video files into /data/video/watcher/<site>/

Sources:
  /data/video/<site>/       — top-level NFS site directories
  /data/video_local/<site>/ — locally-recorded clips
  /data/video/wichitaDriveway/ — current motion output (pre-config-fix)

All content is merged into /data/video/watcher/<site>/YYYY/MM/DD/
Files that already exist at the destination are skipped.
Empty source directories are removed after the move.

Usage:
  python3 consolidate_video.py          # dry run — shows what would happen
  python3 consolidate_video.py --run    # actually move the files
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

DEST_BASE = Path('/data/video/watcher')

# (source_path, destination_site_name)
# Ordered so site-level moves happen before deeper merges
SOURCES = [
    # Top-level NFS directories that are camera sites
    (Path('/data/video/gopro'),           'gopro'),
    # wichitaDriveway from two sources — both merge into same destination
    (Path('/data/video/wichitaDriveway'), 'wichitaDriveway'),
    (Path('/data/video_local/wichitaDriveway'), 'wichitaDriveway'),
]


def move_tree(src: Path, dst: Path, dry_run: bool) -> tuple[int, int]:
    """Recursively merge src into dst using day-level mv where possible.

    Returns (moved_dirs, moved_files).
    """
    if not src.exists():
        return 0, 0

    moved_dirs = moved_files = 0

    try:
        children = sorted(src.iterdir())
    except PermissionError:
        print(f'  PERMISSION DENIED: {src}')
        return 0, 0

    for item in children:
        dst_item = dst / item.name

        if item.is_file():
            if dst_item.exists():
                print(f'  skip  {item.relative_to(Path("/data"))}  (already exists)')
            else:
                print(f'  mv    {item.relative_to(Path("/data"))}')
                if not dry_run:
                    dst.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(item), str(dst_item))
                moved_files += 1

        elif item.is_dir():
            if not dst_item.exists():
                # Destination doesn't exist — move the whole directory at once
                print(f'  mv -r {item.relative_to(Path("/data"))}  ->  {dst_item.relative_to(Path("/data"))}/')
                if not dry_run:
                    dst.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(item), str(dst_item))
                moved_dirs += 1
            else:
                # Destination exists — recurse to merge
                d, f = move_tree(item, dst_item, dry_run)
                moved_dirs += d
                moved_files += f

    return moved_dirs, moved_files


def remove_empty_dirs(path: Path, dry_run: bool):
    """Remove empty directories bottom-up."""
    if not path.exists() or not path.is_dir():
        return
    for child in sorted(path.iterdir()):
        if child.is_dir():
            remove_empty_dirs(child, dry_run)
    if not any(path.iterdir()):
        print(f'  rmdir {path.relative_to(Path("/data"))}')
        if not dry_run:
            path.rmdir()


def main():
    parser = argparse.ArgumentParser(description='Consolidate video files into /data/video/watcher/')
    parser.add_argument('--run', action='store_true', help='Actually move files (default is dry-run)')
    args = parser.parse_args()
    dry_run = not args.run

    if dry_run:
        print('DRY RUN — pass --run to actually move files\n')
    else:
        print('MOVING FILES\n')

    total_dirs = total_files = 0

    for src, site in SOURCES:
        dst = DEST_BASE / site
        if not src.exists():
            print(f'[skip] {src} does not exist')
            continue

        print(f'\n[{site}]  {src}  ->  {dst}')
        d, f = move_tree(src, dst, dry_run)
        total_dirs += d
        total_files += f

        if not dry_run:
            remove_empty_dirs(src, dry_run)

    print(f'\n{"Would move" if dry_run else "Moved"}: {total_dirs} directories, {total_files} files')
    if dry_run:
        print('Run with --run to apply.')


if __name__ == '__main__':
    main()
