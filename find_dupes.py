#!/usr/bin/env python3
"""Find duplicate video files across multiple directories."""

import os
import sys
from collections import defaultdict

import click


def human_size(num_bytes):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} PB"


def collect_files(root_dirs, extensions):
    """Walk directories and collect files matching extensions.

    Returns dict: filename -> list of (full_path, size, root_dir)
    """
    ext_set = {e.lower().lstrip(".") for e in extensions}
    index = defaultdict(list)

    for root_dir in root_dirs:
        root_dir = os.path.abspath(root_dir)
        for dirpath, _dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
                if ext not in ext_set:
                    continue
                full_path = os.path.join(dirpath, filename)
                try:
                    size = os.path.getsize(full_path)
                except OSError:
                    continue
                index[filename].append((full_path, size, root_dir))

    return index


def relative_path(full_path, root_dir):
    return os.path.relpath(full_path, root_dir)


@click.command()
@click.argument("dirs", nargs=-1, required=True, metavar="DIR [DIR ...]")
@click.option(
    "--extensions",
    "-e",
    multiple=True,
    default=["mp4", "avi", "mov", "mkv", "jpg"],
    show_default=True,
    help="File extensions to scan.",
)
@click.option(
    "--size-only",
    is_flag=True,
    default=False,
    help="Only show duplicates confirmed by size match.",
)
@click.option(
    "--tree",
    is_flag=True,
    default=False,
    help="Show relative path within each root for context.",
)
def main(dirs, extensions, size_only, tree):
    """Find duplicate files across DIR(s).

    Duplicates are identified by matching filename and file size.
    """
    # Validate directories
    for d in dirs:
        if not os.path.isdir(d):
            click.echo(f"Error: '{d}' is not a directory or does not exist.", err=True)
            sys.exit(2)

    click.echo(f"Scanning {len(dirs)} director{'y' if len(dirs) == 1 else 'ies'}...")
    index = collect_files(dirs, extensions)

    confirmed = []   # (filename, size, entries)
    collisions = []  # (filename, entries)

    for filename, entries in index.items():
        if len(entries) < 2:
            continue
        sizes = {e[1] for e in entries}
        if len(sizes) == 1:
            confirmed.append((filename, entries[0][1], entries))
        else:
            collisions.append((filename, entries))

    confirmed.sort(key=lambda x: x[0])
    collisions.sort(key=lambda x: x[0])

    found_any = bool(confirmed) or (bool(collisions) and not size_only)

    # Report confirmed duplicates
    if confirmed:
        click.echo("")
        for filename, size, entries in confirmed:
            # Subtract one copy — the rest are "wasted"
            click.secho(
                f"CONFIRMED DUPLICATE: {filename} ({human_size(size)})",
                fg="red",
            )
            for full_path, _size, root_dir in entries:
                click.echo(f"  {full_path}")
                if tree:
                    rel = relative_path(full_path, root_dir)
                    click.echo(f"    [in {root_dir}] → {rel}")

    # Report name collisions (same name, different sizes)
    if not size_only and collisions:
        click.echo("")
        for filename, entries in collisions:
            click.secho(f"NAME COLLISION: {filename}", fg="yellow")
            for full_path, size, root_dir in entries:
                click.echo(f"  {full_path}  ({human_size(size)})")
                if tree:
                    rel = relative_path(full_path, root_dir)
                    click.echo(f"    [in {root_dir}] → {rel}")

    # Summary
    click.echo("")
    total_wasted = sum(size * (len(entries) - 1) for _, size, entries in confirmed)
    click.echo(
        f"Summary: {len(confirmed)} confirmed duplicate(s)"
        + (f", {human_size(total_wasted)} wasted space" if confirmed else "")
        + (
            f"; {len(collisions)} name collision(s)"
            if not size_only and collisions
            else ""
        )
    )

    sys.exit(1 if found_any else 0)


if __name__ == "__main__":
    main()
