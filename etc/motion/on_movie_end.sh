#!/bin/sh
# on_movie_end.sh — motion 5 hook wrapper. $1=event_name(%C).
# Enqueues significant-frame save for the video worker.
set -e
: "${REDIS_URL:=redis://redis}"
exec /usr/local/bin/rq enqueue --retry-interval=10 --retry-max=3 -q event_video -u "$REDIS_URL" \
  'watcher.video.task_save_significant_frame' "$1"
