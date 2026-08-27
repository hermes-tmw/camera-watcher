#!/bin/sh
# on_movie_start.sh — motion 5 hook wrapper.
# Emits an EventObservation payload matching watcher.model schema + historical
# Wichita records. lighting_type is intentionally OMITTED so the model derives
# it from capture_time via astral (sunlight_from_time_for_location) — hardcoding
# "night" here (GH#23) mislabeled every daytime capture.
#
# Args from motion (see motion.conf on_movie_start):
#   $1 = video_fullpath (%f)
#   $2 = capture_time    (%Y-%m-%dT%T)
#   $3 = scene_name      (%$)
#   $4 = event_name      (%C — text_event, e.g. 20260826_214401_WichitaDriveway_114)
#   $5 = threshold       (%o)
#   $6 = noise_level     (%N)
set -e
: "${REDIS_URL:=redis://redis}"

video_fullpath="$1"
capture_time="$2"
scene_name="$3"
event_name="$4"
# %o/%N are numeric but %N is zero-padded to 2 digits (e.g. "05"); normalize
# with printf '%d' so the JSON stays valid (no leading zeros).
threshold=$(printf '%d' "${5:-0}")
noise_level=$(printf '%d' "${6:-0}")

video_file="$(basename "$video_fullpath")"                        # 20260826_214401_WichitaDriveway_114_4401.mkv
video_dir="$(dirname "$video_fullpath")"                          # /data/video/watcher/wichitaDriveway/2026/08/26
# relative to /data/video/watcher -> wichitaDriveway/2026/08/26
video_location="${video_dir#/data/video/watcher/}"
video_location="${video_location#/}"

payload=$(printf '{"video_file": "%s","video_location": "%s","storage_local": true,"capture_time": "%s","scene_name": "%s","camera": "%s","event_name": "%s","threshold": %s,"noise_level": %s}' \
  "$video_file" "$video_location" "$capture_time" "$scene_name" "$scene_name" "$event_name" "$threshold" "$noise_level")

exec /usr/local/bin/rq enqueue --retry-interval=5 --retry-max=3 -q record_event -u "$REDIS_URL" \
  'watcher.lite_tasks.task_record_event' 'EventObservation' "$payload"
