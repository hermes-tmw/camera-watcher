#!/bin/sh
# on_movie_start.sh — motion 5 hook wrapper.
# Emits EventObservation matching watcher.model schema + historical Wichita records:
#   video_file (basename), video_location (relative dir under /data/video/watcher),
#   storage_local, capture_time, scene_name, lighting_type, camera, event_name
# Args from motion: $1=video_fullpath(%f) $2=capture_time(%Y-%m-%dT%T) $3=scene_name(%$)
set -e
: "${REDIS_URL:=redis://redis}"

video_fullpath="$1"
capture_time="$2"
scene_name="$3"

video_file="$(basename "$video_fullpath")"                        # 2026..._01_1830.mkv
video_dir="$(dirname "$video_fullpath")"                          # /data/video/watcher/wichitaDriveway/2026/08/26
# relative to /data/video/watcher -> wichitaDriveway/2026/08/26
video_location="${video_dir#/data/video/watcher/}"
video_location="${video_location#/}"
camera="WichitaDriveway"
event_name="$(echo "$video_file" | sed 's/_[0-9]\{4\}\.[a-zA-Z0-9]*$//')"

# lighting_type is intentionally OMITTED from the payload: the EventObservation
# model derives it from capture_time via astral (sunlight_from_time_for_location).
# Hardcoding "night" here (GH#23) mislabeled every daytime capture.
payload=$(printf '{"video_file": "%s","video_location": "%s","storage_local": true,"capture_time": "%s","scene_name": "%s","camera": "%s","event_name": "%s","threshold": 0,"noise_level": 0}' \
  "$video_file" "$video_location" "$capture_time" "$scene_name" "$camera" "$event_name")

exec /usr/local/bin/rq enqueue --retry-interval=5 --retry-max=3 -q record_event -u "$REDIS_URL" \
  'watcher.lite_tasks.task_record_event' 'EventObservation' "$payload"
