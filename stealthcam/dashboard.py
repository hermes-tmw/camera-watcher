"""Dashboard: map Stealthcam detections into camera-watcher's schema.

Writes EventObservation + Labeling + IntermediateResult rows into camera-watcher's
live Postgres so the existing /browser dashboard renders them as-is.

Mapping (per design doc §3.8):
  - EventObservation: capture_time, scene_name (direction), event_name
    (guid + subimage + detection index), video_file/video_location (sub-image
    path under LOCAL_DATA_DIR).
  - Labeling: labels (YOLO category), probabilities (confidence), description
    (VLM description), decider ('yolo11n+vlm'), git_version.
  - IntermediateResult: file = sub-image path (drives significant_frame_url).

Uses psycopg2 directly (no camera-watcher import — avoids its heavy deps).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import psycopg2
import pytz

CAMERA_TZ = "US/Mountain"


class DashboardWriter:
    def __init__(self, db_url: str, local_data_dir: str = "/data/video/watcher",
                 base_static_url: str = "http://mira.local", git_version: str = "unknown"):
        self.db_url = db_url
        self.local_data_dir = local_data_dir.rstrip("/")
        self.base_static_url = base_static_url.rstrip("/")
        self.git_version = git_version
        self._conn = psycopg2.connect(db_url)
        self._conn.autocommit = False

    # -- helpers -------------------------------------------------------------

    def _subimage_relpath(self, guid: str, n: int) -> str:
        return f"stealthcam/{guid}_{n}.JPG"

    def _capture_time_naive(self, created_dt: str) -> datetime:
        """Parse the API's createdDateTime and return a naive US/Mountain time."""
        dt = datetime.fromisoformat(created_dt)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=pytz.UTC)
        return dt.astimezone(pytz.timezone(CAMERA_TZ)).replace(tzinfo=None)

    def _lighting(self, dt: datetime) -> str:
        return "daylight" if 7 <= dt.hour < 20 else "night"

    # -- write ---------------------------------------------------------------

    def write_detection(self, guid: str, subimage: int, direction: str,
                        category: str, confidence: float, description: str,
                        created_dt: str, bbox: tuple[float, float, float, float],
                        detection_index: int) -> int:
        """Write one detection as an EventObservation + Labeling + IntermediateResult.

        Returns the new event_observations.id.
        """
        event_name = f"{guid}_{subimage}_{detection_index}"
        video_file = f"{guid}_{subimage}.JPG"
        video_location = "stealthcam"
        capture_time = self._capture_time_naive(created_dt)
        lighting = self._lighting(capture_time)

        cur = self._conn.cursor()
        try:
            cur.execute(
                "INSERT INTO event_observations "
                "(event_name, video_file, capture_time, scene_name, storage_local, "
                " video_location, lighting_type) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (event_name) DO NOTHING RETURNING id",
                (event_name, video_file, capture_time, direction, True,
                 video_location, lighting),
            )
            row = cur.fetchone()
            if row is None:
                # already exists — fetch its id
                cur.execute("SELECT id FROM event_observations WHERE event_name = %s",
                            (event_name,))
                row = cur.fetchone()
            event_id = row[0]

            labels = [category]
            probabilities = [confidence]
            cur.execute(
                "INSERT INTO labelings "
                "(decider, decided_at, labels, probabilities, description, git_version, event_id) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                ("yolo11n+vlm", datetime.now(), json.dumps(labels),
                 json.dumps(probabilities), description or None, self.git_version, event_id),
            )

            relpath = self._subimage_relpath(guid, subimage)
            cur.execute(
                "INSERT INTO intermediate_results "
                "(computed_at, step, info, file, event_id) "
                "VALUES (%s, %s, %s, %s, %s)",
                (datetime.now(), "stealthcam_detect",
                 json.dumps({"bbox": list(bbox), "direction": direction}),
                 relpath, event_id),
            )
            self._conn.commit()
            return event_id
        except Exception:
            self._conn.rollback()
            raise

    def close(self) -> None:
        self._conn.close()
