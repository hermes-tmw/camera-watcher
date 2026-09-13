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
        """Parse the API's createdDateTime and return a naive UTC time.

        The browser's ``_fmt_time`` treats the naive ``capture_time`` column as
        container-local UTC (matching the motion path), so we store naive UTC.
        """
        dt = datetime.fromisoformat(created_dt)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=pytz.UTC)
        return dt.astimezone(pytz.UTC).replace(tzinfo=None)

    def _capture_time_mountain(self, created_dt: str) -> datetime:
        """Parse createdDateTime into an aware US/Mountain datetime (for lighting)."""
        dt = datetime.fromisoformat(created_dt)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=pytz.UTC)
        return dt.astimezone(pytz.timezone(CAMERA_TZ))

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
        lighting = self._lighting(self._capture_time_mountain(created_dt))

        cur = self._conn.cursor()
        try:
            cur.execute(
                "INSERT INTO event_observations "
                "(event_name, video_file, capture_time, scene_name, storage_local, "
                " video_location, lighting_type, camera) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (event_name) DO NOTHING RETURNING id",
                (event_name, video_file, capture_time, direction, True,
                 video_location, lighting, "stealthcam"),
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

    def write_note(self, guid: str, subimage: int, direction: str,
                   category: str, confidence: float, note_text: str,
                   created_dt: str, bbox: tuple[float, float, float, float],
                   detection_index: int, tag: str | None = None,
                   make_model_lettering: str = "") -> int:
        """Write a non-alert observation (traffic / background note).

        A pass-by on the road and an expected vehicle in the driveway are
        *noted, not alerted*. This writes the same EventObservation + Labeling +
        IntermediateResult shape as ``write_detection``, but tags the event so
        the dashboard can distinguish it from an alert. ``tag`` is ``traffic``
        or ``background`` (surfaced via the labeling description so the note
        is self-describing). ``make_model_lettering`` carries the VLM-extracted
        make/model + lettering when available.
        """
        event_name = f"{guid}_{subimage}_{detection_index}"
        video_file = f"{guid}_{subimage}.JPG"
        video_location = "stealthcam"
        capture_time = self._capture_time_naive(created_dt)
        lighting = self._lighting(self._capture_time_mountain(created_dt))

        desc_parts = [p for p in (note_text, make_model_lettering) if p]
        description = " | ".join(desc_parts) or None
        if tag:
            description = (f"[{tag}] " + description) if description else f"[{tag}]"

        cur = self._conn.cursor()
        try:
            cur.execute(
                "INSERT INTO event_observations "
                "(event_name, video_file, capture_time, scene_name, storage_local, "
                " video_location, lighting_type, camera) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (event_name) DO NOTHING RETURNING id",
                (event_name, video_file, capture_time, direction, True,
                 video_location, lighting, "stealthcam"),
            )
            row = cur.fetchone()
            if row is None:
                cur.execute("SELECT id FROM event_observations WHERE event_name = %s",
                            (event_name,))
                row = cur.fetchone()
            event_id = row[0]

            cur.execute(
                "INSERT INTO labelings "
                "(decider, decided_at, labels, probabilities, description, git_version, event_id) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                ("yolo11n+vlm", datetime.now(), json.dumps([category]),
                 json.dumps([confidence]), description, self.git_version, event_id),
            )

            relpath = self._subimage_relpath(guid, subimage)
            cur.execute(
                "INSERT INTO intermediate_results "
                "(computed_at, step, info, file, event_id) "
                "VALUES (%s, %s, %s, %s, %s)",
                (datetime.now(), "stealthcam_note",
                 json.dumps({"bbox": list(bbox), "direction": direction,
                             "tag": tag}),
                 relpath, event_id),
            )
            self._conn.commit()
            return event_id
        except Exception:
            self._conn.rollback()
            raise

    def write_telemetry(self, status: dict) -> int:
        """Persist one device-status snapshot into `stealthcam_telemetry`.

        `status` is the `cameraStatuses[0]` dict from the client's
        `get_device_status()`. We store the fields the dashboard's health
        readout needs (battery, disk, signal, sync, errors) plus the raw
        record as JSON for future fields. Returns the new row id.

        The table is append-only (one row per poll) so battery/disk trends
        are queryable over time; the dashboard reads the latest row.
        """
        cur = self._conn.cursor()
        try:
            cur.execute(
                "INSERT INTO stealthcam_telemetry "
                "(device_id, captured_at, battery_pct, battery_volt, "
                " external_battery_pct, external_battery_volt, "
                " sd_card_free_pct, rssi, signal_strength, firmware_version, "
                " last_sync_at, on_demand_state, errors, raw) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "RETURNING id",
                (
                    status.get("physicalDeviceIdentifier"),
                    datetime.now(),
                    status.get("batteryLevel"),
                    status.get("batteryVolt"),
                    status.get("externalBatteryLevel"),
                    status.get("externalBatteryVolt"),
                    status.get("sdCardFreeSpace"),
                    status.get("rssi"),
                    status.get("signalStrength"),
                    status.get("firmwareVersion"),
                    self._sync_time_naive(status.get("lastSyncDateUnixTime")),
                    (status.get("deviceState") or {}).get("onDemandState"),
                    json.dumps(status.get("errors") or []),
                    json.dumps(status),
                ),
            )
            row = cur.fetchone()
            self._conn.commit()
            if row is None:
                raise RuntimeError("write_telemetry: no id returned")
            return row[0]
        except Exception:
            self._conn.rollback()
            raise

    @staticmethod
    def _sync_time_naive(ms: int | None) -> datetime | None:
        """Convert a lastSyncDateUnixTime (ms epoch) to a naive UTC time.

        ``last_sync_at`` is rendered by the browser's ``_fmt_time``, which
        treats the naive value as container-local UTC (same convention as
        ``capture_time``), so we store naive UTC.
        """
        if not ms:
            return None
        dt = datetime.fromtimestamp(ms / 1000.0, tz=pytz.UTC)
        return dt.astimezone(pytz.UTC).replace(tzinfo=None)

    def close(self) -> None:
        self._conn.close()
