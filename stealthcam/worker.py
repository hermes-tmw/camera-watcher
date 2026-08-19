"""Worker: poll -> detect -> describe -> memory -> alert + dashboard.

The main loop polls images-list every 60s (exponential backoff on failures /
empty polls), downloads new captures' 6 sub-images, runs the pipeline, and
dispatches alerts to Mattermost + writes dashboard rows.

Run:  python -m stealthcam.worker
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from .client import StealthcamClient, StealthcamError
from .config import load_config
from .detector import Detector
from .describer import Describer
from .memory import Memory, ALERT, FLAG
from .annotate import annotate
from .notify import Notifier
from .dashboard import DashboardWriter

log = logging.getLogger("stealthcam.worker")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )


def _short_direction(direction: str) -> str:
    # "East_SouthEast" -> "East"
    return direction.split("_")[0]


def process_capture(capture: dict, client: StealthcamClient, detector: Detector,
                    describer: Describer, memory: Memory, notifier: Notifier,
                    dashboard: DashboardWriter | None, subimage_dir: Path,
                    baseline_mode: bool = False) -> dict:
    """Process one capture (6 sub-images). Returns a summary dict.

    In `baseline_mode`, we detect + record memory observations but do NOT
    describe, alert, or write dashboard rows — this seeds the persistence
    baseline (so the parked tractor/car are already "static" before the first
    live alert).
    """
    guid = capture["imageGuid"]
    directions = capture.get("imageDirections") or [f"dir{i}" for i in range(1, 7)]
    created_dt = capture.get("createdDateTime", "")

    summary = {"guid": guid, "alerts": 0, "flags": 0, "suppressed": 0, "detections": 0}

    for n in range(1, 7):
        direction = directions[n - 1] if n - 1 < len(directions) else f"dir{n}"
        local = subimage_dir / f"{guid}_{n}.JPG"
        try:
            client.download_subimage(guid, n, local)
        except StealthcamError as e:
            log.error("download %s_%d failed: %s", guid, n, e)
            continue

        from PIL import Image
        img = Image.open(local)
        detections = detector.detect(img)

        # describe each detection (crop + VLM) — skipped in baseline mode
        descriptions: dict[int, str] = {}
        if not baseline_mode:
            for i, det in enumerate(detections):
                x1, y1, x2, y2 = [int(v) for v in det.bbox]
                crop = img.crop((x1, y1, x2, y2))
                try:
                    descriptions[i] = describer.describe(crop)
                except Exception as e:
                    log.warning("describe %s_%d det %d failed: %s", guid, n, i, e)
                    descriptions[i] = ""

        # memory decision per detection — keyed on the CAPTURE's timestamp, not
        # wall-clock processing time (persistence is measured in capture time).
        now = created_dt or None
        alert_dets = []
        for i, det in enumerate(detections):
            decision = memory.decide(direction, det.cls, det.bbox, now=now)
            summary["detections"] += 1
            if decision.decision == ALERT:
                summary["alerts"] += 1
                alert_dets.append((i, det, decision))
            elif decision.decision == FLAG:
                summary["flags"] += 1
            else:
                summary["suppressed"] += 1

            memory.record_decision(
                guid, n, direction, det.cls, det.confidence, det.bbox,
                descriptions.get(i, ""), decision,
            )

            # write dashboard row for every detection (person/vehicle/animal)
            if (not baseline_mode and dashboard is not None
                    and det.category in ("person", "vehicle", "animal")):
                try:
                    dashboard.write_detection(
                        guid, n, direction, det.category, det.confidence,
                        descriptions.get(i, ""), created_dt, det.bbox, i,
                    )
                except Exception as e:
                    log.error("dashboard write %s_%d det %d failed: %s", guid, n, i, e)

        # alert: annotate + post (skipped in baseline mode)
        if alert_dets and not baseline_mode:
            alert_detections = [detections[i] for i, _, _ in alert_dets]
            alert_desc = {idx: descriptions[i] for idx, (i, _, _) in enumerate(alert_dets)}
            annotated = subimage_dir / f"{guid}_{n}_alert.JPG"
            annotate(img, alert_detections, alert_desc, dest=annotated)

            for i, det, decision in alert_dets:
                label = "Human" if det.category == "person" else det.category.capitalize()
                text = (
                    f"{label} detected — {_short_direction(direction)}, "
                    f"{det.confidence:.2f}, \"{descriptions.get(i, '')}\", "
                    f"{created_dt}"
                )
                try:
                    notifier.post_alert(text, annotated)
                except Exception as e:
                    log.error("notify failed: %s", e)

    memory.mark_seen(guid)
    return summary


def backfill(cfg, client: StealthcamClient, detector: Detector, memory: Memory,
             subimage_dir: Path, max_captures: int = 200) -> int:
    """Seed the memory baseline from recent captures (no alerts, no dashboard).

    Pages back through images-list and processes captures in baseline mode so
    persistent objects (parked tractor/car) are already "static" before the
    live loop starts. Returns the number of captures backfilled.

    Captures are processed in CHRONOLOGICAL order (oldest first). The
    images-list endpoint returns newest-first, and static detection measures
    persistence as (capture_count >= N) AND (span >= M hours) keyed on each
    capture's own `createdDateTime`. Processing newest-first means the first
    sighting of a persistent object is recorded with a *recent* timestamp, so
    the span never reaches M hours and the object is re-alerted as "novel" on
    every capture (the parked-tractor bug). Sorting oldest-first makes the
    first sighting carry the earliest timestamp, so the span grows correctly
    and the object converges to static.
    """
    log.info("backfill: seeding baseline (up to %d captures)", max_captures)
    captures: list[dict] = []
    cursor = None
    while len(captures) < max_captures:
        images = client.list_images(take_count=50, cursor=cursor)
        if not images:
            break
        captures.extend(images)
        cursor = {
            "createdDateTime": images[-1]["createdDateTime"],
            "uploadedDateTime": images[-1].get("uploadedTime"),
        }
        if len(images) < 50:
            break
    # oldest-first so persistence spans grow correctly (see docstring).
    captures.sort(key=lambda c: c.get("createdDateTime", ""))
    captures = captures[:max_captures]

    processed = 0
    for capture in captures:
        if memory.has_seen(capture["imageGuid"]):
            continue
        try:
            process_capture(capture, client, detector, None, memory, None, None,
                            subimage_dir, baseline_mode=True)
            processed += 1
        except Exception as e:
            log.error("backfill capture %s failed: %s", capture["imageGuid"], e)
    log.info("backfill done: %d captures", processed)
    return processed


def run_once(cfg) -> dict:
    client = StealthcamClient(secrets_path=cfg.secrets_stealthcam)
    client.login()

    detector = Detector(cfg.detector_weights, cfg.detector_conf_threshold, cfg.detector_device)
    describer = Describer(cfg.describer_model, cfg.describer_ollama_host, cfg.describer_timeout)
    memory = Memory(cfg.pipeline_db, cfg.memory_iou_threshold,
                    cfg.memory_static_n_captures, cfg.memory_static_min_hours)
    notifier = Notifier(cfg.mattermost_base_url, cfg.mattermost_channel_id,
                        cfg.mattermost_bot, cfg.secrets_mattermost)
    # Dashboard is secondary to alerting: if camera-watcher Postgres is down
    # (or unconfigured), degrade to alert-only instead of halting the pipeline.
    dashboard: DashboardWriter | None = None
    if cfg.dashboard_db_url:
        try:
            dashboard = DashboardWriter(cfg.dashboard_db_url, cfg.dashboard_local_data_dir,
                                        cfg.dashboard_base_static_url)
        except Exception as e:
            log.error("dashboard unavailable, continuing alert-only: %s", e)
            dashboard = None
    subimage_dir = Path(cfg.subimage_dir)
    subimage_dir.mkdir(parents=True, exist_ok=True)

    try:
        images = client.list_images(take_count=cfg.poll_take_count)
        new_captures = [im for im in images if not memory.has_seen(im["imageGuid"])]
        log.info("poll: %d images, %d new", len(images), len(new_captures))

        total = {"alerts": 0, "flags": 0, "suppressed": 0, "detections": 0, "captures": 0}
        for capture in new_captures:
            s = process_capture(capture, client, detector, describer, memory,
                                notifier, dashboard, subimage_dir)
            total["captures"] += 1
            for k in ("alerts", "flags", "suppressed", "detections"):
                total[k] += s[k]
        return total
    finally:
        memory.close()
        if dashboard is not None:
            dashboard.close()


def main() -> None:
    _setup_logging()
    cfg = load_config()

    # --backfill: seed the memory baseline, then exit (run before going live).
    if "--backfill" in sys.argv:
        client = StealthcamClient(secrets_path=cfg.secrets_stealthcam)
        client.login()
        detector = Detector(cfg.detector_weights, cfg.detector_conf_threshold, cfg.detector_device)
        memory = Memory(cfg.pipeline_db, cfg.memory_iou_threshold,
                        cfg.memory_static_n_captures, cfg.memory_static_min_hours)
        subimage_dir = Path(cfg.subimage_dir)
        subimage_dir.mkdir(parents=True, exist_ok=True)
        n = backfill(cfg, client, detector, memory, subimage_dir)
        memory.close()
        log.info("backfill complete: %d captures", n)
        return

    backoff = cfg.poll_interval_seconds
    while True:
        try:
            result = run_once(cfg)
            if result["captures"] == 0:
                # empty poll — back off
                backoff = min(backoff * cfg.poll_backoff_base, cfg.poll_backoff_max)
            else:
                backoff = cfg.poll_interval_seconds
            log.info("cycle done: %s", result)
        except Exception as e:
            log.error("cycle failed: %s", e)
            backoff = min(backoff * cfg.poll_backoff_base, cfg.poll_backoff_max)
        time.sleep(backoff)


if __name__ == "__main__":
    main()
