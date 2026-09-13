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
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .client import StealthcamClient, StealthcamError
from .config import load_config
from .detector import Detector
from .describer import Describer
from .memory import Memory, ALERT, FLAG, SUPPRESS, NOTE
from .annotate import annotate
from .notify import Notifier
from .dashboard import DashboardWriter
from .context import resolve_region, is_night, parse_region_mapping
from .reid import OpenCVFingerprinter

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


def _sanitize_alert_text(text: str, limit: int = 80) -> str:
    """Bound VLM-derived free text before it reaches the Mattermost body.

    The describer output (make/model/lettering, or a free-text description)
    derives from crop content — any legible text in the camera's view becomes
    channel content. Strip newlines and markdown-ish chars and hard-cap the
    length so image-borne text can't flow unbounded into a privileged channel.
    """
    if not text:
        return ""
    # Collapse newlines/tabs to spaces, then squeeze runs of whitespace.
    text = " ".join(text.split())
    # Strip common markdown/quote characters that would render oddly.
    for ch in ("*", "_", "`", "|", ">", "#"):
        text = text.replace(ch, "")
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text.strip()


def process_capture(capture: dict, client: StealthcamClient, detector: Detector,
                    describer: Describer, memory: Memory, notifier: Notifier,
                    dashboard: DashboardWriter | None, subimage_dir: Path,
                    baseline_mode: bool = False,
                    deferred_alert_threshold: int = 3,
                    night_conf_threshold: float = 0.75,
                    day_start_hour: int = 7,
                    day_end_hour: int = 20,
                    region_mapping: dict[str, str] | None = None) -> dict:
    """Process one capture (6 sub-images). Returns a summary dict.

    In `baseline_mode`, we detect + record memory observations but do NOT
    describe, alert, or write dashboard rows — this seeds the persistence
    baseline (so the parked tractor/car are already "static" before the first
    live alert).

    `deferred_alert_threshold` is the number of consecutive polls a capture
    must be deferred (all sub-images failed) before a Mattermost alert is
    posted. The alert fires at most once per guid (tracked in the memory DB).

    Entity-memory wiring: at night the detector confidence bar is raised to
    `night_conf_threshold` (IR noise suppression); each vehicle crop gets a
    re-id fingerprint (identity key) + VLM make/model/lettering (attributes);
    each sub-image is classified into a spatial region (driveway/road) so the
    memory decision can apply the location + expected-presence policy.
    """
    guid = capture["imageGuid"]
    directions = capture.get("imageDirections") or [f"dir{i}" for i in range(1, 7)]
    created_dt = capture.get("createdDateTime", "")

    summary = {"guid": guid, "alerts": 0, "flags": 0, "suppressed": 0,
               "detections": 0, "notes": 0}
    night = is_night(created_dt, day_start_hour=day_start_hour,
                     day_end_hour=day_end_hour)
    conf_bar = night_conf_threshold if night else detector.conf_threshold

    downloaded_any = False
    for n in range(1, 7):
        direction = directions[n - 1] if n - 1 < len(directions) else f"dir{n}"
        local = subimage_dir / f"{guid}_{n}.JPG"
        try:
            client.download_subimage(guid, n, local)
            downloaded_any = True
        except StealthcamError as e:
            log.error("download %s_%d failed: %s", guid, n, e)
            continue

        from PIL import Image
        img = Image.open(local)
        # Night confidence gate: raise the bar at night to kill IR noise.
        detections = detector.detect(img, conf_threshold=conf_bar)

        # Record this sub-image as "monitored" so the presence baseline has a
        # correct denominator even when nothing is detected here.
        region_of_frame = resolve_region(direction, mapping=region_mapping)
        if region_of_frame in ("driveway", "road"):
            memory.record_monitored(direction, region_of_frame, f"{guid}_{n}",
                                    created_dt, is_day=not night)

        # describe each detection (crop + VLM) — skipped in baseline mode;
        # fingerprinting runs in BOTH modes so backfill seeds entity identity
        # (P0-2: identity needs no VLM — describer stays None in baseline).
        descriptions: dict[int, str] = {}
        attributes: dict[int, object] = {}
        fingerprints: dict[int, bytes] = {}
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            crop = img.crop((x1, y1, x2, y2))
            if det.category == "vehicle":
                # Identity key: silhouette/geometry fingerprint (robust
                # where lettering washes out under IR). Always computed so the
                # entity + presence baseline is seeded during backfill too.
                try:
                    fingerprints[i] = memory.fingerprinter.fingerprint(crop)
                except Exception as e:
                    log.warning("fingerprint %s_%d det %d failed: %s",
                                guid, n, i, e)
                if not baseline_mode:
                    # Attributes (make/model/lettering) — best-effort, with
                    # uncertainty; never raise into the decision path.
                    try:
                        attributes[i] = describer.describe_vehicle(crop)
                        descriptions[i] = attributes[i].as_text()
                    except Exception as e:
                        log.warning("describe_vehicle %s_%d det %d failed: %s",
                                    guid, n, i, e)
            elif not baseline_mode:
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
            region = resolve_region(direction, det.bbox,
                                    frame=(img.width, img.height),
                                    mapping=region_mapping)
            decision = memory.decide(
                direction, det.cls, det.bbox, now=now,
                fingerprint=fingerprints.get(i),
                region=region,
                attributes=attributes.get(i),
                attributes_text=descriptions.get(i, ""),
                guid=guid,
            )
            summary["detections"] += 1
            if decision.decision == ALERT:
                summary["alerts"] += 1
                alert_dets.append((i, det, decision))
            elif decision.decision == FLAG:
                summary["flags"] += 1
            elif decision.decision == NOTE:
                summary["notes"] += 1
            else:
                summary["suppressed"] += 1

            memory.record_decision(
                guid, n, direction, det.cls, det.confidence, det.bbox,
                descriptions.get(i, ""), decision,
            )

            # write dashboard row for every NON-suppressed detection
            # (person/vehicle/animal). Static background (the parked tractor)
            # is suppressed by memory and must NOT clutter the dashboard —
            # it's background, not an event (GH#8).
            if (not baseline_mode and dashboard is not None
                    and det.category in ("person", "vehicle", "animal")
                    and decision.decision != SUPPRESS):
                try:
                    if decision.decision == NOTE:
                        dashboard.write_note(
                            guid, n, direction, det.category, det.confidence,
                            decision.reason, created_dt, det.bbox, i,
                            tag=decision.tag,
                            make_model_lettering=decision.attributes_text,
                        )
                    else:
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
                    f"{det.confidence:.2f}, \"{_sanitize_alert_text(descriptions.get(i, ''))}\", "
                    f"{created_dt}"
                )
                try:
                    notifier.post_alert(text, annotated)
                except Exception as e:
                    log.error("notify failed: %s", e)

    if downloaded_any:
        memory.mark_seen(guid)
    else:
        # Every sub-image failed to download (e.g. all 403'd on a stale cookie).
        # Do NOT mark seen so a later poll (or a cookie refresh) can retry the
        # capture — otherwise the blind spot is never revisited.
        log.warning(
            "capture %s deferred: no sub-images downloaded, will retry next poll",
            guid,
        )
        # Track the deferral and, once it crosses the threshold, surface a
        # one-time Mattermost alert so a recurring blind spot is not log-only.
        count = memory.record_deferral(guid)
        if (count >= deferred_alert_threshold
                and not memory.is_deferred_alerted(guid)
                and notifier is not None):
            try:
                notifier.post_alert(
                    f"stealthcam: capture {guid} deferred {count} consecutive "
                    f"polls (all sub-images failing) — possible blind spot"
                )
                memory.mark_deferred_alerted(guid)
            except Exception as e:
                log.error("deferred-alert notify failed: %s", e)
    return summary


def backfill(cfg, client: StealthcamClient, detector: Detector, memory: Memory,
             subimage_dir: Path, max_captures: int = 200,
             region_mapping: dict[str, str] | None = None,
             force_seen: bool = False) -> int:
    """Seed the memory baseline from recent captures (no alerts, no dashboard).

    Pages back through images-list and processes captures in baseline mode so
    persistent objects (parked tractor/car) are already "static" before the
    first live alert. Returns the number of captures backfilled.

    Captures are processed in CHRONOLOGICAL order (oldest first). The
    images-list endpoint returns newest-first, and static detection measures
    persistence as (capture_count >= N) AND (span >= M hours) keyed on each
    capture's own `createdDateTime`. Processing newest-first means the first
    sighting of a persistent object is recorded with a *recent* timestamp, so
    the span never reaches M hours and the object is re-alerted as "novel" on
    every capture (the parked-tractor bug). Sorting oldest-first makes the
    first sighting carry the earliest timestamp, so the span grows correctly
    and the object converges to static.

    ``region_mapping`` must be the SAME direction->region mapping the live
    loop uses: with compass-bearing directions and no mapping, every frame
    resolves to an unclassified region, so neither record_monitored (denominator,
    gated on driveway/road) nor entity resolution (driveway-only) runs and the
    backfill seeds entities=0 — the P0-2 gate fails on the live DB.

    ``force_seen`` reprocesses captures already in seen_guids. The pre-#36
    worker marked every capture seen for weeks, so on a live database the
    seen-set covers the entire backfill window and the default skip would
    process 0 captures (entities=0). Baseline seeding must run over the
    already-seen history; set force_seen=True for the pre-go-live backfill.
    """
    log.info("backfill: seeding baseline (up to %d captures%s)",
             max_captures, ", force_seen" if force_seen else "")
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
        if memory.has_seen(capture["imageGuid"]) and not force_seen:
            continue
        try:
            process_capture(capture, client, detector, None, memory, None, None,
                            subimage_dir, baseline_mode=True,
                            day_start_hour=getattr(cfg, "day_start_hour", 7),
                            day_end_hour=getattr(cfg, "day_end_hour", 20),
                            region_mapping=region_mapping)
            processed += 1
        except Exception as e:
            log.error("backfill capture %s failed: %s", capture["imageGuid"], e)
    log.info("backfill done: %d captures", processed)
    return processed


def requeue_deferred(cfg, client: StealthcamClient, detector: Detector,
                     describer: Describer, memory: Memory, notifier: Notifier,
                     dashboard: DashboardWriter | None, subimage_dir: Path,
                     older_than_hours: float = 24.0,
                     max_captures: int = 200) -> int:
    """Re-process captures that were never marked seen and are older than the
    retry window, closing the pre-#31 historical gap permanently.

    A deferred capture ages out of the live retry window once ~20 newer
    captures exist (`poll_take_count=20` filters by `memory.has_seen`), so a
    capture that failed before the cookie-refresh fix would never be revisited.
    This pages `list_images` back in time and re-processes (in NORMAL mode, so
    real detections/alerts surface) every capture that is still unseen and
    older than `older_than_hours`. Returns the number of captures re-processed.
    """
    log.info("requeue-deferred: paging back for unseen captures older than %.1fh",
             older_than_hours)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=older_than_hours)

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

    processed = 0
    for capture in captures:
        guid = capture["imageGuid"]
        if memory.has_seen(guid):
            continue
        created = capture.get("createdDateTime", "")
        try:
            created_dt = datetime.fromisoformat(created)
        except (ValueError, TypeError):
            log.warning("requeue-deferred: unparseable createdDateTime %r for %s",
                        created, guid)
            continue
        if created_dt >= cutoff:
            # Still inside the live retry window — the normal poll will handle it.
            continue
        try:
            process_capture(capture, client, detector, describer, memory,
                            notifier, dashboard, subimage_dir,
                            night_conf_threshold=getattr(cfg, "detector_night_conf_threshold", 0.75),
                            day_start_hour=getattr(cfg, "day_start_hour", 7),
                            day_end_hour=getattr(cfg, "day_end_hour", 20),
                            region_mapping=parse_region_mapping(
                                getattr(cfg, "direction_regions", "")))
            processed += 1
        except Exception as e:
            log.error("requeue-deferred capture %s failed: %s", guid, e)
    log.info("requeue-deferred done: %d captures", processed)
    return processed


def _make_memory(cfg) -> Memory:
    """Build a Memory wired with the configured entity/re-id parameters."""
    fingerprinter = OpenCVFingerprinter(
        model=cfg.reid_model,
        similarity_threshold=cfg.reid_similarity_threshold,
    )
    return Memory(cfg.pipeline_db, cfg.memory_iou_threshold,
                  cfg.memory_static_n_captures, cfg.memory_static_min_hours,
                  fingerprinter=fingerprinter,
                  entity_expected_presence_ratio=cfg.entity_expected_presence_ratio,
                  entity_baseline_window_days=cfg.entity_baseline_window_days,
                  entity_expected_absence_days=cfg.entity_expected_absence_days)


def run_once(cfg) -> dict:
    client = StealthcamClient(secrets_path=cfg.secrets_stealthcam)
    client.login()

    detector = Detector(cfg.detector_weights, cfg.detector_conf_threshold, cfg.detector_device)
    describer = Describer(cfg.describer_model, cfg.describer_ollama_host, cfg.describer_timeout)
    memory = _make_memory(cfg)
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

        # Capture device telemetry (battery/disk/signal) each poll. This is
        # independent of image processing: it records camera health even when
        # there are no new images (the "no recent images" gap is a camera
        # power/PIR question, not a pipeline bug — telemetry is how we see it).
        if dashboard is not None:
            try:
                status = client.get_device_status()
                dashboard.write_telemetry(status)
                log.info(
                    "telemetry: battery=%s%% sd_free=%s%% signal=%s errors=%s",
                    status.get("batteryLevel"), status.get("sdCardFreeSpace"),
                    status.get("signalStrength"), status.get("errors"),
                )
            except Exception as e:
                log.error("telemetry capture failed: %s", e)

        total = {"alerts": 0, "flags": 0, "suppressed": 0, "detections": 0, "captures": 0}
        region_mapping = parse_region_mapping(cfg.direction_regions)
        for capture in new_captures:
            try:
                s = process_capture(capture, client, detector, describer, memory,
                                    notifier, dashboard, subimage_dir,
                                    deferred_alert_threshold=cfg.deferred_alert_threshold,
                                    night_conf_threshold=cfg.detector_night_conf_threshold,
                                    day_start_hour=cfg.day_start_hour,
                                    day_end_hour=cfg.day_end_hour,
                                    region_mapping=region_mapping)
            except Exception as e:
                # A deterministically-crashing capture must never wedge the
                # newest-20 window (it would stay unseen and abort every poll
                # cycle). Mark it seen so the window advances; the trade is
                # explicit — that one capture's alerts are lost rather than
                # ALL alerts (total pipeline blindness). (P0-1)
                log.error("capture %s failed, marking seen to advance window: %s",
                          capture["imageGuid"], e)
                memory.mark_seen(capture["imageGuid"])
                continue
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
        if not cfg.direction_regions.strip():
            log.warning(
                "backfill: direction_regions is EMPTY — every frame resolves "
                "to an unclassified region, so no entities/presence will be "
                "seeded (P0-2 gate: entities must be > 0). Set "
                "STEALTHCAM_DIRECTION_REGIONS before going live."
            )
        client = StealthcamClient(secrets_path=cfg.secrets_stealthcam)
        client.login()
        detector = Detector(cfg.detector_weights, cfg.detector_conf_threshold, cfg.detector_device)
        memory = _make_memory(cfg)
        subimage_dir = Path(cfg.subimage_dir)
        subimage_dir.mkdir(parents=True, exist_ok=True)
        n = backfill(cfg, client, detector, memory, subimage_dir,
                     region_mapping=parse_region_mapping(cfg.direction_regions),
                     force_seen="--backfill-force-seen" in sys.argv)
        memory.close()
        log.info("backfill complete: %d captures", n)
        return

    # --requeue-deferred: one-shot re-process of unseen captures older than the
    # retry window (closes the pre-#31 historical gap). Runs in NORMAL mode so
    # real detections/alerts surface, then exits.
    if "--requeue-deferred" in sys.argv:
        client = StealthcamClient(secrets_path=cfg.secrets_stealthcam)
        client.login()
        detector = Detector(cfg.detector_weights, cfg.detector_conf_threshold, cfg.detector_device)
        describer = Describer(cfg.describer_model, cfg.describer_ollama_host, cfg.describer_timeout)
        memory = _make_memory(cfg)
        notifier = Notifier(cfg.mattermost_base_url, cfg.mattermost_channel_id,
                            cfg.mattermost_bot, cfg.secrets_mattermost)
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
        n = requeue_deferred(cfg, client, detector, describer, memory, notifier,
                             dashboard, subimage_dir,
                             older_than_hours=cfg.requeue_older_than_hours)
        memory.close()
        if dashboard is not None:
            dashboard.close()
        log.info("requeue-deferred complete: %d captures", n)
        return

    backoff = cfg.poll_interval_seconds
    if not cfg.direction_regions.strip():
        log.warning(
            "live loop: direction_regions is EMPTY — vehicles in unclassified "
            "regions fall back to the novel-vehicle policy (static -> suppress, "
            "else alert) and the presence baseline will not run. Set "
            "STEALTHCAM_DIRECTION_REGIONS to enable the driveway/road policy."
        )
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
