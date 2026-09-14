"""Motion-event ingestor adapter: wichita scope entity-memory.

The Taylor Hill Stealthcam pipeline polls a camera API; the Wichita house
camera is a DIFFERENT ingestor (a fixed RTSP camera watched by the `motion`
daemon, which writes `event_observations` to camera-watcher's Postgres). This
adapter is the seam the 2026-09-13 scope refactor anticipated: it maps motion
events -> detections -> the SHARED entity-memory engine under
``scope='wichita'`` — the engine core is location-agnostic and is NOT forked.

Pipeline per poll cycle:
  1. Query Postgres for this scope's scenes' motion events newer than the
     lookback window that (a) have a saved significant frame (the video
     worker's async output — its presence is the "event is ready" signal),
     and (b) already carry a classify-pipeline VLM labeling, so this
     adapter's labeling is written LAST and wins the dashboard's
     highest-id-wins display (see watcher.browser._ml_info).
  2. Skip events already in the scope's seen-set (seen_guids, per scope).
  3. For each new event: load the significant frame, run the YOLO detector
     (day/night confidence bars), fingerprint + describe vehicle crops,
     resolve each detection's region GEOMETRICALLY (a fixed camera has no
     compass bearings — the road is a band across the upper frame, see
     context.resolve_region_geom), and feed memory.decide().
  4. Dispatch per decision: alert -> Mattermost + annotated frame;
     note (traffic/background) -> Labeling row on the EXISTING event row
     (tagged description; NO Mattermost post); flag/suppress -> record only.

Run modes:
  python -m stealthcam.motion_ingest                 # live loop
  python -m stealthcam.motion_ingest --once          # single poll, exit
  python -m stealthcam.motion_ingest --backfill      # baseline seeding, exit
  python -m stealthcam.motion_ingest --backfill --backfill-days 3

Runbook (window-recovery): if the ingest pipeline (classify labeling or this
adapter) is down for longer than the 12h lookback window, events that aged
out were dropped (loudly — see the no_frame_dropped / failed-event ERROR
logs). Recover the dropped span with a widened backfill AFTER service is
restored: `python -m stealthcam.motion_ingest --backfill --backfill-days N`
(N = outage span + margin; baseline mode, no alerts posted).

Deploy ordering (same P0-2 invariant as the Taylor go-live): run --backfill
BEFORE starting the live loop so the wichita scope's presence baseline
(monitored-captures denominator + entity fingerprints) is seeded before the
first live decision. An empty baseline makes every driveway vehicle
"unfamiliar" on first sight (one alert, then self-healing — see backfill());
the live loop logs a loud warning when it starts against an empty baseline.
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psycopg2
import psycopg2.extras

from .annotate import annotate
from .config import load_config
from .dashboard import DashboardWriter
from .describer import Describer, VehicleAttributes
from .detector import Detector
from .memory import Memory, ALERT, FLAG, NOTE, SUPPRESS
from .notify import Notifier
from .context import is_night, resolve_region_geom, ROAD_DEFAULT_EDGE

log = logging.getLogger("stealthcam.motion_ingest")

# Events older than this never enter a poll window (bounds the query and the
# seen-set miss cost). Backfill widens it via --backfill-days.
LOOKBACK_HOURS = 12.0


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )


# -- event source -------------------------------------------------------------


class MotionEvent:
    """One motion event ready for entity-memory processing."""

    __slots__ = ("event_id", "event_name", "capture_time", "scene_name",
                 "video_location", "video_file", "frame_relpath")

    def __init__(self, event_id: int, event_name: str, capture_time: datetime,
                 scene_name: str, video_location: str, frame_relpath: str,
                 video_file: str = ""):
        self.event_id = event_id
        self.event_name = event_name
        # The capture_time column is naive UTC (container-local convention,
        # matching dashboard.py's writers) — attach UTC explicitly so every
        # downstream parse is unambiguous.
        if capture_time.tzinfo is None:
            capture_time = capture_time.replace(tzinfo=timezone.utc)
        self.capture_time = capture_time
        self.scene_name = scene_name
        self.video_location = video_location
        self.video_file = video_file
        self.frame_relpath = frame_relpath

    @property
    def capture_iso(self) -> str:
        return self.capture_time.isoformat()


class MotionEventSource:
    """Poll camera-watcher Postgres for a scope's motion events."""

    def __init__(self, db_url: str, scenes: list[str], lookback_hours: float):
        if not scenes:
            raise ValueError(
                "MotionEventSource requires a non-empty scenes list "
                "(set the scope's event_scenes param)")
        if not db_url:
            raise ValueError(
                "MotionEventSource requires a dashboard DB URL "
                "(WATCHER_CONFIG-derived or STEALTHCAM_DASHBOARD_DB_URL)")
        self.scenes = list(scenes)
        self.lookback_hours = lookback_hours
        self._db_url = db_url
        self._conn = psycopg2.connect(db_url)
        self._conn.autocommit = True

    def close(self) -> None:
        self._conn.close()

    def poll(self) -> list[MotionEvent]:
        """ALL in-window events with significant frame + classify VLM
        labeling, oldest first — NO SQL LIMIT.

        The seen-filter runs in python (Postgres cannot see the SQLite
        seen_guids set), so a SQL LIMIT here would bound re-fetching of
        already-seen rows, not new work: once the in-window population
        exceeds the limit (observed 317–428 events in the 12h window vs a
        200 LIMIT), every cycle re-fetches the same oldest cohort and the
        newest events are never fetched until the seen cohort ages out
        (Percy PR#43 P0-1 starvation). The 12h lookback window is the
        natural bound on the query; the per-cycle NEW-work cap lives in
        process_batch, applied AFTER the seen-filter.

        Requiring an existing `ollama:%` labeling orders this adapter's
        labeling AFTER the classify pipeline's on every event (highest
        ML-labeling id wins the dashboard display), and makes "classify
        pipeline up" an implicit readiness gate for each event.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.lookback_hours)
        sql = """
            SELECT e.id, e.event_name, e.capture_time, e.scene_name,
                   e.video_location, e.video_file,
                   ir.file AS frame_file
            FROM event_observations e
            JOIN intermediate_results ir
              ON ir.event_id = e.id AND ir.step = 'task_save_significant_frame'
            WHERE e.camera = ANY(%(scenes)s)
              AND e.storage_local
              AND e.capture_time >= %(cutoff)s
              AND EXISTS (SELECT 1 FROM labelings l
                          WHERE l.event_id = e.id AND l.decider LIKE 'ollama:%%')
            ORDER BY e.capture_time ASC
        """
        with self._conn.cursor(
                cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, {"scenes": self.scenes, "cutoff": cutoff})
            rows = cur.fetchall()
        return [MotionEvent(r["id"], r["event_name"], r["capture_time"],
                            r["scene_name"], r["video_location"],
                            r["frame_file"], video_file=r["video_file"])
                for r in rows]

    def ready_but_unlabeled(self) -> int:
        """Count windowed events whose significant frame exists but which no
        classify labeling has reached yet (picked up once labeled; a
        persistently growing number means the classify pipeline is down and
        this ingestor is starving — surfaced per cycle, never silent)."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.lookback_hours)
        sql = """
            SELECT count(*) AS n
            FROM event_observations e
            JOIN intermediate_results ir
              ON ir.event_id = e.id AND ir.step = 'task_save_significant_frame'
            WHERE e.camera = ANY(%(scenes)s)
              AND e.storage_local
              AND e.capture_time >= %(cutoff)s
              AND NOT EXISTS (SELECT 1 FROM labelings l
                              WHERE l.event_id = e.id
                              AND l.decider LIKE 'ollama:%%')
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, {"scenes": self.scenes, "cutoff": cutoff})
            return int(cur.fetchone()[0])


def _frame_abs_path(local_data_dir: str, event: MotionEvent) -> Path | None:
    """Absolute path of the event's saved significant-frame JPEG, or None."""
    p = Path(local_data_dir) / event.frame_relpath
    return p if p.is_file() else None


# Percy PR#43 P1-1: the previous ffmpeg fallback here was removed. It built
# `{video_location}/{event_name}.mp4` (the on-disk convention is
# `{event_name}_{HHMM}.mp4`, carried in `e.video_file`), so it could never
# find the video; and `-frames:v 1` extracts the PRE-MOTION empty scene, so
# even a "working" fallback would run the detector on an empty frame, mark
# the event seen, and silently record a visitor mid-clip as a no-detection
# capture — worse than dropping the event. There is no safe cheap
# re-extraction here (approximate `-ss` seeking can still land on the wrong
# frame silently), so a missing significant-frame JPEG now retries within
# the lookback window (video-worker lag) and then drops LOUDLY: the
# per-cycle `no_frame_dropped` count in process_batch logs an ERROR naming
# every event that left the window frameless.


# -- per-event processing -----------------------------------------------------


def process_event(event: MotionEvent, detector: Detector,
                  describer: Describer | None, memory: Memory,
                  notifier: Notifier | None, dashboard: DashboardWriter | None,
                  subimage_dir: Path, *, baseline_mode: bool = False,
                  night_conf_threshold: float = 0.75,
                  day_conf_threshold: float = 0.45,
                  day_start_hour: int = 7, day_end_hour: int = 20,
                  road_band: dict | None = None,
                  local_data_dir: str = "/data/video/watcher") -> dict:
    """Process one motion event's significant frame through the engine.

    Returns a summary dict {detections, alerts, flags, notes, suppressed}.
    baseline_mode seeds memory (fingerprints + decisions) with no describer,
    no dashboard, and no Mattermost — the backfill path.
    """
    road = road_band or {}
    summary = {"event": event.event_name, "detections": 0, "alerts": 0,
               "flags": 0, "notes": 0, "suppressed": 0}

    frame = _frame_abs_path(local_data_dir, event)
    if frame is None:
        # Not marked seen: the frame may appear later (video worker lag) and
        # the event leaves the window (and is dropped — loudly, see
        # process_batch) after the lookback. No ffmpeg fallback: it extracted
        # the pre-motion frame 0 and silently mis-recorded visitors (P1-1).
        # The log names the source clip (e.video_file is the true on-disk
        # filename, `{event_name}_{HHMM}.mp4`) so an operator can check the
        # video worker's output directly.
        clip = (f"{event.video_location}/{event.video_file}"
                if event.video_file else "unknown (no video_file in DB row)")
        log.warning("event %s: no significant frame on disk, skipping "
                    "(will retry within lookback window); source clip: %s",
                    event.event_name, clip)
        summary["no_frame"] = True
        return summary

    from PIL import Image
    img = Image.open(frame)
    w, h = img.size

    night = is_night(event.capture_iso, day_start_hour=day_start_hour,
                     day_end_hour=day_end_hour)
    conf_bar = night_conf_threshold if night else day_conf_threshold
    detections = detector.detect(img, conf_threshold=conf_bar)

    # Per-detection index maps, filled in the loop below.
    descriptions: dict[int, str] = {}
    attributes: dict[int, VehicleAttributes] = {}
    fingerprints: dict[int, bytes] = {}

    # Denominator regions EVIDENCED by this frame (issue #44, Percy PR#43
    # P2-4): each surviving detection contributes ITS region's denominator
    # row, so a frame showing both road traffic AND a driveway vehicle feeds
    # BOTH denominators — the old single-`monitored_region` attribution sent
    # the whole frame to 'road', losing genuine driveway denominator rows and
    # letting road traffic volume silently retune the driveway expectedness
    # threshold. Near-band (road_edge) counts toward the road denominator
    # (P2-3: near-band captures are pass-by evidence, not property presence —
    # a pass-by miscounted as a driveway capture manufactures unfamiliar
    # alerts). A frame with NO surviving detection still counts as a property
    # monitored capture (#43: every processed frame is monitored) — that is
    # the one case the property region is the default rather than evidenced.
    monitored_regions: set[str] = set()
    alert_dets = []
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = [float(v) for v in det.bbox]
        bbox_frac = (x1 / w, y1 / h, x2 / w, y2 / h)
        region = resolve_region_geom(
            bbox_frac,
            road_y_max=road.get("y_max", 0.42),
            road_x_min=road.get("x_min", 0.30),
            road_x_max=road.get("x_max", 0.52),
            road_edge=road.get("edge", ROAD_DEFAULT_EDGE))
        if region is None:
            log.warning("event %s det %d: degenerate bbox %s skipped",
                        event.event_name, i, det.bbox)
            continue
        if region in ("road", "road_edge"):
            monitored_regions.add("road")
        else:
            monitored_regions.add("driveway")

        if det.category == "vehicle":
            try:
                fp = memory.fingerprinter.fingerprint(
                    img.crop((int(x1), int(y1), int(x2), int(y2))))
                if fp is not None:
                    fingerprints[i] = fp
            except Exception as e:
                log.warning("fingerprint %s det %d failed: %s",
                            event.event_name, i, e)
            if not baseline_mode and describer is not None:
                try:
                    attributes[i] = describer.describe_vehicle(
                        img.crop((int(x1), int(y1), int(x2), int(y2))))
                    descriptions[i] = attributes[i].as_text()
                except Exception as e:
                    log.warning("describe_vehicle %s det %d failed: %s",
                                event.event_name, i, e)

        decision = memory.decide(
            event.scene_name, det.cls, det.bbox, now=event.capture_iso,
            fingerprint=fingerprints.get(i), region=region,
            attributes=attributes.get(i),
            attributes_text=descriptions.get(i, ""),
            guid=event.event_name,
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
            event.event_name, 0, event.scene_name, det.cls, det.confidence,
            det.bbox, descriptions.get(i, ""), decision,
        )

        # Dashboard: a labeling row on the EXISTING event row (one card per
        # motion event; description carries the [tag]). Suppressed background
        # stays off the dashboard (GH#8 parity with the taylor worker). FLAG
        # gets a row too: a near-band (road_edge) flag exists precisely so a
        # human reviews it — a flag visible only in the decisions table is a
        # silent path (FLAG never posts to Mattermost, by policy).
        if (not baseline_mode and dashboard is not None
                and decision.decision in (NOTE, ALERT, FLAG)):
            text = decision.reason
            if decision.attributes_text:
                text = f"{text} — {decision.attributes_text}"
            if decision.tag:
                text = f"[{decision.tag}] {text}"
            if decision.decision == ALERT:
                text = f"ALERT: {text}"
            try:
                dashboard.write_event_labeling(
                    event.event_id, det.category, det.confidence, text)
            except Exception as e:
                log.error("dashboard labeling %s det %d failed: %s",
                          event.event_name, i, e)

    # No surviving detection -> the frame still counts as a property
    # monitored capture (#43: every processed frame is monitored).
    if not monitored_regions:
        monitored_regions.add("driveway")
    for region in sorted(monitored_regions):
        memory.record_monitored(event.scene_name, region,
                                event.event_name, event.capture_iso,
                                is_day=not night)

    if alert_dets and not baseline_mode:
        alert_detections = [detections[i] for i, _, _ in alert_dets]
        alert_desc = {idx: descriptions[i]
                      for idx, (i, _, _) in enumerate(alert_dets)
                      if i in descriptions}
        annotated = subimage_dir / f"{event.event_name}_alert.jpg"
        annotate(img, alert_detections, alert_desc, dest=annotated)
        for i, det, decision in alert_dets:
            label = ("Human" if det.category == "person"
                     else det.category.capitalize())
            text = (f"wichita: {label} — {decision.reason}, "
                    f"{det.confidence:.2f}, \"{descriptions.get(i, '')}\", "
                    f"{event.capture_iso}")
            if notifier is not None:
                try:
                    notifier.post_alert(text, annotated)
                except Exception as e:
                    log.error("notify failed for %s: %s", event.event_name, e)

    memory.mark_seen(event.event_name)
    return summary


# -- config helpers -----------------------------------------------------------


def _road_band(cfg) -> dict:
    scope = cfg.memory_scope
    return {
        "y_max": float(cfg.scope_setting(scope, "road_band_y_max", 0.42)),
        "x_min": float(cfg.scope_setting(scope, "road_band_x_min", 0.30)),
        "x_max": float(cfg.scope_setting(scope, "road_band_x_max", 0.52)),
        "edge": float(cfg.scope_setting(scope, "road_band_edge",
                                        ROAD_DEFAULT_EDGE)),
    }


def _make_dashboard(cfg) -> DashboardWriter | None:
    if not cfg.dashboard_db_url:
        return None
    try:
        return DashboardWriter(cfg.dashboard_db_url,
                               cfg.dashboard_local_data_dir,
                               cfg.dashboard_base_static_url)
    except Exception as e:
        log.error("dashboard unavailable, continuing alert-only: %s", e)
        return None


def _make_memory_ingest(cfg) -> Memory:
    """Memory bound to this ingestor's scope (cfg.memory_scope)."""
    from .reid import OpenCVFingerprinter

    scope = cfg.memory_scope
    fingerprinter = OpenCVFingerprinter(
        model=cfg.scope_setting(scope, "reid_model", "cv2-silhouette-v2"),
        similarity_threshold=cfg.scope_setting(
            scope, "reid_similarity_threshold", 0.80),
    )
    return Memory(cfg.pipeline_db, cfg.memory_iou_threshold,
                  cfg.memory_static_n_captures, cfg.memory_static_min_hours,
                  fingerprinter=fingerprinter,
                  entity_expected_presence_ratio=cfg.scope_setting(
                      scope, "entity_expected_presence_ratio", 0.5),
                  entity_baseline_window_days=cfg.scope_setting(
                      scope, "entity_baseline_window_days", 7.0),
                  entity_expected_absence_days=cfg.scope_setting(
                      scope, "entity_expected_absence_days", 2.0),
                  scope=scope)


# -- loops --------------------------------------------------------------------


def process_batch(cfg, source: MotionEventSource, memory: Memory,
                  detector: Detector, describer: Describer | None,
                  notifier: Notifier | None, dashboard: DashboardWriter | None,
                  subimage_dir: Path, *, baseline_mode: bool = False,
                  new_limit: int | None = None) -> dict:
    """Process unseen in-window events; returns batch totals.

    The source returns ALL in-window events (no SQL LIMIT — Percy P0-1);
    the seen-filter runs here in python, and ``new_limit`` caps how many
    NEW (unseen) events are processed this call. The cap therefore bounds
    new work (the VLM drain), never re-fetching of seen rows.
    """
    events = source.poll()
    new = [e for e in events if not memory.has_seen(e.event_name)]
    if new_limit is not None:
        # Oldest-first keeps ordering stable; the remainder stays unseen and
        # is picked up next cycle (or by the backfill drain loop).
        new = new[:new_limit]
    totals = {"seen_in_window": len(events), "new": len(new), "alerts": 0,
              "notes": 0, "flags": 0, "detections": 0, "processed": 0,
              "no_frame_dropped": 0}
    road = _road_band(cfg)
    night_conf = float(cfg.scope_setting(
        cfg.memory_scope, "detector_night_conf_threshold", 0.75))
    day_conf = float(cfg.scope_setting(
        cfg.memory_scope, "detector_conf_threshold", 0.45))
    day_start = int(cfg.scope_setting(cfg.memory_scope, "day_start_hour", 7))
    day_end = int(cfg.scope_setting(cfg.memory_scope, "day_end_hour", 20))

    for event in new:
        try:
            s = process_event(event, detector, describer, memory, notifier,
                              dashboard, subimage_dir,
                              baseline_mode=baseline_mode,
                              night_conf_threshold=night_conf,
                              day_conf_threshold=day_conf,
                              day_start_hour=day_start,
                              day_end_hour=day_end,
                              road_band=road,
                              local_data_dir=cfg.dashboard_local_data_dir)
        except Exception as e:
            # One bad event must not wedge the batch; it stays unseen and is
            # retried until it ages out of the lookback window.
            log.error("event %s failed: %s", event.event_name, e)
            continue
        if s.get("no_frame"):
            # Frameless: stays unseen (retry within window), counted loudly,
            # and NOT counted as processed (the backfill drain relies on
            # processed==0 to mean "nothing more to do").
            totals["no_frame_dropped"] += 1
            log.info("event %s: no frame, deferred", event.event_name)
            continue
        totals["processed"] += 1
        totals["alerts"] += s["alerts"]
        totals["notes"] += s["notes"]
        totals["flags"] += s["flags"]
        totals["detections"] += s["detections"]
        log.info("event %s: %s", event.event_name, s)
    return totals


def run_once(cfg) -> dict:
    """One live poll cycle (new events -> decisions -> alerts/notes)."""
    source = MotionEventSource(cfg.dashboard_db_url,
                               scenes=cfg.scope_setting(
                                   cfg.memory_scope, "event_scenes", []),
                               lookback_hours=LOOKBACK_HOURS)
    memory = _make_memory_ingest(cfg)
    detector = Detector(cfg.detector_weights,
                        float(cfg.scope_setting(
                            cfg.memory_scope, "detector_conf_threshold", 0.45)),
                        cfg.detector_device)
    describer = Describer(
        cfg.scope_setting(cfg.memory_scope, "describer_model",
                          "llama3.2-vision:latest"),
        cfg.describer_ollama_host, cfg.describer_timeout)
    # Degradation policy (deliberate, symmetric with _make_dashboard): a
    # broken notifier degrades to alert-logged-not-posted, never kills the
    # cycle — every suppressed alert still surfaces loudly in the logs.
    notifier = None
    try:
        notifier = Notifier(cfg.mattermost_base_url, cfg.mattermost_channel_id,
                            cfg.mattermost_bot, cfg.secrets_mattermost)
    except Exception as e:
        log.error("notifier unavailable, continuing alert-logged-only: %s", e)
    dashboard = _make_dashboard(cfg)
    subimage_dir = Path(cfg.subimage_dir)
    subimage_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = process_batch(cfg, source, memory, detector, describer,
                               notifier, dashboard, subimage_dir,
                               baseline_mode=False,
                               new_limit=int(cfg.event_backlog_max))
        unlabeled = source.ready_but_unlabeled()
        if unlabeled:
            log.warning("%d windowed events await classify labeling "
                        "(ingestor waiting on them)", unlabeled)
        if result.get("no_frame_dropped"):
            # Loud drop (Percy P1-1): frameless events that exhausted their
            # lookback window must never disappear silently.
            log.error("%d in-window event(s) had no significant frame on "
                      "disk and were left unseen; they will drop out of the "
                      "%dh lookback window unprocessed — investigate the "
                      "video worker / thumb GC",
                      result["no_frame_dropped"], int(LOOKBACK_HOURS))
        return result
    finally:
        memory.close()
        if dashboard is not None:
            dashboard.close()
        source.close()


def backfill(cfg, backfill_days: float = 2.0) -> int:
    """Seed the wichita presence baseline (baseline mode, no alerts/notes).

    MUST run before the live loop (P0-2): without a seeded baseline the first
    live driveway sighting of the resident vehicle is "unfamiliar" -> alert.
    Drains the window: loops process_batch until it returns processed == 0,
    so the full ``--backfill-days`` window is seeded, not just the first
    event_backlog_max slice (Percy P1-2: 48h held ~2,700 eligible events vs
    a 200 LIMIT — a 7.5% baseline seeded from the oldest day). An
    empty/partial baseline is self-healing rather than alert-storm-prone
    (first sighting ratio 0.0 -> one unfamiliar ALERT, then converges), so
    backfill-before-live is ENFORCED HERE BY WARNING + draining, not by a
    hard refusal (which would wedge a genuinely new site with no history).
    Returns the number of events processed.
    """
    source = MotionEventSource(cfg.dashboard_db_url,
                               scenes=cfg.scope_setting(
                                   cfg.memory_scope, "event_scenes", []),
                               lookback_hours=backfill_days * 24.0)
    memory = _make_memory_ingest(cfg)
    detector = Detector(cfg.detector_weights,
                        float(cfg.scope_setting(
                            cfg.memory_scope, "detector_conf_threshold", 0.45)),
                        cfg.detector_device)
    subimage_dir = Path(cfg.subimage_dir)
    subimage_dir.mkdir(parents=True, exist_ok=True)
    return _backfill_drain(cfg, source, memory, detector, subimage_dir,
                           backlog_max=int(cfg.event_backlog_max))


def _backfill_drain(cfg, source: MotionEventSource, memory: Memory,
                    detector: Detector, subimage_dir: Path,
                    *, backlog_max: int) -> int:
    """Drain the source's window: loop process_batch until a pass processes
    nothing. Testable seam — the source/memory/detector are injected."""
    total = 0
    try:
        # Drain: each pass consumes up to backlog_max NEW events (the cap
        # applies to unseen work, never to re-fetching seen rows); loop
        # until a pass processes nothing. The guard trips on a pathological
        # cycle where events persistently fail AND stay unseen (the only
        # way processed can return to 0 without the window draining).
        for pass_num in range(1, 1000):
            result = process_batch(cfg, source, memory, detector, None, None,
                                   None, subimage_dir, baseline_mode=True,
                                   new_limit=backlog_max)
            n = result.get("processed", 0)
            total += n
            log.info("backfill pass %d: %s", pass_num, result)
            if n == 0:
                remaining = result.get("new", 0)
                if remaining:
                    log.error(
                        "backfill stopped with %d unseen event(s) that "
                        "failed processing (repeatedly erroring) — these "
                        "are NOT seeded; investigate before going live",
                        remaining)
                break
        else:
            log.error("backfill hit the 999-pass drain guard — aborting "
                      "(window not fully drained)")
    finally:
        memory.close()
        source.close()
    return total


def _parse_args(argv: list[str]):
    """argparse (Percy P3): `--backfill-days` as the last argv token used to
    raise IndexError, and `--backfill-days` without `--backfill` was silently
    ignored. Both are now argparse errors."""
    import argparse
    p = argparse.ArgumentParser(
        prog="python -m stealthcam.motion_ingest",
        description="Wichita motion-event ingestor adapter (entity-memory "
                    "scope adapter over camera-watcher motion events)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true",
                      help="single poll cycle, then exit")
    mode.add_argument("--backfill", action="store_true",
                      help="baseline seeding pass (drains the window), then "
                           "exit; run BEFORE the live loop (P0-2)")
    p.add_argument("--backfill-days", type=float, default=2.0,
                   help="lookback window for --backfill (default: 2.0 days)")
    args = p.parse_args(argv)
    if args.backfill_days <= 0:
        p.error("--backfill-days must be positive")
    return args


def main() -> None:
    _setup_logging()
    args = _parse_args(sys.argv[1:])
    cfg = load_config()
    scenes = cfg.scope_setting(cfg.memory_scope, "event_scenes", [])
    if not scenes:
        log.error("event_scenes is empty for scope %r — nothing to ingest. "
                  "Set the scope block via STEALTHCAM_SCOPES or config.yaml.",
                  cfg.memory_scope)
        sys.exit(2)

    if args.backfill:
        n = backfill(cfg, backfill_days=args.backfill_days)
        log.info("backfill complete: %d events", n)
        return

    if args.once:
        result = run_once(cfg)
        log.info("once done: %s", result)
        return

    # Live loop: bounded poll, exponential backoff on failure, loud baseline
    # warning on an empty scope (deploy-ordering guard, P0-2).
    probe = Memory(cfg.pipeline_db, scope=cfg.memory_scope)
    empty = probe.count_entities() == 0 and probe.count_monitored() == 0
    probe.close()
    if empty:
        log.warning(
            "live loop: wichita baseline is EMPTY (0 entities, 0 monitored) —"
            " every driveway vehicle will alert as unfamiliar on first live "
            "sighting. Run `python -m stealthcam.motion_ingest --backfill` "
            "BEFORE this live loop (P0-2 deploy ordering).")

    backoff = float(cfg.event_poll_seconds)
    while True:
        try:
            result = run_once(cfg)
            log.info("cycle done: %s", result)
            backoff = float(cfg.event_poll_seconds)
        except Exception as e:
            log.error("cycle failed: %s", e)
            backoff = min(backoff * 2.0, 1800.0)
        time.sleep(backoff)


if __name__ == "__main__":
    main()