import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import configparser
import pytz
from flask import Blueprint, render_template_string, request, jsonify, send_file, abort
from sqlalchemy import select, desc, func, exists, Text
from sqlalchemy.orm import joinedload
from PIL import Image

from .model import EventObservation, Labeling, IntermediateResult, StealthcamFeedback, StealthcamTelemetry
from .connection import application_config

__all__ = ['browser_bp']

log = logging.getLogger(__name__)

browser_bp = Blueprint('browser', __name__)

# ML deciders: the existing ollama:<model> prefix, plus the stealthcam
# pipeline's yolo11n+vlm decider (see stealthcam-pipeline).
ML_DECIDER_PREFIX = 'ollama:'
ML_DECIDER_EXACT = 'yolo11n+vlm'

# Media file extensions that are still photos, not video. The stealthcam
# pipeline writes sub-image JPEGs (video_file = '<guid>_<n>.JPG'); the
# dashboard must render those as <img>, not a <video> player.
PHOTO_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}


# ── lazy thumbnails ──────────────────────────────────────────────────────────
#
# The /browser list renders each card's <img> at 192x108 but was serving the
# full 1024x576 JPEG (~142KB) and downscaling in-browser. We generate a ~320px
# wide JPEG thumbnail lazily on first request, cache it on disk under a
# deterministic name derived from the source path, and serve it with a long
# Cache-Control (GUID filenames are immutable). The enlarge/click path keeps
# the full-resolution URL.
#
# Thumbnails are regenerable, so purging them is always safe (no data loss).

THUMB_MAX_WIDTH = 320
THUMB_JPEG_QUALITY = 80
THUMB_DIR_NAME = 'thumb'

# Throttle state for the purge check (module-level, not a function attribute).
_last_purge_run = 0.0


def _thumb_config():
    """Read thumbnail knobs from [thumb] in the app config (with defaults).

    The [thumb] section is optional; a missing section or missing key falls
    back to the default (never raises).
    """
    def _get(key, default):
        try:
            val = application_config('thumb', key)
        except (KeyError, configparser.Error):
            return default
        try:
            return int(val) if val else default
        except (TypeError, ValueError):
            return default

    return {
        'max_count': _get('MAX_COUNT', 20000),
        'max_bytes': _get('MAX_BYTES', 0),
        'max_age_days': _get('MAX_AGE_DAYS', 0),
    }


def _thumb_dir() -> Path:
    """Root directory thumbnails are written under (LOCAL_DATA_DIR/thumb)."""
    return Path(application_config('system', 'LOCAL_DATA_DIR')) / THUMB_DIR_NAME


def _thumb_path_for(source_relpath: str) -> Path:
    """Deterministic on-disk path for a source file's thumbnail.

    The source relpath (e.g. 'stealthcam/<guid>_1.JPG' or
    'wichitaDriveway/2025/10/02/....jpg') is mirrored under the thumb root so
    the mapping is idempotent and collision-free across cameras.
    """
    rel = Path(source_relpath)
    # Guard against path traversal: never escape the thumb root.
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError(f"unsafe source relpath: {source_relpath}")
    return _thumb_dir() / rel.with_suffix('.jpg')


def _thumb_url_for(source_relpath: str) -> str:
    """Public URL for a source file's thumbnail (served by the /thumb route)."""
    return f"/watcher/thumb/{source_relpath}"


def _source_abs_path(source_relpath: str) -> Path:
    """Absolute path of the source image under LOCAL_DATA_DIR."""
    return Path(application_config('system', 'LOCAL_DATA_DIR')) / source_relpath


def _generate_thumbnail(source_abs: Path, dest: Path) -> None:
    """Downscale source_abs to a ~320px JPEG at dest (atomic write)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix('.tmp')
    try:
        with Image.open(source_abs) as img:
            img = img.convert('RGB')
            if img.width > THUMB_MAX_WIDTH:
                height = round(img.height * (THUMB_MAX_WIDTH / img.width))
                img = img.resize((THUMB_MAX_WIDTH, height), Image.Resampling.LANCZOS)
            img.save(tmp, 'JPEG', quality=THUMB_JPEG_QUALITY, optimize=True)
        os.replace(tmp, dest)
    except Exception:
        # Never leave a partial thumbnail behind. unlink(missing_ok=True) only
        # raises on a real I/O error (e.g. permission), which we log and let
        # the original exception propagate.
        try:
            tmp.unlink(missing_ok=True)
        except OSError as cleanup_err:
            log.warning("failed to clean up partial thumbnail %s: %s", tmp, cleanup_err)
        raise


def _purge_thumbnails_if_needed() -> None:
    """Best-effort thumbnail cleanup, bounded to run at most once per minute.

    Enforces the [thumb] knobs (max count / max bytes / max age). Thumbnails
    are regenerable so deletion is always safe. Failures are logged and
    swallowed — a purge error must never break thumbnail serving.
    """
    global _last_purge_run

    cfg = _thumb_config()
    if not (cfg['max_count'] or cfg['max_bytes'] or cfg['max_age_days']):
        return

    # Throttle: at most one purge per minute across the process.
    now = time.time()
    if now - _last_purge_run < 60:
        return
    _last_purge_run = now

    root = _thumb_dir()
    if not root.is_dir():
        return
    try:
        files = [p for p in root.rglob('*.jpg') if p.is_file()]
        removed = 0

        if cfg['max_age_days']:
            cutoff = now - cfg['max_age_days'] * 86400
            for p in files:
                if p.stat().st_mtime < cutoff:
                    p.unlink(missing_ok=True)
                    removed += 1
            files = [p for p in files if p.exists()]

        if cfg['max_bytes']:
            files.sort(key=lambda p: p.stat().st_mtime)
            total = sum(p.stat().st_size for p in files)
            while total > cfg['max_bytes'] and files:
                p = files.pop(0)
                total -= p.stat().st_size
                p.unlink(missing_ok=True)
                removed += 1

        if cfg['max_count']:
            files.sort(key=lambda p: p.stat().st_mtime)
            while len(files) > cfg['max_count']:
                p = files.pop(0)
                p.unlink(missing_ok=True)
                removed += 1

        if removed:
            log.info("thumbnail purge removed %d files", removed)
    except Exception as e:
        log.warning("thumbnail purge failed: %s", e)


def _is_photo(event) -> bool:
    """True when the event's media file is a still image, not a video."""
    name = (event.video_file or '').lower()
    return any(name.endswith(ext) for ext in PHOTO_EXTENSIONS)


def _escape_like(s: str) -> str:
    """Escape LIKE wildcards so user input matches literally."""
    return s.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')


def _description_subq(description: str):
    """Subquery: event ids whose ML labeling description matches (substring)."""
    pattern = '%' + _escape_like(description) + '%'
    return (select(Labeling.event_id)
            .where(_ml_decider_predicate())
            .where(Labeling.description.ilike(pattern, escape='\\')))


def _is_ml_decider(decider):
    return bool(decider) and (decider.startswith(ML_DECIDER_PREFIX) or decider == ML_DECIDER_EXACT)


def _ml_decider_predicate():
    from sqlalchemy import or_
    return or_(Labeling.decider.like(ML_DECIDER_PREFIX + '%'),
               Labeling.decider == ML_DECIDER_EXACT)


def _model_name(decider):
    if decider.startswith(ML_DECIDER_PREFIX):
        return decider.removeprefix(ML_DECIDER_PREFIX)
    return decider

PAGE_SIZE = 24
RECENT_DAYS = 60   # "recent" filter window


CATEGORY_ICON = {
    'person':           '🚶',
    'vehicle':          '🚗',
    'animal':           '🐾',
    'lighting_change':  '💡',
    'wind_vegetation':  '🌿',
    'shadow':           '🌑',
    'insect':           '🪲',
    'unknown':          '❓',
}

LIGHTING_LABEL = {
    'daylight':  '☀️',
    'twilight':  '🌆',
    'night':     '🌙',
    'midnight':  '🌑',
}


def _tz():
    return pytz.timezone(application_config('location', 'TIMEZONE') or 'UTC')


def _fmt_time(dt):
    if dt is None:
        return ''
    try:
        local = dt.astimezone(_tz())
    except Exception:
        local = dt
    # strftime %-I is Linux-only; strip manually for portability
    s = local.strftime('%b %d, %Y  %I:%M %p').lstrip('0')
    return s.replace('  0', '  ').replace(' 0', ' ')


def _ml_info(event):
    # Use the most recently decided ML labeling (highest id)
    ml = [l for l in event.labelings if _is_ml_decider(l.decider)]
    lbl = max(ml, key=lambda l: l.id) if ml else None
    if not lbl:
        return None
    cats = [l for l in lbl.labels if l != 'noise']
    confidence = round(lbl.probabilities[0] * 100) if (lbl.probabilities and lbl.probabilities[0] > 0) else None
    model_name = _model_name(lbl.decider)
    return {
        'category':    cats[0] if cats else 'unknown',
        'interesting': 'noise' not in lbl.labels,
        'confidence':  confidence,
        'model':       model_name,
        'description': lbl.description or '',
        'git_version': (lbl.git_version or '')[:7],
    }


def _human_info(event):
    lbl = next((l for l in event.labelings if not l.probabilities), None)
    if not lbl:
        return None
    return {'labels': lbl.labels, 'decider': lbl.decider}


def _feedback_info(event):
    """Current 👍/👎 feedback for this event, if any (toggle state)."""
    fb = next((f for f in event.feedback), None)
    if not fb:
        return None
    return {'label': fb.label, 'reason': fb.reason or ''}


def _latest_telemetry(db_session):
    """Most recent stealthcam device-status snapshot, or None.

    Feeds the header health readout (battery / disk / signal). Returns a
    plain dict so the template can render it without touching the ORM.
    """
    row = (db_session.execute(
        select(StealthcamTelemetry).order_by(desc(StealthcamTelemetry.captured_at)).limit(1)
    ).scalars().first())
    if row is None:
        return None
    return {
        'battery_pct': row.battery_pct,
        'sd_card_free_pct': row.sd_card_free_pct,
        'signal_strength': row.signal_strength,
        'last_sync_at': _fmt_time(row.last_sync_at) if row.last_sync_at else None,
        'errors': row.errors or [],
    }


def _serialize(event):
    return {
        'id':           event.id,
        'event_name':   event.event_name,
        'capture_time': _fmt_time(event.capture_time),
        'scene_name':   event.scene_name or '',
        'camera':       event.camera or '',
        'lighting':     event.lighting_type or '',
        'video_url':    event.video_url,
        'frame_url':    event.significant_frame_url,
        'thumb_url':    _thumb_url(event),
        'is_photo':     _is_photo(event),
        'ml':           _ml_info(event),
        'human':        _human_info(event),
        'feedback':     _feedback_info(event),
    }


def _thumb_url(event):
    """Thumbnail URL for an event's significant frame, or None if no frame.

    Derived from the frame's on-disk relpath (the IntermediateResult.file),
    which is stable and deterministic. Returns None when the event has no
    significant frame or the relpath is malformed (the card then renders its
    'no frame' placeholder).
    """
    if not event.results:
        return None
    relpath = event.results[-1].file
    if not relpath:
        return None
    rel = Path(relpath)
    if rel.is_absolute() or '..' in rel.parts:
        log.warning("skipping thumbnail for unsafe relpath: %s", relpath)
        return None
    return _thumb_url_for(relpath)


def _recent_cutoff():
    return datetime.now(timezone.utc) - timedelta(days=RECENT_DAYS)


def _base_stmt():
    return (select(EventObservation)
            .options(joinedload(EventObservation.labelings),
                     joinedload(EventObservation.results),
                     joinedload(EventObservation.feedback))
            .order_by(desc(EventObservation.capture_time)))


def _has_frame_subq():
    return select(IntermediateResult.event_id).correlate(EventObservation)


def _fetch(db_session, page, filter_mode, camera=None, description=None):
    stmt = _base_stmt()

    if camera:
        stmt = stmt.where(EventObservation.camera == camera)

    if description:
        stmt = stmt.where(EventObservation.id.in_(_description_subq(description)))

    if filter_mode == 'recent':
        stmt = stmt.where(EventObservation.capture_time >= _recent_cutoff())

    elif filter_mode == 'interesting':
        ids = (select(Labeling.event_id)
               .where(_ml_decider_predicate())
               .where(Labeling.labels.cast(Text).not_like('%noise%')))
        stmt = stmt.where(EventObservation.id.in_(ids))

    elif filter_mode == 'noise':
        ids = (select(Labeling.event_id)
               .where(_ml_decider_predicate())
               .where(Labeling.labels.cast(Text).like('%noise%')))
        stmt = stmt.where(EventObservation.id.in_(ids))

    elif filter_mode == 'unclassified':
        classified = select(Labeling.event_id).where(_ml_decider_predicate())
        has_frame  = select(IntermediateResult.event_id)
        stmt = (stmt
                .where(EventObservation.id.notin_(classified))
                .where(EventObservation.id.in_(has_frame))
                .where(EventObservation.capture_time >= _recent_cutoff()))

    # 'all' — no filter, but cap at recent to avoid drowning in old frameless events
    else:
        stmt = stmt.where(EventObservation.capture_time >= _recent_cutoff())

    offset = (page - 1) * PAGE_SIZE
    rows = db_session.execute(stmt.offset(offset).limit(PAGE_SIZE + 1)).scalars().unique().all()
    has_more = len(rows) > PAGE_SIZE
    return [_serialize(e) for e in rows[:PAGE_SIZE]], has_more


def _counts(db_session):
    cutoff = _recent_cutoff()
    has_frame = select(IntermediateResult.event_id)
    classified_ids = select(Labeling.event_id).where(_ml_decider_predicate())

    recent = db_session.execute(
        select(func.count()).where(EventObservation.capture_time >= cutoff)
    ).scalar()
    interesting = db_session.execute(
        select(func.count(Labeling.event_id.distinct()))
        .where(_ml_decider_predicate())
        .where(Labeling.labels.cast(Text).not_like('%noise%'))
    ).scalar()
    noise = db_session.execute(
        select(func.count(Labeling.event_id.distinct()))
        .where(_ml_decider_predicate())
        .where(Labeling.labels.cast(Text).like('%noise%'))
    ).scalar()
    unclassified = db_session.execute(
        select(func.count()).select_from(EventObservation)
        .where(EventObservation.capture_time >= cutoff)
        .where(EventObservation.id.notin_(classified_ids))
        .where(EventObservation.id.in_(has_frame))
    ).scalar()
    return dict(recent=recent, interesting=interesting, noise=noise, unclassified=unclassified)


def _cameras(db_session):
    """Distinct camera values (alphabetical) for the filter dropdown."""
    rows = db_session.execute(
        select(EventObservation.camera)
        .where(EventObservation.camera.isnot(None))
        .where(EventObservation.camera != '')
        .distinct()
    ).scalars().all()
    return sorted({c for c in rows if c})


# ── routes ────────────────────────────────────────────────────────────────────

@browser_bp.route('/browser')
def browser_page():
    from api import db

    filter_mode = request.args.get('filter', 'recent')
    page = request.args.get('page', 1, type=int)
    camera = request.args.get('camera') or None
    description = (request.args.get('description') or '').strip() or None

    events, has_more = _fetch(db.session, page, filter_mode, camera, description)
    counts = _counts(db.session)
    cameras = _cameras(db.session)
    telemetry = _latest_telemetry(db.session)

    return render_template_string(
        _TEMPLATE,
        events=events,
        filter=filter_mode,
        page=page,
        has_more=has_more,
        counts=counts,
        cameras=cameras,
        camera=camera,
        telemetry=telemetry,
        description=description or '',
        CATEGORY_ICON=CATEGORY_ICON,
        LIGHTING_LABEL=LIGHTING_LABEL,
    )


@browser_bp.route('/events')
def events_json():
    from api import db

    filter_mode = request.args.get('filter', 'recent')
    page        = request.args.get('page', 1, type=int)
    camera      = request.args.get('camera') or None
    description = (request.args.get('description') or '').strip() or None
    since_id    = request.args.get('since_id', type=int)   # newer events only
    ids         = request.args.getlist('id', type=int)      # re-fetch specific events

    if ids:
        # Re-fetch specific events by id (for badge refresh)
        stmt = (_base_stmt()
                .where(EventObservation.id.in_(ids)))
        rows = db.session.execute(stmt).scalars().unique().all()
        return jsonify({'events': [_serialize(e) for e in rows]})

    if since_id is not None:
        # New events only — ignore pagination, just return what arrived since last poll
        stmt = (_base_stmt()
                .where(EventObservation.capture_time >= _recent_cutoff())
                .where(EventObservation.id > since_id))
        if camera:
            stmt = stmt.where(EventObservation.camera == camera)
        if description:
            stmt = stmt.where(EventObservation.id.in_(_description_subq(description)))
        rows = db.session.execute(stmt).scalars().unique().all()
        return jsonify({'events': [_serialize(e) for e in rows]})

    events, has_more = _fetch(db.session, page, filter_mode, camera, description)
    return jsonify({'events': events, 'has_more': has_more, 'page': page})


# ── thumbnail route ──────────────────────────────────────────────────────────

@browser_bp.route('/thumb/<path:relpath>')
def thumbnail(relpath):
    """Serve a lazily-generated thumbnail for a source image.

    The relpath is the source file's path under LOCAL_DATA_DIR (e.g.
    'stealthcam/<guid>_1.JPG'). On first request the thumbnail is generated
    (downscaled to ~320px JPEG) and cached on disk; subsequent requests are
    served from disk with a long Cache-Control (source filenames are
    immutable). Missing source -> 404; unreadable source -> 404 (never a 500
    that would break the card's onerror fallback).
    """
    try:
        source_abs = _source_abs_path(relpath)
        dest = _thumb_path_for(relpath)
    except ValueError:
        abort(404)

    if not source_abs.is_file():
        abort(404)

    if not dest.is_file():
        try:
            _generate_thumbnail(source_abs, dest)
        except Exception as e:
            log.warning("thumbnail generation failed for %s: %s", relpath, e)
            abort(404)

    _purge_thumbnails_if_needed()

    resp = send_file(dest, mimetype='image/jpeg', conditional=True)
    resp.headers['Cache-Control'] = 'public, max-age=2592000, immutable'
    return resp


# ── feedback route ───────────────────────────────────────────────────────────

@browser_bp.route('/feedback', methods=['POST'])
def feedback():
    """Set/clear 👍/👎 feedback on an event (toggle semantics).

    Body: {event_id, label: 'good'|'bad', reason?: str}. One row per event
    (UNIQUE event_id): setting the same label again is an undo (row deleted);
    setting the other label replaces it. Returns the new feedback state.
    """
    from api import db

    data = request.get_json(silent=True) or {}
    event_id = data.get('event_id')
    label = data.get('label')
    reason = (data.get('reason') or '').strip()

    if label not in ('good', 'bad'):
        return jsonify({'error': "label must be 'good' or 'bad'"}), 400

    if event_id is None:
        return jsonify({'error': 'event_id is required'}), 400
    try:
        event_id = int(event_id)
    except (TypeError, ValueError):
        return jsonify({'error': 'event_id must be an integer'}), 400

    event = db.session.get(EventObservation, event_id)
    if event is None:
        return jsonify({'error': 'event not found'}), 404

    existing = db.session.execute(
        select(StealthcamFeedback).where(StealthcamFeedback.event_id == event_id)
    ).scalar_one_or_none()

    if existing is not None and existing.label == label:
        # Same label again = undo (toggle off).
        db.session.delete(existing)
        db.session.commit()
        return jsonify({'feedback': None})

    if existing is not None:
        # Switching label (👍 -> 👎 or vice versa).
        existing.label = label
        existing.reason = reason or None
        existing.created_at = datetime.now()
    else:
        db.session.add(StealthcamFeedback(event_id=event_id, label=label, reason=reason or None))
    db.session.commit()

    return jsonify({'feedback': {'label': label, 'reason': reason}}), 201


# ── compare route ────────────────────────────────────────────────────────────

def _compare_data(db_session, limit=50, camera=None, description=None):
    """Return events that have ≥2 distinct ollama models, with per-model results."""
    from sqlalchemy import func

    # Events with labelings from multiple different ollama models
    multi = (select(Labeling.event_id)
             .where(_ml_decider_predicate())
             .group_by(Labeling.event_id)
             .having(func.count(Labeling.decider.distinct()) >= 1))

    stmt = (_base_stmt()
            .where(EventObservation.id.in_(multi))
            .where(EventObservation.capture_time >= _recent_cutoff()))

    if camera:
        stmt = stmt.where(EventObservation.camera == camera)

    if description:
        stmt = stmt.where(EventObservation.id.in_(_description_subq(description)))

    stmt = stmt.limit(limit)

    events = db_session.execute(stmt).scalars().unique().all()

    # Collect all distinct model names across these events
    all_models = sorted({
        _model_name(l.decider)
        for e in events for l in e.labelings
        if _is_ml_decider(l.decider)
    })

    rows = []
    for ev in events:
        by_model = {}
        for lbl in sorted(ev.labelings, key=lambda l: l.id):
            if not _is_ml_decider(lbl.decider):
                continue
            mname = _model_name(lbl.decider)
            cats = [l for l in lbl.labels if l != 'noise']
            by_model[mname] = {
                'category':    cats[0] if cats else 'unknown',
                'interesting': 'noise' not in lbl.labels,
                'confidence':  round(lbl.probabilities[0] * 100) if (lbl.probabilities and lbl.probabilities[0] > 0) else None,
                'description': lbl.description or '',
                'git_version': (lbl.git_version or '')[:7],
            }
        rows.append({
            'id':           ev.id,
            'capture_time': _fmt_time(ev.capture_time),
            'scene_name':   ev.scene_name or '',
            'frame_url':    ev.significant_frame_url,
            'video_url':    ev.video_url,
            'is_photo':     _is_photo(ev),
            'models':       by_model,
        })

    return rows, all_models


@browser_bp.route('/compare')
def compare_page():
    from api import db
    camera = request.args.get('camera') or None
    description = (request.args.get('description') or '').strip() or None
    rows, models = _compare_data(db.session, camera=camera, description=description)
    cameras = _cameras(db.session)
    return render_template_string(_COMPARE_TEMPLATE, rows=rows, models=models,
                                  camera=camera, cameras=cameras,
                                  description=description or '',
                                  CATEGORY_ICON=CATEGORY_ICON)


# ── template ──────────────────────────────────────────────────────────────────

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Camera Events</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    .card-thumb { width:192px; min-height:108px; flex-shrink:0; background:#e5e7eb; }
    .card-thumb img { width:192px; height:108px; object-fit:cover; display:block; }
    @media(max-width:600px){
      .card-thumb{ width:100%; }
      .card-thumb img{ width:100%; height:auto; }
      .event-card{ flex-direction:column; }
    }
  </style>
</head>
<body class="bg-slate-100 min-h-screen text-slate-800">

<header class="bg-slate-900 text-white px-5 py-3 flex flex-wrap items-center gap-3 sticky top-0 z-10 shadow">
  <span class="text-lg font-semibold tracking-tight">📷 Camera Events</span>
  <a href="/watcher/compare" class="text-slate-400 hover:text-white text-sm ml-2">Compare ↗</a>
  {% if telemetry %}
  <span class="text-xs text-slate-300 flex items-center gap-2 ml-1" title="Stealthcam health (latest poll)">
    <span>🔋 {{ telemetry.battery_pct if telemetry.battery_pct is not none else '—' }}%</span>
    <span>💾 {{ telemetry.sd_card_free_pct if telemetry.sd_card_free_pct is not none else '—' }}% free</span>
    <span>📶 {{ telemetry.signal_strength or '—' }}</span>
    {% if telemetry.errors %}<span class="text-amber-400" title="{{ telemetry.errors | join(', ') }}">⚠ {{ telemetry.errors | length }}</span>{% endif %}
  </span>
  {% endif %}
  {% if cameras %}
  <select id="camera-select"
          class="bg-slate-800 text-slate-200 text-sm rounded px-2 py-1 border border-slate-700 focus:outline-none"
          data-filter="{{ filter }}"
          data-description="{{ description }}"
          onchange="location.href='?filter='+encodeURIComponent(this.dataset.filter)+'&camera='+encodeURIComponent(this.value)+'&description='+encodeURIComponent(this.dataset.description)">
    <option value="">All cameras</option>
    {% for c in cameras %}
    <option value="{{ c }}" {% if camera == c %}selected{% endif %}>{{ c }}</option>
    {% endfor %}
  </select>
  {% endif %}
  <input id="description-input" type="text" value="{{ description }}" placeholder="Filter by description…"
         class="bg-slate-800 text-slate-200 text-sm rounded px-2 py-1 border border-slate-700 focus:outline-none w-48"
         data-filter="{{ filter }}" data-camera="{{ camera or '' }}"
         onkeydown="if(event.key==='Enter'){location.href='?filter='+encodeURIComponent(this.dataset.filter)+'&camera='+encodeURIComponent(this.dataset.camera)+'&description='+encodeURIComponent(this.value)}">
  <nav class="flex gap-1 ml-auto flex-wrap">
    {% set tabs = [
        ('recent',       'Recent',       counts.recent),
        ('interesting',  'Interesting',  counts.interesting),
        ('noise',        'Noise',        counts.noise),
        ('unclassified', 'Needs Review', counts.unclassified),
    ] %}
    {% for f, label, count in tabs %}
    <a href="?filter={{ f }}{% if camera %}&camera={{ camera }}{% endif %}{% if description %}&description={{ description }}{% endif %}"
       class="px-3 py-1 rounded-full text-sm transition-colors flex items-center gap-1
              {% if filter == f %}bg-white text-slate-900 font-medium
              {% else %}text-slate-300 hover:bg-slate-700{% endif %}">
      {{ label }}
      {% if count is not none %}<span class="text-xs opacity-70">{{ count }}</span>{% endif %}
    </a>
    {% endfor %}
  </nav>
</header>

<main class="max-w-3xl mx-auto py-5 px-3 space-y-3" id="event-list">

  {% for ev in events %}
  {% set ml = ev.ml %}
  {% if ml and ml.interesting %}{% set border = 'border-l-green-400' %}
  {% elif ml and not ml.interesting %}{% set border = 'border-l-slate-300' %}
  {% else %}{% set border = 'border-l-amber-300' %}{% endif %}

  <div class="event-card bg-white rounded-xl shadow-sm flex flex-col overflow-hidden border border-slate-200 border-l-4 {{ border }}"
       data-id="{{ ev.id }}" data-classified="{{ '1' if ml else '0' }}" data-video="{{ ev.video_url }}"
       onclick="cardClick(event)">

    <div class="flex">
      <div class="card-thumb flex-shrink-0 cursor-pointer">
        {% if ev.thumb_url %}
        <img src="{{ ev.thumb_url }}" alt="frame" loading="lazy"
             onerror="this.closest('.card-thumb').innerHTML='<div class=\'flex items-center justify-center h-full text-slate-400 text-xs p-2\'>no frame</div>'">
        {% else %}
        <div class="flex items-center justify-center h-full text-slate-400 text-xs p-2">{% if ev.is_photo %}🖼 photo{% else %}▶ video{% endif %}</div>
        {% endif %}
      </div>

      <div class="p-4 flex-1 flex flex-col gap-1 min-w-0">
        <div class="flex items-start justify-between gap-2 flex-wrap">
          <div>
            <div class="font-medium text-sm">{{ ev.capture_time }}</div>
            <div class="text-xs text-slate-400 mt-0.5">
              {{ ev.scene_name }}{% if ev.lighting %} · {{ LIGHTING_LABEL.get(ev.lighting,'') }} {{ ev.lighting }}{% endif %}
            </div>
          </div>

          {% if ml %}
            {% set icon = CATEGORY_ICON.get(ml.category, '❓') %}
            {% if ml.interesting %}
            <span data-badge class="flex-shrink-0 px-2 py-1 rounded-full text-xs font-semibold bg-green-100 text-green-800">
              {{ icon }} {{ ml.category | replace('_',' ') | title }}{% if ml.confidence %} · {{ ml.confidence }}%{% endif %}
            </span>
            {% else %}
            <span data-badge class="flex-shrink-0 px-2 py-1 rounded-full text-xs font-medium bg-slate-100 text-slate-500">
              {{ icon }} {{ ml.category | replace('_',' ') }}{% if ml.confidence %} {{ ml.confidence }}%{% endif %}
            </span>
            {% endif %}
          {% else %}
            <span data-badge class="flex-shrink-0 px-2 py-1 rounded-full text-xs bg-amber-100 text-amber-700">unclassified</span>
          {% endif %}
        </div>

        {% if ml and ml.description %}
        <div class="text-xs text-slate-500 mt-1 italic">{{ ml.description }}</div>
        {% endif %}
        {% if ml %}
        <div class="text-xs text-slate-300">{{ ml.model }}{% if ml.git_version %} · {{ ml.git_version }}{% endif %}</div>
        {% endif %}

        {% if ev.human %}
        <div class="flex items-center gap-1 flex-wrap mt-1">
          <span class="text-xs text-slate-400">Human:</span>
          {% for lbl in ev.human.labels %}
          <span class="px-1.5 py-0.5 rounded bg-blue-100 text-blue-800 text-xs">{{ lbl }}</span>
          {% endfor %}
        </div>
        {% endif %}

        <div class="flex items-center gap-1 mt-1" data-feedback-row>
          <button data-fb="good" onclick="toggleFeedback(this, 'good')"
                  class="fb-btn px-1.5 py-0.5 rounded text-sm leading-none transition-colors
                         {% if ev.feedback and ev.feedback.label == 'good' %}fb-active bg-green-200 ring-1 ring-green-400{% else %}bg-slate-100 hover:bg-slate-200{% endif %}">👍</button>
          <button data-fb="bad" onclick="toggleFeedback(this, 'bad')"
                  class="fb-btn px-1.5 py-0.5 rounded text-sm leading-none transition-colors
                         {% if ev.feedback and ev.feedback.label == 'bad' %}fb-active bg-red-200 ring-1 ring-red-400{% else %}bg-slate-100 hover:bg-slate-200{% endif %}">👎</button>
          {% if ev.feedback and ev.feedback.reason %}
          <span class="text-xs text-slate-400 italic" data-fb-reason>{{ ev.feedback.reason }}</span>
          {% endif %}
        </div>

        {% if ev.is_photo %}
        <button
                class="mt-auto pt-2 text-xs text-blue-500 hover:text-blue-700 hover:underline w-fit text-left">
          🔍 Enlarge
        </button>
        {% else %}
        <button
                class="mt-auto pt-2 text-xs text-blue-500 hover:text-blue-700 hover:underline w-fit text-left">
          ▶ Watch clip
        </button>
        {% endif %}
      </div>
    </div>

    {% if ev.is_photo %}
    <div class="photo-player hidden">
      <img src="{{ ev.frame_url }}" alt="enlarged frame" loading="lazy"
           style="width:100%;display:block;max-height:600px;object-fit:contain;background:#000">
    </div>
    {% else %}
    <div class="video-player hidden">
      <video controls playsinline style="width:100%;display:block;max-height:400px;background:#000">
        <source src="{{ ev.video_url }}" type="video/mp4">
      </video>
    </div>
    {% endif %}

  </div>
  {% else %}
  <div class="text-center text-slate-400 py-16">No events found.</div>
  {% endfor %}

</main>

{% if has_more %}
<div id="scroll-sentinel" class="py-6 text-center text-slate-400 text-sm"
     data-page="{{ page + 1 }}" data-filter="{{ filter }}" data-camera="{{ camera or '' }}" data-description="{{ description }}">
  <span id="scroll-sentinel-label">Loading…</span>
</div>
{% endif %}

<script>
const CATEGORY_ICON = {{ CATEGORY_ICON | tojson }};
const LIGHTING_LABEL = {{ LIGHTING_LABEL | tojson }};
// The app is served under the /watcher prefix (nginx rewrites /watcher/* -> /*
// to the api upstream). All fetch() calls must carry the prefix or they hit
// nginx's static `location /` and 404.
const API_BASE = '/watcher';

function badge(ml) {
  if (!ml) return '<span data-badge class="flex-shrink-0 px-2 py-1 rounded-full text-xs bg-amber-100 text-amber-700">unclassified</span>';
  const icon = CATEGORY_ICON[ml.category] || '❓';
  const conf = ml.confidence ? ` · ${ml.confidence}%` : '';
  if (ml.interesting)
    return `<span data-badge class="flex-shrink-0 px-2 py-1 rounded-full text-xs font-semibold bg-green-100 text-green-800">${icon} ${ml.category.replace(/_/g,' ')}${conf}</span>`;
  return `<span data-badge class="flex-shrink-0 px-2 py-1 rounded-full text-xs font-medium bg-slate-100 text-slate-500">${icon} ${ml.category.replace(/_/g,' ')}${conf}</span>`;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function renderCard(ev) {
  const ml = ev.ml;
  const isPhoto = !!ev.is_photo;
  const border = ml ? (ml.interesting ? 'border-l-green-400' : 'border-l-slate-300') : 'border-l-amber-300';
  const lighting = ev.lighting ? ` · ${LIGHTING_LABEL[ev.lighting] || ''} ${ev.lighting}` : '';
  const thumb = ev.thumb_url
    ? `<img src="${ev.thumb_url}" loading="lazy" style="width:192px;height:108px;object-fit:cover;display:block" onerror="this.closest('.card-thumb').innerHTML='<div style=padding:8px;color:#9ca3af;font-size:12px>no frame</div>'">`
    : `<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#9ca3af;font-size:12px;padding:8px">${isPhoto ? '🖼 photo' : '▶ video'}</div>`;
  const desc = (ml && ml.description)
    ? `<div style="font-size:11px;color:#64748b;margin-top:4px;font-style:italic">${ml.description}</div>` : '';
  const modelName = ml
    ? `<div style="font-size:11px;color:#cbd5e1">${ml.model}${ml.git_version ? ' · ' + ml.git_version : ''}</div>` : '';
  const human = ev.human
    ? `<div style="display:flex;gap:4px;flex-wrap:wrap;margin-top:4px">${ev.human.labels.map(l=>`<span style="font-size:11px;padding:1px 6px;border-radius:4px;background:#dbeafe;color:#1d4ed8">${l}</span>`).join('')}</div>`
    : '';
  const fb = ev.feedback;
  const fbGood = fb && fb.label === 'good' ? 'background:#bbf7d0;box-shadow:0 0 0 1px #4ade80' : 'background:#f1f5f9';
  const fbBad  = fb && fb.label === 'bad'  ? 'background:#fecaca;box-shadow:0 0 0 1px #f87171' : 'background:#f1f5f9';
  const fbGoodCls = fb && fb.label === 'good' ? ' fb-active' : '';
  const fbBadCls  = fb && fb.label === 'bad'  ? ' fb-active' : '';
  const fbReason = (fb && fb.reason) ? `<span style="font-size:11px;color:#94a3b8;font-style:italic" data-fb-reason>${esc(fb.reason)}</span>` : '';
  const feedbackRow = `<div style="display:flex;align-items:center;gap:4px;margin-top:4px" data-feedback-row>
      <button data-fb="good" onclick="toggleFeedback(this,'good')" class="fb-btn${fbGoodCls}" style="font-size:14px;line-height:1;padding:2px 6px;border-radius:4px;border:none;cursor:pointer;${fbGood}">👍</button>
      <button data-fb="bad" onclick="toggleFeedback(this,'bad')" class="fb-btn${fbBadCls}" style="font-size:14px;line-height:1;padding:2px 6px;border-radius:4px;border:none;cursor:pointer;${fbBad}">👎</button>
      ${fbReason}
    </div>`;
  const watchBtn = isPhoto
    ? `<button class="mt-auto pt-2 text-xs text-blue-500 hover:underline w-fit text-left">🔍 Enlarge</button>`
    : `<button class="mt-auto pt-2 text-xs text-blue-500 hover:underline w-fit text-left">▶ Watch clip</button>`;
  const player = isPhoto
    ? `<div class="photo-player hidden">
        <img src="${ev.frame_url}" alt="enlarged frame" loading="lazy" style="width:100%;display:block;max-height:600px;object-fit:contain;background:#000">
      </div>`
    : `<div class="video-player hidden">
        <video controls playsinline style="width:100%;display:block;max-height:400px;background:#000">
          <source src="${ev.video_url}" type="video/mp4">
        </video>
      </div>`;
  return `
    <div class="event-card bg-white rounded-xl shadow-sm flex flex-col overflow-hidden border border-slate-200 border-l-4 ${border}"
         data-id="${ev.id}" data-classified="${ml ? 1 : 0}" data-video="${ev.video_url}" onclick="cardClick(event)">
      <div class="flex">
        <div class="card-thumb flex-shrink-0 cursor-pointer">${thumb}</div>
        <div class="p-4 flex-1 flex flex-col gap-1 min-w-0">
          <div class="flex items-start justify-between gap-2 flex-wrap">
            <div>
              <div class="font-medium text-sm">${ev.capture_time}</div>
              <div class="text-xs text-slate-400 mt-0.5">${ev.scene_name}${lighting}</div>
            </div>
            ${badge(ml)}
          </div>
          ${desc}${modelName}${human}${feedbackRow}
          ${watchBtn}
        </div>
      </div>
      ${player}
    </div>`;
}

// ── inline media player (photo enlarge / video) ──────────────────────────────

// Whole-cell click toggles enlarge/play. Interactive controls that have their
// own handlers (feedback buttons, the video player) are excluded so a 👍/👎
// click or a play/pause tap doesn't also collapse the card.
function cardClick(event) {
  if (event.target.closest('[data-feedback-row]')) return;
  if (event.target.closest('.video-player')) return;
  toggleMedia(event.currentTarget);
}

function toggleMedia(card) {
  const photo = card.querySelector('.photo-player');
  const video = card.querySelector('.video-player');
  const player = photo || video;
  if (!player) return;
  if (player.classList.contains('hidden')) {
    player.classList.remove('hidden');
    if (video) video.play();
  } else {
    if (video) { video.pause(); video.currentTime = 0; }
    player.classList.add('hidden');
  }
}

// ── feedback toggles ─────────────────────────────────────────────────────────

async function toggleFeedback(btn, label) {
  const card = btn.closest('.event-card');
  const eventId = parseInt(card.dataset.id);
  const row = card.querySelector('[data-feedback-row]');
  const goodBtn = row.querySelector('[data-fb="good"]');
  const badBtn  = row.querySelector('[data-fb="bad"]');
  const reasonEl = row.querySelector('[data-fb-reason]');

  // If the clicked button is already active, this is an undo — no reason needed.
  const isUndo = btn.classList.contains('fb-active');

  let reason = '';
  if (label === 'bad' && !isUndo) {
    reason = prompt('Why is this wrong? (helps the agent improve)');
    if (reason === null) return; // cancelled
  }

  try {
    const resp = await fetch(API_BASE + '/feedback', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({event_id: eventId, label: label, reason: reason}),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      alert('Feedback failed: ' + (err.error || resp.status));
      return;
    }
    const data = await resp.json();
    applyFeedbackState(row, data.feedback);
  } catch (e) {
    alert('Feedback failed: network error');
  }
}

function applyFeedbackState(row, fb) {
  const goodBtn = row.querySelector('[data-fb="good"]');
  const badBtn  = row.querySelector('[data-fb="bad"]');
  const reasonEl = row.querySelector('[data-fb-reason]');

  const active = fb ? fb.label : null;
  goodBtn.classList.toggle('fb-active', active === 'good');
  badBtn.classList.toggle('fb-active', active === 'bad');
  // Clear server-rendered Tailwind classes so inline styles are authoritative.
  goodBtn.classList.remove('bg-green-200', 'ring-1', 'ring-green-400', 'bg-slate-100', 'hover:bg-slate-200');
  badBtn.classList.remove('bg-red-200', 'ring-1', 'ring-red-400', 'bg-slate-100', 'hover:bg-slate-200');
  goodBtn.style.background = active === 'good' ? '#bbf7d0' : '#f1f5f9';
  goodBtn.style.boxShadow = active === 'good' ? '0 0 0 1px #4ade80' : 'none';
  badBtn.style.background  = active === 'bad'  ? '#fecaca' : '#f1f5f9';
  badBtn.style.boxShadow  = active === 'bad'  ? '0 0 0 1px #f87171' : 'none';

  if (reasonEl) reasonEl.remove();
  if (fb && fb.reason) {
    const span = document.createElement('span');
    span.dataset.fbReason = '';
    span.style.cssText = 'font-size:11px;color:#94a3b8;font-style:italic';
    span.textContent = fb.reason;
    row.appendChild(span);
  }
}

// ── auto-refresh ─────────────────────────────────────────────────────────────

const POLL_INTERVAL = 30_000; // ms
const activeFilter = new URLSearchParams(location.search).get('filter') || 'recent';
const activeCamera = new URLSearchParams(location.search).get('camera') || '';
const activeDescription = new URLSearchParams(location.search).get('description') || '';

// Seed maxId from server-rendered cards
let maxId = 0;
document.querySelectorAll('.event-card[data-id]').forEach(el => {
  maxId = Math.max(maxId, parseInt(el.dataset.id));
});

function getUnclassifiedIds() {
  return Array.from(document.querySelectorAll('.event-card[data-id][data-classified="0"]'))
              .map(el => parseInt(el.dataset.id));
}

async function poll() {
  try {
    // 1. Fetch any events newer than what we have
    const newResp = await fetch(`${API_BASE}/events?filter=${activeFilter}&camera=${encodeURIComponent(activeCamera)}&description=${encodeURIComponent(activeDescription)}&since_id=${maxId}`);
    const newData = await newResp.json();
    if (newData.events.length) {
      const list = document.getElementById('event-list');
      newData.events.forEach(ev => {
        list.insertAdjacentHTML('afterbegin', renderCard(ev));
        maxId = Math.max(maxId, ev.id);
      });
    }

    // 2. Re-fetch unclassified cards to see if labels have arrived
    const pendingIds = getUnclassifiedIds();
    if (pendingIds.length) {
      const qs = pendingIds.map(id => `id=${id}`).join('&');
      const updResp = await fetch(`${API_BASE}/events?${qs}`);
      const updData = await updResp.json();
      updData.events.forEach(ev => {
        if (!ev.ml) return; // still unclassified
        const card = document.querySelector(`.event-card[data-id="${ev.id}"]`);
        if (!card) return;
        // Swap badge and border
        const badgeEl = card.querySelector('[data-badge]');
        if (badgeEl) badgeEl.outerHTML = badge(ev.ml);
        const border = ev.ml.interesting ? 'border-l-green-400' : 'border-l-slate-300';
        card.classList.remove('border-l-amber-300', 'border-l-green-400', 'border-l-slate-300');
        card.classList.add(border);
        card.dataset.classified = '1';
      });
    }
  } catch(e) { /* network hiccup — try again next interval */ }
}

setInterval(poll, POLL_INTERVAL);

// ─────────────────────────────────────────────────────────────────────────────

// ── infinite scroll ──────────────────────────────────────────────────────────
// When the sentinel scrolls into view, fetch the next page and append it.
// The sentinel is re-armed with the next page number until has_more is false.
//
// Loading is driven by an explicit position check (sentinelInView) that runs on
// scroll AND after every load, with IntersectionObserver kept as an extra
// trigger. This is deliberate: IntersectionObserver alone is unreliable here —
// (a) browsers suppress IO callbacks while the tab is hidden/backgrounded, and
// (b) IO only fires on intersection *changes*, so if the sentinel is already in
// view when a page finishes loading (short cards / tall viewport) no further
// callback fires and loading stalls. The position check covers both cases.

let loading = false;

function sentinelInView() {
  const sentinel = document.getElementById('scroll-sentinel');
  if (!sentinel) return false;
  const r = sentinel.getBoundingClientRect();
  const margin = 200; // px — match the IO rootMargin
  return r.top < window.innerHeight + margin && r.bottom > -margin;
}

async function loadMore() {
  const sentinel = document.getElementById('scroll-sentinel');
  if (!sentinel || loading) return;
  loading = true;
  const page = parseInt(sentinel.dataset.page);
  const filter = sentinel.dataset.filter;
  const camera = sentinel.dataset.camera || '';
  const description = sentinel.dataset.description || '';
  const label = document.getElementById('scroll-sentinel-label');
  if (label) label.textContent = 'Loading…';
  try {
    const resp = await fetch(`${API_BASE}/events?page=${page}&filter=${filter}&camera=${encodeURIComponent(camera)}&description=${encodeURIComponent(description)}`);
    const data = await resp.json();
    const list = document.getElementById('event-list');
    data.events.forEach(ev => {
      list.insertAdjacentHTML('beforeend', renderCard(ev));
      maxId = Math.max(maxId, ev.id);
    });
    if (data.has_more) {
      sentinel.dataset.page = page + 1;
      if (label) label.textContent = 'Scroll for more…';
    } else {
      sentinel.remove();
    }
  } catch(e) {
    if (label) label.textContent = 'Error — scroll to retry';
  } finally {
    loading = false;
    // If the sentinel is still in view after this page (short cards / tall
    // viewport), keep loading until it scrolls out of view or has_more is false.
    if (sentinelInView()) loadMore();
  }
}

const sentinel = document.getElementById('scroll-sentinel');
if (sentinel) {
  // Primary trigger: explicit position check on scroll (fires even in hidden
  // tabs, unlike IntersectionObserver).
  window.addEventListener('scroll', () => {
    if (sentinelInView()) loadMore();
  }, {passive: true});

  // Extra trigger: IntersectionObserver, when available.
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver((entries) => {
      if (entries.some(e => e.isIntersecting)) loadMore();
    }, {rootMargin: '200px'});
    observer.observe(sentinel);
  } else {
    // Fallback: keep the sentinel clickable if IntersectionObserver is unavailable.
    sentinel.style.cursor = 'pointer';
    sentinel.addEventListener('click', loadMore);
  }

  // Initial check: if the sentinel is already in view on first paint (e.g. a
  // short first page), start loading immediately.
  if (sentinelInView()) loadMore();
}
</script>
</body>
</html>"""


_COMPARE_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Model Comparison</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    .thumb { width:120px; height:68px; object-fit:cover; flex-shrink:0; }
    .col-cell { min-width:160px; max-width:220px; }
  </style>
</head>
<body class="bg-slate-100 min-h-screen text-slate-800">

<header class="bg-slate-900 text-white px-5 py-3 flex items-center gap-4 shadow">
  <a href="/watcher/browser" class="text-slate-400 hover:text-white text-sm">← Browser</a>
  <span class="text-lg font-semibold tracking-tight">📊 Model Comparison</span>
  <span class="text-slate-400 text-sm ml-2">{{ rows|length }} events · {{ models|length }} models</span>
  <select class="bg-slate-800 text-slate-200 text-sm rounded px-2 py-1 border border-slate-700 focus:outline-none"
          data-description="{{ description }}"
          onchange="location.href='?camera='+encodeURIComponent(this.value)+'&description='+encodeURIComponent(this.dataset.description)">
    <option value="">All cameras</option>
    {% for c in cameras %}
    <option value="{{ c }}" {% if camera == c %}selected{% endif %}>{{ c }}</option>
    {% endfor %}
  </select>
  <input type="text" value="{{ description }}" placeholder="Filter by description…"
         class="bg-slate-800 text-slate-200 text-sm rounded px-2 py-1 border border-slate-700 focus:outline-none w-48"
         data-camera="{{ camera or '' }}"
         onkeydown="if(event.key==='Enter'){location.href='?camera='+encodeURIComponent(this.dataset.camera)+'&description='+encodeURIComponent(this.value)}">
</header>

{% if not rows %}
<div class="text-center text-slate-400 py-20">
  No events with multiple model classifications yet.<br>
  <span class="text-sm">Run backfill_classify with different -m flags to populate this view.</span>
</div>
{% else %}
<div style="overflow:auto; height:calc(100vh - 52px)">
<table class="w-full text-sm border-collapse">
  <thead class="sticky top-0 z-20">
    <tr class="bg-white border-b border-slate-200 shadow-sm">
      <th class="sticky left-0 bg-white px-4 py-3 text-left text-xs font-semibold text-slate-500 uppercase min-w-48 z-30">Event</th>
      {% for m in models %}
      <th class="bg-white px-4 py-3 text-left text-xs font-semibold text-slate-500 uppercase col-cell border-l border-slate-100">
        {{ m }}
      </th>
      {% endfor %}
    </tr>
  </thead>
  <tbody>
  {% for row in rows %}
  <tr class="bg-white border-b border-slate-100 hover:bg-slate-50">

    {# thumbnail + time #}
    <td class="sticky left-0 bg-white px-3 py-2 z-10 shadow-[1px_0_0_0_#e2e8f0]">
      <div class="flex gap-2 items-start">
        {% if row.frame_url %}
          {% if row.is_photo %}
          <a href="{{ row.video_url }}" target="_blank">
            <img src="{{ row.frame_url }}" class="thumb rounded"
                 onerror="this.style.display='none'">
          </a>
          {% else %}
          <img src="{{ row.frame_url }}" class="thumb rounded cursor-pointer"
               onclick="toggleVid(this, '{{ row.video_url }}')"
               onerror="this.style.display='none'">
          {% endif %}
        {% endif %}
        <div class="min-w-0">
          <div class="font-medium text-xs leading-tight">{{ row.capture_time }}</div>
          <div class="text-xs text-slate-400">{{ row.scene_name }}</div>
        </div>
      </div>
      {% if not row.is_photo %}
      <video id="vid-{{ row.id }}" controls playsinline class="hidden mt-1 rounded"
             style="width:360px;max-height:600px">
        <source src="{{ row.video_url }}" type="video/mp4">
      </video>
      {% endif %}
    </td>

    {# one column per model #}
    {% for m in models %}
    {% set r = row.models.get(m) %}
    <td class="px-4 py-2 align-top col-cell border-l border-slate-100
               {% if r and r.interesting %}bg-green-50
               {% elif r and not r.interesting %}bg-slate-50
               {% else %}bg-amber-50{% endif %}">
      {% if r %}
        {% set icon = CATEGORY_ICON.get(r.category, '❓') %}
        <div class="flex items-center gap-1 flex-wrap">
          {% if r.interesting %}
          <span class="px-1.5 py-0.5 rounded-full text-xs font-semibold bg-green-100 text-green-800">
            {{ icon }} {{ r.category | replace('_',' ') | title }}{% if r.confidence is not none %} · {{ r.confidence }}%{% endif %}
          </span>
          {% else %}
          <span class="px-1.5 py-0.5 rounded-full text-xs bg-slate-200 text-slate-500">
            {{ icon }} {{ r.category | replace('_',' ') }}{% if r.confidence is not none %} {{ r.confidence }}%{% endif %}
          </span>
          {% endif %}
        </div>
        {% if r.description %}
        <div class="text-xs text-slate-600 mt-1 leading-snug">{{ r.description }}</div>
        {% endif %}
        <div class="text-xs text-slate-300 mt-0.5">
          {% if r.confidence is not none %}{{ r.confidence }}%{% endif %}
          {% if r.git_version %}<span class="ml-1">{{ r.git_version }}</span>{% endif %}
        </div>
      {% else %}
        <span class="text-xs text-slate-300">—</span>
      {% endif %}
    </td>
    {% endfor %}

  </tr>
  {% endfor %}
  </tbody>
</table>
</div>
{% endif %}

<script>
function toggleVid(img, url) {
  const vid = document.getElementById('vid-' + img.closest('tr').querySelector('[id^=vid-]').id.split('-')[1]);
  if (vid.classList.contains('hidden')) {
    vid.classList.remove('hidden');
    vid.play();
  } else {
    vid.pause();
    vid.classList.add('hidden');
  }
}
</script>
</body>
</html>"""
