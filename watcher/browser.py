from datetime import datetime, timedelta, timezone

import pytz
from flask import Blueprint, render_template_string, request, jsonify
from sqlalchemy import select, desc, func, exists, Text
from sqlalchemy.orm import joinedload

from .model import EventObservation, Labeling, IntermediateResult
from .connection import application_config

__all__ = ['browser_bp']

browser_bp = Blueprint('browser', __name__)

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
    # Use the most recently decided ollama labeling (highest id)
    ollama = [l for l in event.labelings if l.decider and l.decider.startswith('ollama:')]
    lbl = max(ollama, key=lambda l: l.id) if ollama else None
    if not lbl:
        return None
    cats = [l for l in lbl.labels if l != 'noise']
    confidence = round(lbl.probabilities[0] * 100) if lbl.probabilities else None
    return {
        'category':    cats[0] if cats else 'unknown',
        'interesting': 'noise' not in lbl.labels,
        'confidence':  confidence,
        'decider':     lbl.decider,
    }


def _human_info(event):
    lbl = next((l for l in event.labelings if not l.probabilities), None)
    if not lbl:
        return None
    return {'labels': lbl.labels, 'decider': lbl.decider}


def _serialize(event):
    return {
        'id':           event.id,
        'event_name':   event.event_name,
        'capture_time': _fmt_time(event.capture_time),
        'scene_name':   event.scene_name or '',
        'lighting':     event.lighting_type or '',
        'video_url':    event.video_url,
        'frame_url':    event.significant_frame_url,
        'ml':           _ml_info(event),
        'human':        _human_info(event),
    }


def _recent_cutoff():
    return datetime.now(timezone.utc) - timedelta(days=RECENT_DAYS)


def _base_stmt():
    return (select(EventObservation)
            .options(joinedload(EventObservation.labelings),
                     joinedload(EventObservation.results))
            .order_by(desc(EventObservation.capture_time)))


def _has_frame_subq():
    return select(IntermediateResult.event_id).correlate(EventObservation)


def _fetch(db_session, page, filter_mode):
    stmt = _base_stmt()

    if filter_mode == 'recent':
        stmt = stmt.where(EventObservation.capture_time >= _recent_cutoff())

    elif filter_mode == 'interesting':
        ids = (select(Labeling.event_id)
               .where(Labeling.decider.like('ollama:%'))
               .where(Labeling.labels.cast(Text).not_like('%noise%')))
        stmt = stmt.where(EventObservation.id.in_(ids))

    elif filter_mode == 'noise':
        ids = (select(Labeling.event_id)
               .where(Labeling.decider.like('ollama:%'))
               .where(Labeling.labels.cast(Text).like('%noise%')))
        stmt = stmt.where(EventObservation.id.in_(ids))

    elif filter_mode == 'unclassified':
        classified = select(Labeling.event_id).where(Labeling.decider.like('ollama:%'))
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
    classified_ids = select(Labeling.event_id).where(Labeling.decider.like('ollama:%'))

    recent = db_session.execute(
        select(func.count()).where(EventObservation.capture_time >= cutoff)
    ).scalar()
    interesting = db_session.execute(
        select(func.count(Labeling.event_id.distinct()))
        .where(Labeling.decider.like('ollama:%'))
        .where(Labeling.labels.cast(Text).not_like('%noise%'))
    ).scalar()
    noise = db_session.execute(
        select(func.count(Labeling.event_id.distinct()))
        .where(Labeling.decider.like('ollama:%'))
        .where(Labeling.labels.cast(Text).like('%noise%'))
    ).scalar()
    unclassified = db_session.execute(
        select(func.count()).select_from(EventObservation)
        .where(EventObservation.capture_time >= cutoff)
        .where(EventObservation.id.notin_(classified_ids))
        .where(EventObservation.id.in_(has_frame))
    ).scalar()
    return dict(recent=recent, interesting=interesting, noise=noise, unclassified=unclassified)


# ── routes ────────────────────────────────────────────────────────────────────

@browser_bp.route('/browser')
def browser_page():
    from api import db

    filter_mode = request.args.get('filter', 'recent')
    page = request.args.get('page', 1, type=int)

    events, has_more = _fetch(db.session, page, filter_mode)
    counts = _counts(db.session)

    return render_template_string(
        _TEMPLATE,
        events=events,
        filter=filter_mode,
        page=page,
        has_more=has_more,
        counts=counts,
        CATEGORY_ICON=CATEGORY_ICON,
        LIGHTING_LABEL=LIGHTING_LABEL,
    )


@browser_bp.route('/events')
def events_json():
    from api import db

    filter_mode = request.args.get('filter', 'recent')
    page = request.args.get('page', 1, type=int)

    events, has_more = _fetch(db.session, page, filter_mode)
    return jsonify({'events': events, 'has_more': has_more, 'page': page})


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
  <nav class="flex gap-1 ml-auto flex-wrap">
    {% set tabs = [
        ('recent',       'Recent',       counts.recent),
        ('interesting',  'Interesting',  counts.interesting),
        ('noise',        'Noise',        counts.noise),
        ('unclassified', 'Needs Review', counts.unclassified),
    ] %}
    {% for f, label, count in tabs %}
    <a href="?filter={{ f }}"
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

  <div class="event-card bg-white rounded-xl shadow-sm flex overflow-hidden border border-slate-200 border-l-4 {{ border }}">

    <div class="card-thumb flex-shrink-0">
      {% if ev.frame_url %}
      <a href="{{ ev.video_url }}" target="_blank" title="Watch video">
        <img src="{{ ev.frame_url }}" alt="frame" loading="lazy"
             onerror="this.closest('.card-thumb').innerHTML='<div class=\'flex items-center justify-center h-full text-slate-400 text-xs p-2\'>no frame</div>'">
      </a>
      {% else %}
      <a href="{{ ev.video_url }}" target="_blank"
         class="flex items-center justify-center h-full text-slate-400 text-xs p-2 hover:bg-slate-100">
        ▶ video
      </a>
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
          <span class="flex-shrink-0 px-2 py-1 rounded-full text-xs font-semibold bg-green-100 text-green-800">
            {{ icon }} {{ ml.category | replace('_',' ') | title }}{% if ml.confidence %} · {{ ml.confidence }}%{% endif %}
          </span>
          {% else %}
          <span class="flex-shrink-0 px-2 py-1 rounded-full text-xs font-medium bg-slate-100 text-slate-500">
            {{ icon }} {{ ml.category | replace('_',' ') }}{% if ml.confidence %} {{ ml.confidence }}%{% endif %}
          </span>
          {% endif %}
        {% else %}
          <span class="flex-shrink-0 px-2 py-1 rounded-full text-xs bg-amber-100 text-amber-700">unclassified</span>
        {% endif %}
      </div>

      {% if ev.human %}
      <div class="flex items-center gap-1 flex-wrap mt-1">
        <span class="text-xs text-slate-400">Human:</span>
        {% for lbl in ev.human.labels %}
        <span class="px-1.5 py-0.5 rounded bg-blue-100 text-blue-800 text-xs">{{ lbl }}</span>
        {% endfor %}
      </div>
      {% endif %}

      <a href="{{ ev.video_url }}" target="_blank"
         class="mt-auto pt-2 text-xs text-blue-500 hover:text-blue-700 hover:underline w-fit">
        ▶ Watch clip
      </a>
    </div>
  </div>
  {% else %}
  <div class="text-center text-slate-400 py-16">No events found.</div>
  {% endfor %}

</main>

{% if has_more %}
<div class="text-center py-6" id="load-more-wrap">
  <button id="load-more-btn"
          class="px-6 py-2 bg-slate-800 text-white text-sm rounded-full hover:bg-slate-700 transition-colors"
          data-page="{{ page + 1 }}" data-filter="{{ filter }}">
    Load more
  </button>
</div>
{% endif %}

<script>
const CATEGORY_ICON = {{ CATEGORY_ICON | tojson }};
const LIGHTING_LABEL = {{ LIGHTING_LABEL | tojson }};

function badge(ml) {
  if (!ml) return '<span class="flex-shrink-0 px-2 py-1 rounded-full text-xs bg-amber-100 text-amber-700">unclassified</span>';
  const icon = CATEGORY_ICON[ml.category] || '❓';
  const conf = ml.confidence ? ` · ${ml.confidence}%` : '';
  if (ml.interesting)
    return `<span class="flex-shrink-0 px-2 py-1 rounded-full text-xs font-semibold bg-green-100 text-green-800">${icon} ${ml.category.replace(/_/g,' ')}${conf}</span>`;
  return `<span class="flex-shrink-0 px-2 py-1 rounded-full text-xs font-medium bg-slate-100 text-slate-500">${icon} ${ml.category.replace(/_/g,' ')}${conf}</span>`;
}

function renderCard(ev) {
  const ml = ev.ml;
  const border = ml ? (ml.interesting ? 'border-l-green-400' : 'border-l-slate-300') : 'border-l-amber-300';
  const lighting = ev.lighting ? ` · ${LIGHTING_LABEL[ev.lighting] || ''} ${ev.lighting}` : '';
  const thumb = ev.frame_url
    ? `<a href="${ev.video_url}" target="_blank"><img src="${ev.frame_url}" loading="lazy" style="width:192px;height:108px;object-fit:cover;display:block" onerror="this.closest('.card-thumb').innerHTML='<div style=padding:8px;color:#9ca3af;font-size:12px>no frame</div>'"></a>`
    : `<a href="${ev.video_url}" target="_blank" style="display:flex;align-items:center;justify-content:center;height:100%;color:#9ca3af;font-size:12px;padding:8px">▶ video</a>`;
  const human = ev.human
    ? `<div style="display:flex;gap:4px;flex-wrap:wrap;margin-top:4px">${ev.human.labels.map(l=>`<span style="font-size:11px;padding:1px 6px;border-radius:4px;background:#dbeafe;color:#1d4ed8">${l}</span>`).join('')}</div>`
    : '';
  return `
    <div class="event-card bg-white rounded-xl shadow-sm flex overflow-hidden border border-slate-200 border-l-4 ${border}">
      <div class="card-thumb flex-shrink-0">${thumb}</div>
      <div class="p-4 flex-1 flex flex-col gap-1 min-w-0">
        <div class="flex items-start justify-between gap-2 flex-wrap">
          <div>
            <div class="font-medium text-sm">${ev.capture_time}</div>
            <div class="text-xs text-slate-400 mt-0.5">${ev.scene_name}${lighting}</div>
          </div>
          ${badge(ml)}
        </div>
        ${human}
        <a href="${ev.video_url}" target="_blank" class="mt-auto pt-2 text-xs text-blue-500 hover:underline w-fit">▶ Watch clip</a>
      </div>
    </div>`;
}

document.getElementById('load-more-btn')?.addEventListener('click', async function() {
  const btn = this;
  const page = parseInt(btn.dataset.page);
  const filter = btn.dataset.filter;
  btn.textContent = 'Loading…';
  btn.disabled = true;
  try {
    const resp = await fetch(`/events?page=${page}&filter=${filter}`);
    const data = await resp.json();
    const list = document.getElementById('event-list');
    data.events.forEach(ev => list.insertAdjacentHTML('beforeend', renderCard(ev)));
    if (data.has_more) {
      btn.dataset.page = page + 1;
      btn.textContent = 'Load more';
      btn.disabled = false;
    } else {
      document.getElementById('load-more-wrap').remove();
    }
  } catch(e) {
    btn.textContent = 'Error — try again';
    btn.disabled = false;
  }
});
</script>
</body>
</html>"""
