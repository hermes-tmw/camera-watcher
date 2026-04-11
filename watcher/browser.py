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
    # Strip 'ollama:' prefix for display
    model_name = lbl.decider.removeprefix('ollama:')
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
    page        = request.args.get('page', 1, type=int)
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
        rows = db.session.execute(stmt).scalars().unique().all()
        return jsonify({'events': [_serialize(e) for e in rows]})

    events, has_more = _fetch(db.session, page, filter_mode)
    return jsonify({'events': events, 'has_more': has_more, 'page': page})


# ── compare route ────────────────────────────────────────────────────────────

def _compare_data(db_session, limit=50):
    """Return events that have ≥2 distinct ollama models, with per-model results."""
    from sqlalchemy import func

    # Events with labelings from multiple different ollama models
    multi = (select(Labeling.event_id)
             .where(Labeling.decider.like('ollama:%'))
             .group_by(Labeling.event_id)
             .having(func.count(Labeling.decider.distinct()) >= 1))

    stmt = (_base_stmt()
            .where(EventObservation.id.in_(multi))
            .where(EventObservation.capture_time >= _recent_cutoff())
            .limit(limit))

    events = db_session.execute(stmt).scalars().unique().all()

    # Collect all distinct model names across these events
    all_models = sorted({
        l.decider.removeprefix('ollama:')
        for e in events for l in e.labelings
        if l.decider and l.decider.startswith('ollama:')
    })

    rows = []
    for ev in events:
        by_model = {}
        for lbl in sorted(ev.labelings, key=lambda l: l.id):
            if not lbl.decider or not lbl.decider.startswith('ollama:'):
                continue
            mname = lbl.decider.removeprefix('ollama:')
            cats = [l for l in lbl.labels if l != 'noise']
            by_model[mname] = {
                'category':    cats[0] if cats else 'unknown',
                'interesting': 'noise' not in lbl.labels,
                'confidence':  round(lbl.probabilities[0] * 100) if lbl.probabilities else None,
                'description': lbl.description or '',
                'git_version': (lbl.git_version or '')[:7],
            }
        rows.append({
            'id':           ev.id,
            'capture_time': _fmt_time(ev.capture_time),
            'scene_name':   ev.scene_name or '',
            'frame_url':    ev.significant_frame_url,
            'video_url':    ev.video_url,
            'models':       by_model,
        })

    return rows, all_models


@browser_bp.route('/compare')
def compare_page():
    from api import db
    rows, models = _compare_data(db.session)
    return render_template_string(_COMPARE_TEMPLATE, rows=rows, models=models,
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

  <div class="event-card bg-white rounded-xl shadow-sm flex flex-col overflow-hidden border border-slate-200 border-l-4 {{ border }}"
       data-id="{{ ev.id }}" data-classified="{{ '1' if ml else '0' }}" data-video="{{ ev.video_url }}">

    <div class="flex">
      <div class="card-thumb flex-shrink-0 cursor-pointer" onclick="toggleVideo(this.closest('.event-card'))">
        {% if ev.frame_url %}
        <img src="{{ ev.frame_url }}" alt="frame" loading="lazy"
             onerror="this.closest('.card-thumb').innerHTML='<div class=\'flex items-center justify-center h-full text-slate-400 text-xs p-2\'>no frame</div>'">
        {% else %}
        <div class="flex items-center justify-center h-full text-slate-400 text-xs p-2">▶ video</div>
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

        <button onclick="toggleVideo(this.closest('.event-card'))"
                class="mt-auto pt-2 text-xs text-blue-500 hover:text-blue-700 hover:underline w-fit text-left">
          ▶ Watch clip
        </button>
      </div>
    </div>

    <div class="video-player hidden">
      <video controls playsinline style="width:100%;display:block;max-height:400px;background:#000">
        <source src="{{ ev.video_url }}" type="video/mp4">
      </video>
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
  if (!ml) return '<span data-badge class="flex-shrink-0 px-2 py-1 rounded-full text-xs bg-amber-100 text-amber-700">unclassified</span>';
  const icon = CATEGORY_ICON[ml.category] || '❓';
  const conf = ml.confidence ? ` · ${ml.confidence}%` : '';
  if (ml.interesting)
    return `<span data-badge class="flex-shrink-0 px-2 py-1 rounded-full text-xs font-semibold bg-green-100 text-green-800">${icon} ${ml.category.replace(/_/g,' ')}${conf}</span>`;
  return `<span data-badge class="flex-shrink-0 px-2 py-1 rounded-full text-xs font-medium bg-slate-100 text-slate-500">${icon} ${ml.category.replace(/_/g,' ')}${conf}</span>`;
}

function renderCard(ev) {
  const ml = ev.ml;
  const border = ml ? (ml.interesting ? 'border-l-green-400' : 'border-l-slate-300') : 'border-l-amber-300';
  const lighting = ev.lighting ? ` · ${LIGHTING_LABEL[ev.lighting] || ''} ${ev.lighting}` : '';
  const thumb = ev.frame_url
    ? `<a href="${ev.video_url}" target="_blank"><img src="${ev.frame_url}" loading="lazy" style="width:192px;height:108px;object-fit:cover;display:block" onerror="this.closest('.card-thumb').innerHTML='<div style=padding:8px;color:#9ca3af;font-size:12px>no frame</div>'"></a>`
    : `<a href="${ev.video_url}" target="_blank" style="display:flex;align-items:center;justify-content:center;height:100%;color:#9ca3af;font-size:12px;padding:8px">▶ video</a>`;
  const desc = (ml && ml.description)
    ? `<div style="font-size:11px;color:#64748b;margin-top:4px;font-style:italic">${ml.description}</div>` : '';
  const modelName = ml
    ? `<div style="font-size:11px;color:#cbd5e1">${ml.model}${ml.git_version ? ' · ' + ml.git_version : ''}</div>` : '';
  const human = ev.human
    ? `<div style="display:flex;gap:4px;flex-wrap:wrap;margin-top:4px">${ev.human.labels.map(l=>`<span style="font-size:11px;padding:1px 6px;border-radius:4px;background:#dbeafe;color:#1d4ed8">${l}</span>`).join('')}</div>`
    : '';
  return `
    <div class="event-card bg-white rounded-xl shadow-sm flex flex-col overflow-hidden border border-slate-200 border-l-4 ${border}"
         data-id="${ev.id}" data-classified="${ml ? 1 : 0}" data-video="${ev.video_url}">
      <div class="flex">
        <div class="card-thumb flex-shrink-0 cursor-pointer" onclick="toggleVideo(this.closest('.event-card'))">${thumb}</div>
        <div class="p-4 flex-1 flex flex-col gap-1 min-w-0">
          <div class="flex items-start justify-between gap-2 flex-wrap">
            <div>
              <div class="font-medium text-sm">${ev.capture_time}</div>
              <div class="text-xs text-slate-400 mt-0.5">${ev.scene_name}${lighting}</div>
            </div>
            ${badge(ml)}
          </div>
          ${desc}${modelName}${human}
          <button onclick="toggleVideo(this.closest('.event-card'))" class="mt-auto pt-2 text-xs text-blue-500 hover:underline w-fit text-left">▶ Watch clip</button>
        </div>
      </div>
      <div class="video-player hidden">
        <video controls playsinline style="width:100%;display:block;max-height:400px;background:#000">
          <source src="${ev.video_url}" type="video/mp4">
        </video>
      </div>
    </div>`;
}

// ── inline video player ───────────────────────────────────────────────────────

function toggleVideo(card) {
  const player = card.querySelector('.video-player');
  const video  = card.querySelector('video');
  if (player.classList.contains('hidden')) {
    player.classList.remove('hidden');
    video.play();
  } else {
    video.pause();
    video.currentTime = 0;
    player.classList.add('hidden');
  }
}

// ── auto-refresh ─────────────────────────────────────────────────────────────

const POLL_INTERVAL = 30_000; // ms
const activeFilter = new URLSearchParams(location.search).get('filter') || 'recent';

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
    const newResp = await fetch(`/events?filter=${activeFilter}&since_id=${maxId}`);
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
      const updResp = await fetch(`/events?${qs}`);
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

<header class="bg-slate-900 text-white px-5 py-3 flex items-center gap-4 sticky top-0 z-10 shadow">
  <a href="/watcher/browser" class="text-slate-400 hover:text-white text-sm">← Browser</a>
  <span class="text-lg font-semibold tracking-tight">📊 Model Comparison</span>
  <span class="text-slate-400 text-sm ml-2">{{ rows|length }} events · {{ models|length }} models</span>
</header>

{% if not rows %}
<div class="text-center text-slate-400 py-20">
  No events with multiple model classifications yet.<br>
  <span class="text-sm">Run backfill_classify with different -m flags to populate this view.</span>
</div>
{% else %}
<div class="overflow-x-auto">
<table class="w-full text-sm border-collapse">
  <thead>
    <tr class="bg-white border-b border-slate-200">
      <th class="sticky left-0 bg-white px-4 py-3 text-left text-xs font-semibold text-slate-500 uppercase min-w-48 z-10">Event</th>
      {% for m in models %}
      <th class="px-4 py-3 text-left text-xs font-semibold text-slate-500 uppercase col-cell border-l border-slate-100">
        {{ m }}
      </th>
      {% endfor %}
    </tr>
  </thead>
  <tbody>
  {% for row in rows %}
  <tr class="bg-white border-b border-slate-100 hover:bg-slate-50">

    {# thumbnail + time #}
    <td class="sticky left-0 bg-white px-3 py-2 z-10">
      <div class="flex gap-2 items-start">
        {% if row.frame_url %}
        <img src="{{ row.frame_url }}" class="thumb rounded cursor-pointer"
             onclick="toggleVid(this, '{{ row.video_url }}')"
             onerror="this.style.display='none'">
        {% endif %}
        <div class="min-w-0">
          <div class="font-medium text-xs leading-tight">{{ row.capture_time }}</div>
          <div class="text-xs text-slate-400">{{ row.scene_name }}</div>
        </div>
      </div>
      <video id="vid-{{ row.id }}" controls playsinline class="hidden mt-1 rounded"
             style="width:360px;max-height:600px">
        <source src="{{ row.video_url }}" type="video/mp4">
      </video>
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
            {{ icon }} {{ r.category | replace('_',' ') | title }}
          </span>
          {% else %}
          <span class="px-1.5 py-0.5 rounded-full text-xs bg-slate-200 text-slate-500">
            {{ icon }} {{ r.category | replace('_',' ') }}
          </span>
          {% endif %}
          {% if r.confidence is not none %}
          <span class="text-xs text-slate-400">{{ r.confidence }}%</span>
          {% endif %}
        </div>
        {% if r.description %}
        <div class="text-xs text-slate-500 mt-1 italic leading-snug">{{ r.description }}</div>
        {% endif %}
        {% if r.git_version %}
        <div class="text-xs text-slate-300 mt-0.5">{{ r.git_version }}</div>
        {% endif %}
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
