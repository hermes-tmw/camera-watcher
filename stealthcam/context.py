"""Capture context: spatial region + day/night classification.

Two small, pure helpers the memory decision and the worker both need:

- ``resolve_region`` maps a direction + bbox to a named spatial region
  (``driveway`` / ``road`` / None-unclassified) per the location policy.
- ``is_night`` decides the day/night gate for a capture timestamp, reusing the
  same US/Mountain wall-hour convention as ``dashboard.py``'s ``_lighting``.

These are kept free of the detector/describer so they can be unit-tested
without the heavy imports.
"""
from __future__ import annotations

from datetime import datetime

import pytz

CAMERA_TZ = "US/Mountain"

# The wall-hour window treated as "day". Everything else is night. These are
# the DEFAULTS; the worker/config can override them per settled decision 4
# ("configured dark hours") via the `day_start_hour` / `day_end_hour` kwargs.
DAY_START_HOUR = 7
DAY_END_HOUR = 20  # exclusive


def resolve_region(direction: str,
                   bbox: tuple[float, float, float, float] | None = None,
                   frame: tuple[int, int] | None = None,
                   mapping: dict[str, str] | None = None) -> str | None:
    """Return the named region for a detection, or None if unclassified.

    The location policy (settled decision 3):

    - ``driveway`` — vehicles here get identity-expectedness evaluation
      (unfamiliar -> alert).
    - ``road``     — vehicles here are pass-by: dashboard note tagged
      ``traffic``, no alert.
    - None         — unclassified frame region -> fall back to the current
      novel-vehicle policy.

    The camera names sub-images by COMPASS bearing (e.g. "East",
    "South_SouthWest"). Which bearing faces the driveway vs the road is
    deployment knowledge, supplied via ``mapping`` (a dict of
    bearing-prefix -> region, e.g. ``{"east": "driveway", "west": "road"}``).
    Matching is case-insensitive on the direction's leading token (the text
    before any underscore), so "East_SouthEast" matches an "east" key.

    With no mapping, we fall back to a name heuristic (a direction literally
    naming "driveway"/"road"), and finally to None (unclassified), which
    routes to the safe novel-vehicle policy. ``bbox``/``frame`` are accepted
    for a future geometric override but not used yet.
    """
    d = (direction or "").lower()
    lead = d.split("_")[0].strip()

    if mapping:
        for key, region in mapping.items():
            if key.strip().lower() == lead:
                return region

    if "drive" in d or "driveway" in d:
        return "driveway"
    if "road" in d or "wichita" in d:
        return "road"
    return None


# --- wichita (motion camera) in-frame geometry ------------------------------
# The Wichita house camera is a single FIXED view (scene "WichitaDriveway"):
# the dirt road (Wichita Rd) runs across the UPPER part of the frame while the
# driveway/concrete pad fills the lower two-thirds. Region policy is therefore
# geometric (bbox position within the frame), not per-bearing. Values below
# are measured, not guessed: a YOLO probe over 14 significant frames
# (2026-09-13) puts road-band vehicles at bbox center-y ~0.36-0.40 (x roughly
# 0.39-0.52, the gap between the gate pillars) and the parked truck on the
# driveway pad at center-y ~0.44 (x 0.56-0.78). Left-edge detections
# (x < 0.30, center-y ~0.63) are the property's own side yard.
#
# NIGHT re-measure (t_f1b4349d, 2026-09-13): same detector (yolo11n, the
# live pipeline's weights, conf floor 0.26) over a uniform 1-in-10 sample of
# 14 days' night significant frames (2,230 frames, 2,335 detections). The
# band holds at night: the in-x-band road cohort (n=80) sits at cy
# 0.3695-0.385 (day prior: 0.378-0.381) — no IR optics shift; the parked
# truck stays at cy 0.4373-0.4455, cx 0.6425-0.6742 (off the x-band); ZERO
# night detections land inside the +/-0.02 near-band zone, and zero in
# x-band cy 0.42-0.46. Caveat carried forward: 0/80 night road detections
# clear the 0.75 night conf bar (19/80 clear the 0.45 day bar) — overnight
# road "silence" is a detector-threshold property, not evidence the band is
# empty; re-check after a week live (Percy P2-3 last bullet).
#
# road test: bbox CENTER inside the x-band AND above (smaller y than) the
# road's y cutoff. A NEAR-BAND zone straddles the cutoff: cy within
# ``road_edge`` (default 0.02) of y_max, inside the x-band, resolves to
# "road_edge" instead of a firm road/driveway — the caller feeds that to
# memory.decide(), which FLAGs road_edge vehicles for human review rather
# than betting the asymmetric miss (driveway->road = silent missed visitor
# alert; road->driveway = one noisy pass-by note). The measured cohorts sit
# outside the zone on both sides (pass-bys cy 0.378-0.381, >=0.039 above the
# cutoff; the parked truck cy 0.435-0.442 at cx ~0.67, outside the x-band),
# so normal traffic is unaffected. Everything else on the property ->
# "driveway" (the camera's motion field is the property; vehicles there get
# identity-expectedness).
ROAD_DEFAULT_Y_MAX = 0.42   # bbox center-y must be BELOW (nearer horizon)
ROAD_DEFAULT_X_MIN = 0.30   # bbox center-x band between the gate pillars
ROAD_DEFAULT_X_MAX = 0.52
ROAD_DEFAULT_EDGE = 0.02    # near-band margin around the y cutoff (P2-3)


def resolve_region_geom(bbox_frac: tuple[float, float, float, float],
                        road_y_max: float = ROAD_DEFAULT_Y_MAX,
                        road_x_min: float = ROAD_DEFAULT_X_MIN,
                        road_x_max: float = ROAD_DEFAULT_X_MAX,
                        road_edge: float = ROAD_DEFAULT_EDGE) -> str | None:
    """Classify a fixed-camera detection's region from its fractional bbox.

    ``bbox_frac`` is (x1, y1, x2, y2) in fractions of the frame (0-1). The
    road is a horizontal band: bbox center inside [x_min, x_max] horizontally
    and above ``road_y_max`` vertically. Detections whose center-y sits
    within ``road_edge`` of the cutoff — inside the x-band — resolve to
    "road_edge" (near-band: the caller FLAGs these for human review instead
    of trusting the edge). Everything else resolves to ``driveway`` (the
    fixed camera watches the property; a vehicle anywhere off the road band
    is on the property). Vehicles are the only category the caller routes by
    region, but the function is category-agnostic.

    Returns "road", "road_edge", "driveway", or None for a degenerate bbox
    (outside the frame or inverted) so a detector glitch can't silently
    misroute.
    """
    x1, y1, x2, y2 = bbox_frac
    if not all(0.0 <= v <= 1.0 for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        return None
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    if not (road_x_min <= cx <= road_x_max):
        return "driveway"
    if abs(cy - road_y_max) <= road_edge:
        # Straddles the cutoff (e.g. a pass-by shifted by night IR optics, or
        # a vehicle entering through the gate gap). Flag, don't guess.
        return "road_edge"
    return "road" if cy < road_y_max else "driveway"


def parse_region_mapping(spec: str) -> dict[str, str]:
    """Parse a ``bearing=region,bearing=region`` string into a dict.

    Empty/blank entries are skipped; the region is normalized to one of
    "driveway"/"road"/None (unknown regions are dropped so a typo can't
    silently misroute a detection to a bogus policy).
    """
    out: dict[str, str] = {}
    if not spec:
        return out
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            continue
        key, _, region = part.partition("=")
        key = key.strip().lower()
        region = region.strip().lower()
        if not key or region not in ("driveway", "road"):
            continue
        out[key] = region
    return out


def is_night(created_dt: str | None, now_wall: datetime | None = None,
             day_start_hour: int = DAY_START_HOUR,
             day_end_hour: int = DAY_END_HOUR) -> bool:
    """True when a capture timestamp falls in the configured dark hours.

    ``created_dt`` is the API's ``createdDateTime`` (naive UTC, or offset/Z
    suffixed). We convert to US/Mountain wall time and compare the hour against
    the day window (``day_start_hour``..``day_end_hour``, configurable per
    settled decision 4), matching ``dashboard._lighting``. ``now_wall`` lets a
    test pin the wall clock.
    """
    if now_wall is not None:
        return not (day_start_hour <= now_wall.hour < day_end_hour)
    if not created_dt:
        return False
    dt = datetime.fromisoformat(created_dt)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=pytz.UTC)
    wall = dt.astimezone(pytz.timezone(CAMERA_TZ))
    return not (day_start_hour <= wall.hour < day_end_hour)
