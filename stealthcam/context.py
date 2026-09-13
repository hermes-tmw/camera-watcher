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
