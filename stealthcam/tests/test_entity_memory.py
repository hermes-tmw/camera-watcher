"""Tests for entity identity + expected-presence baseline + location policy.

Covers the camera-watcher entity-memory work (settled decisions 1-4):

1. Identity: two detections of the SAME vehicle (different captures, same
   direction, same fingerprint) resolve to one entity — persistence survives
   a leave-and-return without re-alerting.
2. Expected-presence baseline: an entity present in >= the threshold fraction
   of monitored captures is "typically here" -> background note (no alert).
3. Location policy: a vehicle on the `road` region is a pass-by -> note tagged
   `traffic`, no alert. A vehicle in `driveway` gets identity-expectedness
   evaluation (unfamiliar -> alert).
4. Night gate: the detector raises its confidence bar at night, so
   sub-threshold detections are dropped.
5. The false-negative case: a visitor parked >1h (would have been "static"
   under the old model) is NOT wrongly suppressed — identity still flags it.

These tests use a trivial deterministic Fingerprinter (no OpenCV dependency) so
they run anywhere; the OpenCV backend has its own smoke test.
"""
from __future__ import annotations

import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from stealthcam.describer import VehicleAttributes
from stealthcam.memory import Memory, ALERT, NOTE, FLAG, TAG_TRAFFIC, TAG_BACKGROUND


def _iso(dt: datetime) -> str:
    return dt.isoformat()


class _TrivialFingerprinter:
    """Deterministic fingerprint: identity is just the input bytes, so the
    tests control entity matching directly without OpenCV."""

    model = "trivial"

    def fingerprint(self, crop):
        # The crop is a PIL Image; use its size + a caller-supplied tag via
        # ``crop.info`` if present, else a constant. Tests pass raw bytes
        # through ``_fp`` below instead of using this method.
        return b"x"

    def matches(self, a: bytes, b: bytes) -> bool:
        return a == b


def _fp(tag: str) -> bytes:
    """A stable fingerprint for entity `tag` (identity = the tag)."""
    return tag.encode()


def _attrs(make="", model="", lettering="", uncertain=True) -> VehicleAttributes:
    return VehicleAttributes(make=make, model=model, lettering=lettering,
                             uncertain=uncertain)


def _mk_memory():
    return Memory(":memory:", fingerprinter=_TrivialFingerprinter(),
                  entity_expected_presence_ratio=0.5,
                  entity_baseline_window_days=7.0)


class TestEntityIdentity(unittest.TestCase):
    def test_same_vehicle_across_captures_resolves_to_one_entity(self):
        m = _mk_memory()
        t0 = datetime(2026, 9, 10, 12, 0, 0, tzinfo=timezone.utc)
        bbox = (100.0, 100.0, 300.0, 300.0)

        # First sighting (unfamiliar -> alert).
        d1 = m.decide("East", "car", bbox, now=_iso(t0), fingerprint=_fp("my-truck"),
                      region="driveway", attributes=_attrs(make="Ford", model="F-250"))
        self.assertEqual(d1.decision, ALERT)
        self.assertIsNotNone(d1.entity_id)

        # Leave and return: a later capture, same fingerprint -> same entity.
        t1 = t0 + timedelta(hours=6)
        d2 = m.decide("East", "truck", bbox, now=_iso(t1), fingerprint=_fp("my-truck"),
                      region="driveway", attributes=_attrs(make="Ford", model="F-250"))
        # Still unfamiliar until the presence baseline crosses the threshold.
        self.assertEqual(d2.decision, ALERT)
        self.assertEqual(d2.entity_id, d1.entity_id)
        m.close()

    def test_different_vehicle_same_spot_alerts(self):
        m = _mk_memory()
        t0 = datetime(2026, 9, 10, 12, 0, 0, tzinfo=timezone.utc)
        bbox = (100.0, 100.0, 300.0, 300.0)

        m.decide("East", "car", bbox, now=_iso(t0), fingerprint=_fp("my-truck"),
                 region="driveway", attributes=_attrs(make="Ford"))
        # A DIFFERENT vehicle (different fingerprint) in the SAME spot -> alert.
        d = m.decide("East", "car", bbox, now=_iso(t0), fingerprint=_fp("stranger"),
                     region="driveway", attributes=_attrs(make="Honda"))
        self.assertEqual(d.decision, ALERT)
        self.assertNotEqual(d.entity_id, "my-truck")
        m.close()

    def test_expected_presence_suppresses_alert(self):
        m = _mk_memory()
        bbox = (100.0, 100.0, 300.0, 300.0)

        # Seed: my truck is present in 8 of 10 monitored captures over the
        # window (>= 50% -> expected). Each sighting is a DISTINCT capture guid.
        t0 = datetime(2026, 9, 10, 12, 0, 0, tzinfo=timezone.utc)
        eid = None
        for k in range(8):
            t = t0 + timedelta(hours=k)
            g = str(uuid.uuid4())
            d = m.decide("East", "car", bbox, now=_iso(t), fingerprint=_fp("my-truck"),
                         region="driveway", guid=g)
            eid = d.entity_id
        # Record 10 monitored captures total (the presence denominator).
        for k in range(10):
            t = t0 + timedelta(hours=k)
            m.record_monitored("East", "driveway", str(uuid.uuid4()), _iso(t))

        # Now a fresh sighting: 8/10 = 0.8 >= 0.5 -> expected -> note, no alert.
        d = m.decide("East", "car", bbox, now=_iso(t0 + timedelta(hours=10)),
                     fingerprint=_fp("my-truck"), region="driveway",
                     guid=str(uuid.uuid4()))
        self.assertEqual(d.decision, NOTE)
        self.assertEqual(d.tag, TAG_BACKGROUND)
        self.assertEqual(d.entity_id, eid)
        m.close()

    def test_visitor_parked_over_hour_not_suppressed(self):
        """The false-negative case today: a visitor parked >1h would become
        'static' and be wrongly backgrounded. Identity must still flag it."""
        m = _mk_memory()
        bbox = (100.0, 100.0, 300.0, 300.0)
        guid = str(uuid.uuid4())
        t0 = datetime(2026, 9, 10, 12, 0, 0, tzinfo=timezone.utc)

        # A stranger parks in the driveway and stays >1h (many captures, long
        # span -> would be "static" by position alone).
        for k in range(6):
            t = t0 + timedelta(hours=k * 0.5)  # 3h span, 6 captures
            d = m.decide("East", "car", bbox, now=_iso(t), fingerprint=_fp("stranger"),
                         region="driveway", guid=guid)
            # Every sighting is an alert (unfamiliar), never suppressed despite
            # the long span + high capture count.
            self.assertEqual(d.decision, ALERT, f"capture {k} should alert")
        m.close()


class TestLocationPolicy(unittest.TestCase):
    def test_road_passby_is_note_no_alert(self):
        m = _mk_memory()
        bbox = (100.0, 100.0, 300.0, 300.0)
        t0 = datetime(2026, 9, 10, 12, 0, 0, tzinfo=timezone.utc)
        d = m.decide("West", "car", bbox, now=_iso(t0), fingerprint=_fp("any"),
                     region="road", attributes=_attrs(make="Freightliner", lettering="FedEx"))
        self.assertEqual(d.decision, NOTE)
        self.assertEqual(d.tag, TAG_TRAFFIC)
        self.assertIn("FedEx", d.attributes_text)
        m.close()

    def test_driveway_unfamiliar_alerts_with_attributes(self):
        m = _mk_memory()
        bbox = (100.0, 100.0, 300.0, 300.0)
        t0 = datetime(2026, 9, 10, 12, 0, 0, tzinfo=timezone.utc)
        d = m.decide("East", "car", bbox, now=_iso(t0), fingerprint=_fp("stranger"),
                     region="driveway",
                     attributes=_attrs(make="Chevrolet", model="Tahoe", uncertain=False))
        self.assertEqual(d.decision, ALERT)
        self.assertIn("Chevrolet", d.attributes_text)
        m.close()

    def test_unclassified_region_uses_novel_vehicle_policy(self):
        m = _mk_memory()
        bbox = (100.0, 100.0, 300.0, 300.0)
        t0 = datetime(2026, 9, 10, 12, 0, 0, tzinfo=timezone.utc)
        # No region -> current novel-vehicle policy: alert.
        d = m.decide("North", "car", bbox, now=_iso(t0), fingerprint=_fp("x"),
                     region=None)
        self.assertEqual(d.decision, ALERT)
        m.close()

    def test_non_vehicle_behavior_unchanged(self):
        m = _mk_memory()
        bbox = (100.0, 100.0, 300.0, 300.0)
        t0 = datetime(2026, 9, 10, 12, 0, 0, tzinfo=timezone.utc)
        # person -> alert, animal -> flag, regardless of region/fingerprint.
        self.assertEqual(m.decide("East", "person", bbox, now=_iso(t0)).decision, ALERT)
        self.assertEqual(m.decide("East", "sheep", bbox, now=_iso(t0)).decision, FLAG)
        m.close()


class TestNightGate(unittest.TestCase):
    def test_night_uses_higher_threshold(self):
        # Behavior, not reflection: drive the real process_capture with a
        # recording detector stub and assert the NIGHT bar (0.75) actually
        # reaches detect() for a night capture, and the DAY bar (0.6) for a day
        # capture.
        from stealthcam.worker import process_capture
        from stealthcam.memory import Memory

        class _RecDetector:
            conf_threshold = 0.6
            def __init__(self):
                self.seen_bars = []
            def detect(self, image, conf_threshold=None):
                self.seen_bars.append(conf_threshold)
                return []

        class _Client:
            def download_subimage(self, guid, n, dest):
                from PIL import Image
                Image.new("RGB", (8, 8)).save(dest, format="JPEG")

        def run(created):
            det = _RecDetector()
            mem = Memory(":memory:")
            capture = {"imageGuid": str(uuid.uuid4()),
                       "imageDirections": ["East"] * 6,
                       "createdDateTime": created}
            process_capture(capture, _Client(), det, None, mem, None, None,
                            Path("/tmp"), baseline_mode=True)
            mem.close()
            return det.seen_bars

        # Day: 16:00 UTC == 10:00 MDT (day) -> day bar 0.6.
        day_bars = run("2026-09-08T16:00:00")
        self.assertTrue(day_bars, "expected at least one detect() call")
        self.assertTrue(all(b == 0.6 for b in day_bars),
                        f"day captures should use the day bar, got {day_bars}")

        # Night: 04:00 UTC == 22:00 MDT (previous day) -> night bar 0.75.
        night_bars = run("2026-09-08T04:00:00")
        self.assertTrue(night_bars, "expected at least one detect() call")
        self.assertTrue(all(b == 0.75 for b in night_bars),
                        f"night captures should use the night bar, got {night_bars}")

    def test_is_night_boundaries(self):
        from stealthcam.context import is_night
        # 10:00 MDT (daylight) -> not night; 22:00 MDT -> night.
        # createdDateTime is naive UTC; 16:00 UTC == 10:00 MDT (day).
        self.assertFalse(is_night("2026-09-08T16:00:00"))
        # 04:00 UTC == 22:00 MDT (previous day) -> night.
        self.assertTrue(is_night("2026-09-08T04:00:00"))


class TestRegionMapping(unittest.TestCase):
    def test_mapping_by_leading_token(self):
        from stealthcam.context import resolve_region, parse_region_mapping
        mapping = parse_region_mapping("east=driveway,west=road")
        # "East_SouthEast" matches the "east" key (leading token).
        self.assertEqual(resolve_region("East_SouthEast", mapping=mapping), "driveway")
        self.assertEqual(resolve_region("West", mapping=mapping), "road")

    def test_parse_ignores_bogus(self):
        from stealthcam.context import parse_region_mapping
        out = parse_region_mapping("east=driveway, west=garage, =road, bad")
        self.assertEqual(out, {"east": "driveway"})

    def test_no_mapping_unclassified(self):
        from stealthcam.context import resolve_region
        # A compass bearing with no mapping and no name hint -> None.
        self.assertIsNone(resolve_region("North_NorthEast"))


class TestOpenCVFingerprinter(unittest.TestCase):
    def test_smoke_fingerprint_and_match(self):
        from PIL import Image
        from stealthcam.reid import OpenCVFingerprinter
        fp = OpenCVFingerprinter()
        # Two identical crops -> identical fingerprint -> match.
        img = Image.new("RGB", (64, 64), (200, 40, 40))
        a = fp.fingerprint(img)
        b = fp.fingerprint(img)
        self.assertIsNotNone(a)
        self.assertTrue(fp.matches(a, b))
        # A very different crop -> no match.
        other = Image.new("RGB", (64, 64), (20, 20, 200))
        c = fp.fingerprint(other)
        self.assertFalse(fp.matches(a, c))

    def test_discriminative_different_vehicles_do_not_match(self):
        # P0-3 regression: the v1 descriptor matched DIFFERENT vehicles
        # (cab-reversed truck 0.906, truck-vs-sedan 0.927) at the shipped 0.85
        # threshold. The v2 descriptor + 0.80 threshold must separate
        # same-vehicle (scale/aspect variation) from different-vehicle.
        from PIL import Image, ImageDraw
        from stealthcam.reid import OpenCVFingerprinter

        def truck(w=120, h=70, body=(60, 60, 60), cab=(80, 80, 80), flip=False):
            img = Image.new("RGB", (w, h), (25, 25, 25))
            d = ImageDraw.Draw(img)
            if not flip:
                d.rectangle([5, int(h * 0.35), int(w * 0.62), int(h * 0.8)], fill=body)
                d.rectangle([int(w * 0.62), int(h * 0.15), w - 8, int(h * 0.8)], fill=cab)
            else:
                d.rectangle([int(w * 0.38), int(h * 0.35), w - 5, int(h * 0.8)], fill=body)
                d.rectangle([8, int(h * 0.15), int(w * 0.38), int(h * 0.8)], fill=cab)
            return img

        fp = OpenCVFingerprinter()
        mine = truck()
        mine_s = truck(w=100, h=58)          # same truck, rescaled
        lookalike = truck(flip=True)          # DIFFERENT truck (cab reversed)
        sedan = Image.new("RGB", (120, 70), (25, 25, 25))
        dc = ImageDraw.Draw(sedan)
        dc.rectangle([12, 18, 108, 55], fill=(70, 70, 70))

        f_mine, f_mine_s, f_alike, f_sedan = (fp.fingerprint(x)
                                              for x in (mine, mine_s, lookalike, sedan))

        # SAME vehicle at different scale/aspect MUST match at the shipped bar.
        self.assertTrue(fp.matches(f_mine, f_mine_s),
                        "same vehicle rescaled must match")
        # DIFFERENT vehicles MUST NOT match.
        self.assertFalse(fp.matches(f_mine, f_alike),
                         "cab-reversed truck must NOT match")
        self.assertFalse(fp.matches(f_mine, f_sedan),
                         "truck vs sedan must NOT match")

    def test_cross_direction_identical_fingerprint_no_collision(self):
        # P0-1 regression: a byte-identical fingerprint in a second direction
        # used to raise sqlite3.IntegrityError out of decide(). Direction
        # salting must keep the two sightings as distinct entities.
        m = _mk_memory()
        t0 = datetime(2026, 9, 10, 12, 0, 0, tzinfo=timezone.utc)
        bbox = (100.0, 100.0, 300.0, 300.0)
        d1 = m.decide("East", "car", bbox, now=_iso(t0), fingerprint=b"same-bytes",
                      region="driveway")
        d2 = m.decide("West", "car", bbox, now=_iso(t0), fingerprint=b"same-bytes",
                      region="driveway")
        self.assertIsNotNone(d1.entity_id)
        self.assertIsNotNone(d2.entity_id)
        self.assertNotEqual(d1.entity_id, d2.entity_id,
                            "cross-direction identical fingerprint must not collide")
        m.close()


class TestExpectedHysteresis(unittest.TestCase):
    def test_expected_stays_expected_across_detection_gap(self):
        # P1-1: once expected, an entity stays expected (no alert) across a
        # short detection gap even if the presence ratio would dip below 0.5.
        m = _mk_memory()
        bbox = (100.0, 100.0, 300.0, 300.0)
        t0 = datetime(2026, 9, 10, 12, 0, 0, tzinfo=timezone.utc)
        eid = None
        # Seed: present 8 of 10 monitored captures -> expected.
        for k in range(8):
            t = t0 + timedelta(hours=k)
            d = m.decide("East", "car", bbox, now=_iso(t), fingerprint=_fp("truck"),
                         region="driveway", guid=str(uuid.uuid4()))
            eid = d.entity_id
        for k in range(10):
            t = t0 + timedelta(hours=k)
            m.record_monitored("East", "driveway", str(uuid.uuid4()), _iso(t))
        d = m.decide("East", "car", bbox, now=_iso(t0 + timedelta(hours=10)),
                     fingerprint=_fp("truck"), region="driveway",
                     guid=str(uuid.uuid4()))
        self.assertEqual(d.decision, NOTE)  # now expected

        # A short gap (e.g. 1 day, under the 2-day absence threshold) — the
        # presence ratio would now be 8/11 < 0.5, but hysteresis keeps it
        # expected.
        d2 = m.decide("East", "car", bbox, now=_iso(t0 + timedelta(days=1, hours=12)),
                      fingerprint=_fp("truck"), region="driveway",
                      guid=str(uuid.uuid4()))
        self.assertEqual(d2.decision, NOTE)
        self.assertEqual(d2.entity_id, eid)
        m.close()

    def test_expected_reverts_after_long_absence(self):
        # Use a short baseline window so the presence history ages out during
        # the absence, dropping the ratio below threshold -> the expected flag
        # lapses (absent >= 2 days) AND the ratio check alerts.
        m = Memory(":memory:", fingerprinter=_TrivialFingerprinter(),
                   entity_expected_presence_ratio=0.5,
                   entity_baseline_window_days=1.0)
        bbox = (100.0, 100.0, 300.0, 300.0)
        t0 = datetime(2026, 9, 10, 12, 0, 0, tzinfo=timezone.utc)
        for k in range(8):
            t = t0 + timedelta(hours=k)
            m.decide("East", "car", bbox, now=_iso(t), fingerprint=_fp("truck"),
                     region="driveway", guid=str(uuid.uuid4()))
        for k in range(10):
            t = t0 + timedelta(hours=k)
            m.record_monitored("East", "driveway", str(uuid.uuid4()), _iso(t))
        d = m.decide("East", "car", bbox, now=_iso(t0 + timedelta(hours=10)),
                     fingerprint=_fp("truck"), region="driveway",
                     guid=str(uuid.uuid4()))
        self.assertEqual(d.decision, NOTE)  # expected at 8/10

        # Absent >= 2 days -> expected lapses; the 1-day window has rolled past
        # every presence/monitored row -> ratio 0.0 -> alert.
        d = m.decide("East", "car", bbox, now=_iso(t0 + timedelta(days=4)),
                     fingerprint=_fp("truck"), region="driveway",
                     guid=str(uuid.uuid4()))
        self.assertEqual(d.decision, ALERT)
        m.close()


if __name__ == "__main__":
    unittest.main()
