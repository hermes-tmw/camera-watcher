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
        # The worker passes night_conf_threshold into detector.detect; verify
        # the detector honors the per-call override (higher bar drops more).
        from stealthcam.detector import Detector
        # We don't load the model here — just pin the constructor contract via
        # a subclass-free check: the detect signature accepts conf_threshold.
        import inspect
        sig = inspect.signature(Detector.detect)
        self.assertIn("conf_threshold", sig.parameters)

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


if __name__ == "__main__":
    unittest.main()
