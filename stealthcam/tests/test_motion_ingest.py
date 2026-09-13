"""Tests for the wichita motion-ingestor adapter + fixed-camera geometry.

Covers the card's four test deliverables:
1. Scope isolation — the same motion event processed under both scopes stays
   independent (seen-set, decisions, entities); wichita rows never leak into
   taylor_hill and vice versa.
2. Pass-by / no-alert — a road-band vehicle produces a NOTE tagged `traffic`
   and NEVER calls the Mattermost notifier.
3. Driveway expected-suppression — after baseline seeding, the resident
   vehicle's sighting is a background NOTE, not an alert.
4. Ingestor correctness — geometric region resolution, source-query contract,
   frame fallback, WAL multi-writer readiness, dashboard labeling write path.
"""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from stealthcam.context import (resolve_region_geom, ROAD_DEFAULT_Y_MAX)
from stealthcam.memory import Memory, NOTE


def _iso(dt: datetime) -> str:
    return dt.isoformat()


_T0 = datetime(2026, 9, 13, 15, 0, 0, tzinfo=timezone.utc)


class _TrivialFingerprinter:
    model = "trivial"

    def __init__(self, key: bytes = b"veh"):
        self.key = key

    def fingerprint(self, crop):
        return self.key

    def matches(self, a: bytes, b: bytes) -> bool:
        return a == b


def _mk(db: str, scope: str, fp=None) -> Memory:
    return Memory(db, fingerprinter=fp or _TrivialFingerprinter(),
                  entity_expected_presence_ratio=0.5,
                  entity_baseline_window_days=7.0,
                  scope=scope)


# -- 1. geometry --------------------------------------------------------------


class TestResolveRegionGeom(unittest.TestCase):
    def test_road_band_vehicle(self):
        # measured pass-by on Wichita Rd: bbox center (0.42, 0.38)
        self.assertEqual(
            resolve_region_geom((0.39, 0.359, 0.451, 0.398)), "road")

    def test_parked_truck_is_driveway(self):
        # measured parked-truck bbox x[0.56-0.78] y[0.37-0.52], cy=0.44
        self.assertEqual(
            resolve_region_geom((0.556, 0.368, 0.781, 0.515)), "driveway")

    def test_left_edge_property_vehicle_is_driveway(self):
        self.assertEqual(resolve_region_geom((0.0, 0.42, 0.09, 0.84)),
                         "driveway")

    def test_center_y_at_cutoff_is_driveway(self):
        # cy == road_y_max is NOT road (strictly-above rule)
        cy = ROAD_DEFAULT_Y_MAX
        self.assertEqual(
            resolve_region_geom((0.35, cy - 0.02, 0.45, cy + 0.02)),
            "driveway")

    def test_x_band_edges(self):
        # inside the band horizontally but low in the frame -> driveway
        self.assertEqual(resolve_region_geom((0.38, 0.6, 0.45, 0.8)),
                         "driveway")
        # outside the x band entirely -> driveway even when high
        self.assertEqual(resolve_region_geom((0.10, 0.30, 0.20, 0.40)),
                         "driveway")

    def test_degenerate_bbox_is_none(self):
        self.assertIsNone(resolve_region_geom((0.5, 0.5, 0.5, 0.5)))
        self.assertIsNone(resolve_region_geom((0.6, 0.5, 0.4, 0.6)))
        self.assertIsNone(resolve_region_geom((-0.1, 0.2, 0.3, 0.4)))
        self.assertIsNone(resolve_region_geom((0.1, 0.2, 1.3, 0.4)))

    def test_custom_band_overrides(self):
        self.assertEqual(
            resolve_region_geom((0.7, 0.30, 0.8, 0.35),
                                road_y_max=0.42, road_x_min=0.6,
                                road_x_max=0.9),
            "road")


# -- 2. engine policy on the wichita scope ------------------------------------


class TestWichitaEnginePolicy(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "pipeline.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_pass_by_note_no_alert(self):
        m = _mk(self.db, "wichita")
        d = m.decide("WichitaDriveway", "car", (390.0, 250.0, 500.0, 290.0),
                     now=_iso(_T0), fingerprint=b"passby", region="road",
                     guid="ev-1")
        self.assertEqual(d.decision, "note")
        self.assertEqual(d.tag, "traffic")
        self.assertFalse(d.is_alert)
        m.close()

    def test_driveway_first_sight_alerts(self):
        m = _mk(self.db, "wichita")
        d = m.decide("WichitaDriveway", "car", (550.0, 260.0, 780.0, 380.0),
                     now=_iso(_T0), fingerprint=b"stranger",
                     region="driveway", guid="ev-2")
        self.assertEqual(d.decision, "alert")
        m.close()

    def test_driveway_expected_suppresses_to_note(self):
        m = _mk(self.db, "wichita")
        for i in range(10):
            now = _iso(_T0 + timedelta(hours=i))
            m.record_monitored("WichitaDriveway", "driveway", f"ev-{i}",
                               now, is_day=True)
            if i < 8:
                m.decide("WichitaDriveway", "car",
                         (550.0, 260.0, 780.0, 380.0), now=now,
                         fingerprint=b"resident", region="driveway",
                         guid=f"ev-{i}")
        d = m.decide("WichitaDriveway", "car", (550.0, 260.0, 780.0, 380.0),
                     now=_iso(_T0), fingerprint=b"resident",
                     region="driveway", guid="ev-live")
        self.assertEqual(d.decision, "note")
        self.assertEqual(d.tag, "background")
        m.close()

    def test_no_fingerprint_driveway_is_conservative_alert(self):
        m = _mk(self.db, "wichita")
        d = m.decide("WichitaDriveway", "car", (550.0, 260.0, 780.0, 380.0),
                     now=_iso(_T0), fingerprint=None, region="driveway",
                     guid="ev-3")
        self.assertEqual(d.decision, "alert")
        m.close()


# -- 3. scope isolation at the adapter seam -----------------------------------


class TestAdapterScopeIsolation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "pipeline.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_same_event_name_two_scopes_independent(self):
        wi = _mk(self.db, "wichita")
        th = _mk(self.db, "taylor_hill")
        for m in (wi, th):
            m.decide("WichitaDriveway", "car", (100.0, 100.0, 300.0, 300.0),
                     now=_iso(_T0), fingerprint=b"same-bytes",
                     region="driveway", guid="20260913_shared_event")
            m.mark_seen("20260913_shared_event")
            m.close()
        wi2 = _mk(self.db, "wichita")
        th2 = _mk(self.db, "taylor_hill")
        self.assertTrue(wi2.has_seen("20260913_shared_event"))
        self.assertTrue(th2.has_seen("20260913_shared_event"))
        self.assertEqual(wi2.count_entities(), 1)
        self.assertEqual(th2.count_entities(), 1)
        wi2.close()
        th2.close()
        rows = sqlite3.connect(self.db).execute(
            "SELECT scope, entity_id FROM entities ORDER BY scope").fetchall()
        self.assertEqual({s for s, _ in rows}, {"taylor_hill", "wichita"})
        # scope-salted entity ids: identical fingerprint bytes -> distinct ids
        self.assertNotEqual(rows[0][1], rows[1][1])

    def test_wichita_rows_never_written_to_taylor_scope(self):
        wi = _mk(self.db, "wichita")
        wi.decide("WichitaDriveway", "truck", (10.0, 10.0, 200.0, 200.0),
                  now=_iso(_T0), fingerprint=b"wichita-only", region="road",
                  guid="ev-w1")
        wi.record_monitored("WichitaDriveway", "road", "ev-w1",
                            _iso(_T0), is_day=True)
        wi.close()
        conn = sqlite3.connect(self.db)
        for table in ("decisions", "entities", "entity_presence",
                      "monitored_captures", "seen_guids",
                      "memory_observations"):
            n = conn.execute(
                f"SELECT COUNT(*) FROM {table} "
                f"WHERE scope = 'taylor_hill'").fetchone()[0]
            self.assertEqual(n, 0, f"{table} leaked into taylor_hill")
        conn.close()


# -- 4. WAL multi-writer readiness ---------------------------------------------


class TestWalMultiWriter(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "pipeline.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_memory_opens_in_wal_mode(self):
        m = _mk(self.db, "wichita")
        mode = m._conn.execute("PRAGMA journal_mode").fetchone()[0]
        self.assertEqual(str(mode).lower(), "wal")
        m.close()

    def test_wal_persists_for_second_scope_writer(self):
        m1 = _mk(self.db, "wichita")
        m1.decide("WichitaDriveway", "car", (1.0, 1.0, 2.0, 2.0),
                  now=_iso(_T0), fingerprint=b"a", region="driveway",
                  guid="ev-1")
        d = m1.decide("WichitaDriveway", "car", (1.0, 1.0, 2.0, 2.0),
                      now=_iso(_T0), fingerprint=b"a", region="driveway",
                      guid="ev-1b")
        m1.record_decision("ev-1b", 0, "WichitaDriveway", "car", 0.9,
                           (1.0, 1.0, 2.0, 2.0), "", d)
        m1.close()
        m2 = _mk(self.db, "taylor_hill")
        mode = m2._conn.execute("PRAGMA journal_mode").fetchone()[0]
        self.assertEqual(str(mode).lower(), "wal")
        m2.decide("East", "car", (1.0, 1.0, 2.0, 2.0), now=_iso(_T0),
                  fingerprint=b"a", region="driveway", guid="ev-1")
        d2 = m2.decide("East", "car", (1.0, 1.0, 2.0, 2.0), now=_iso(_T0),
                       fingerprint=b"a", region="driveway", guid="ev-1b")
        m2.record_decision("ev-1b", 0, "East", "car", 0.9,
                           (1.0, 1.0, 2.0, 2.0), "", d2)
        m2.close()
        conn = sqlite3.connect(self.db)
        self.assertEqual(conn.execute(
            "SELECT COUNT(*) FROM decisions WHERE scope='wichita'"
        ).fetchone()[0], 1)
        self.assertEqual(conn.execute(
            "SELECT COUNT(*) FROM decisions WHERE scope='taylor_hill'"
        ).fetchone()[0], 1)
        conn.close()

    def test_shared_db_concurrent_interleaved_writes(self):
        # two Memory handles OPEN SIMULTANEOUSLY (the live deploy shape:
        # taylor worker + wichita ingestor both connected) interleave writes
        # without lock errors.
        m_wi = _mk(self.db, "wichita")
        m_th = _mk(self.db, "taylor_hill")
        for i in range(6):
            m_wi.decide("WichitaDriveway", "car", (10.0, 10.0, 50.0, 50.0),
                        now=_iso(_T0), fingerprint=b"wi", region="driveway",
                        guid=f"wi-{i}")
            m_th.decide("East", "car", (10.0, 10.0, 50.0, 50.0),
                        now=_iso(_T0), fingerprint=b"th", region="driveway",
                        guid=f"th-{i}")
        self.assertEqual(m_wi.count_entities(), 1)
        self.assertEqual(m_th.count_entities(), 1)
        m_wi.close()
        m_th.close()


# -- 5. source query contract --------------------------------------------------


class TestMotionEventSource(unittest.TestCase):
    def _make(self, rows):
        import stealthcam.motion_ingest as mi
        captured = {}

        class _Cur:
            description = []

            def execute(self, sql, params=None):
                captured["sql"] = sql
                captured["params"] = params

            def fetchall(self):
                return rows

            def fetchone(self):
                return (len(rows),)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        class _Conn:
            autocommit = False

            def cursor(self, cursor_factory=None):
                return _Cur()

            def close(self):
                pass

        conn = _Conn()
        with patch.object(mi.psycopg2, "connect", return_value=conn):
            src = mi.MotionEventSource("postgresql://fake",
                                       scenes=["WichitaDriveway"],
                                       lookback_hours=12)
        return src, captured

    def test_poll_requires_scenes(self):
        import stealthcam.motion_ingest as mi
        with self.assertRaises(ValueError):
            mi.MotionEventSource("postgresql://x", scenes=[],
                                 lookback_hours=12)

    def test_poll_requires_db_url(self):
        import stealthcam.motion_ingest as mi
        with self.assertRaises(ValueError):
            mi.MotionEventSource("", scenes=["WichitaDriveway"],
                                 lookback_hours=12)

    def test_poll_query_contract(self):
        ct = datetime(2026, 9, 13, 12, 0, 0)
        rows = [{
            "id": 42, "event_name": "ev42", "capture_time": ct,
            "scene_name": "WichitaDriveway",
            "video_location": "wichitaDriveway/2026/09/13",
            "frame_file": "wichitaDriveway/2026/09/13/ev42_f3.jpg",
        }]
        src, captured = self._make(rows)
        events = src.poll(limit=200)
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev.event_id, 42)
        self.assertEqual(ev.event_name, "ev42")
        self.assertEqual(ev.capture_time.tzinfo, timezone.utc)
        self.assertEqual(ev.scene_name, "WichitaDriveway")
        self.assertEqual(ev.frame_relpath,
                         "wichitaDriveway/2026/09/13/ev42_f3.jpg")
        sql = captured["sql"]
        self.assertIn("task_save_significant_frame", sql)
        self.assertIn("e.camera = ANY", sql)
        self.assertIn("e.storage_local", sql)
        self.assertIn("decider LIKE 'ollama:%%'", sql)
        self.assertIn("ORDER BY e.capture_time ASC", sql)
        self.assertEqual(captured["params"]["scenes"], ["WichitaDriveway"])
        self.assertEqual(captured["params"]["limit"], 200)
        src.close()


# -- 6. process_event end-to-end (fakes for heavy deps) ------------------------


def _mk_detection(cls: str, conf: float, bbox: tuple):
    from stealthcam.detector import (PERSON_CLASSES, VEHICLE_CLASSES,
                                     ANIMAL_CLASSES)
    if cls in PERSON_CLASSES:
        cat = "person"
    elif cls in VEHICLE_CLASSES:
        cat = "vehicle"
    elif cls in ANIMAL_CLASSES:
        cat = "animal"
    else:
        cat = "other"

    class _Det:
        pass

    d = _Det()
    d.cls = cls
    d.confidence = conf
    d.bbox = bbox
    d.category = cat
    return d


class _FixedDetector:
    def __init__(self, detections):
        self.detections = detections

    def detect(self, image, conf_threshold=None):
        return self.detections


class _FakeNotifier:
    def __init__(self):
        self.alerts = []

    def post_alert(self, text, image_path=None):
        self.alerts.append((text, image_path))
        return "post-id"


class _FakeDashboard:
    def __init__(self):
        self.notes = []

    def write_event_labeling(self, event_id, category, confidence, text):
        self.notes.append((event_id, category, confidence, text))
        return len(self.notes)


class _FakeDescriber:
    def describe_vehicle(self, crop):
        from stealthcam.describer import VehicleAttributes
        return VehicleAttributes(make="Ford", model="F-150",
                                 lettering="", uncertain=False)

    def describe(self, crop):
        return "thing"


def _write_frame(path: Path, w: int = 1280, h: int = 720) -> None:
    from PIL import Image
    img = Image.new("RGB", (w, h), (120, 120, 120))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="JPEG")


class TestProcessEventRunner(unittest.TestCase):
    """process_event with the real control flow; detector/describer/notifier/
    dashboard are fakes (the mock boundary — heavy ML + network are mocked,
    the routing/decision/persistence logic is real)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "pipeline.db")
        _write_frame(Path(self.tmp.name) / "frames" / "ev1_f3.jpg")
        self.notifier = _FakeNotifier()
        self.dashboard = _FakeDashboard()

    def tearDown(self):
        self.tmp.cleanup()

    def _event(self, name="20260913_150000_WichitaDriveway_1"):
        from stealthcam.motion_ingest import MotionEvent
        return MotionEvent(1, name, _T0, "WichitaDriveway",
                           "wichitaDriveway/2026/09/13",
                           "frames/ev1_f3.jpg")

    def _run(self, detections, event=None, baseline=False, scope="wichita",
             describer=None):
        import stealthcam.motion_ingest as mi
        m = _mk(self.db, scope)
        s = mi.process_event(event or self._event(),
                             _FixedDetector(detections), describer, m,
                             self.notifier, self.dashboard,
                             Path(self.tmp.name) / "sub",
                             baseline_mode=baseline,
                             local_data_dir=self.tmp.name)
        return m, s

    # -- deliverable 2: pass-by -> note, never alert -------------------------

    def test_road_pass_by_note_not_alert(self):
        # mirrors the measured road-band vehicle: frac (0.39,0.36-0.45,0.40)
        # on the 1280x720 frame -> bbox (500, 259, 576, 288)
        road_car = _mk_detection("car", 0.72, (500.0, 259.0, 576.0, 288.0))
        m, s = self._run([road_car], describer=_FakeDescriber())
        self.assertEqual(s["alerts"], 0)
        self.assertEqual(s["notes"], 1)
        self.assertEqual(self.notifier.alerts, [])  # NO Mattermost post
        self.assertEqual(len(self.dashboard.notes), 1)
        event_id, category, conf, text = self.dashboard.notes[0]
        self.assertIn("[traffic]", text)
        self.assertIn("pass-by", text)
        self.assertEqual(category, "vehicle")
        m.close()

    def test_pass_by_captures_make_model(self):
        road_car = _mk_detection("car", 0.72, (500.0, 259.0, 576.0, 288.0))
        m, s = self._run([road_car], describer=_FakeDescriber())
        self.assertIn("Ford F-150", self.dashboard.notes[0][3])
        m.close()

    # -- driveway policies ---------------------------------------------------

    def test_driveway_unfamiliar_alert(self):
        truck = _mk_detection("truck", 0.8, (712.0, 266.0, 1000.0, 374.0))
        m, s = self._run([truck], describer=_FakeDescriber())
        self.assertEqual(s["alerts"], 1)
        self.assertEqual(len(self.notifier.alerts), 1)
        self.assertIn("unfamiliar", self.notifier.alerts[0][0])
        self.assertIn("Ford F-150", self.notifier.alerts[0][0])
        m.close()

    def test_person_alerts(self):
        person = _mk_detection("person", 0.9, (600.0, 330.0, 660.0, 470.0))
        m, s = self._run([person])
        self.assertEqual(s["alerts"], 1)
        self.assertIn("Human", self.notifier.alerts[0][0])
        m.close()

    # -- deliverable 3: driveway expected-suppression (backfill -> live) -----

    def test_backfill_baseline_then_live_expected_note(self):
        import stealthcam.motion_ingest as mi
        resident = _mk_detection("truck", 0.8, (712.0, 266.0, 1000.0, 374.0))
        # Phase 1: backfill (baseline mode) — decisions are recorded (the
        # first sighting IS unfamiliar) but nothing notifies: no describer,
        # no notifier, no dashboard are passed at all.
        m1 = _mk(self.db, "wichita")
        s1 = mi.process_event(self._event("20260913_120000_WichitaDriveway_b"),
                              _FixedDetector([resident]), None, m1, None,
                              None, Path(self.tmp.name) / "sub",
                              baseline_mode=True,
                              local_data_dir=self.tmp.name)
        self.assertEqual(self.notifier.alerts, [])
        self.assertEqual(self.dashboard.notes, [])
        m1.close()
        # Seed the monitored denominator + resident presence (as further
        # backfilled days would).
        m = _mk(self.db, "wichita")
        for i in range(9):
            t = _iso(_T0 + timedelta(days=1 + i))
            m.record_monitored("WichitaDriveway", "driveway", f"mon-{i}", t,
                               is_day=True)
            m.decide("WichitaDriveway", "truck", (712.0, 266.0, 1000.0, 374.0),
                     now=t, fingerprint=b"veh", region="driveway",
                     guid=f"mon-{i}")
        m.close()
        # Phase 2: live sighting of the SAME entity -> background note.
        m2 = _mk(self.db, "wichita")
        s2 = mi.process_event(self._event("20260914_120000_WichitaDriveway_2"),
                              _FixedDetector([resident]), _FakeDescriber(),
                              m2, self.notifier, self.dashboard,
                              Path(self.tmp.name) / "sub",
                              baseline_mode=False,
                              local_data_dir=self.tmp.name)
        self.assertEqual(s2["alerts"], 0)
        self.assertEqual(s2["notes"], 1)
        self.assertEqual(self.notifier.alerts, [])
        self.assertIn("[background]", self.dashboard.notes[-1][3])
        m2.close()

    def test_unfamiliar_after_resident_converges(self):
        # a DIFFERENT vehicle (distinct fingerprint) in the driveway after
        # baseline seeding ALERTs — the stranger must not inherit the
        # resident's expectedness.
        import stealthcam.motion_ingest as mi
        resident = _mk_detection("truck", 0.8, (712.0, 266.0, 1000.0, 374.0))
        stranger = _mk_detection("car", 0.75, (520.0, 324.0, 730.0, 432.0))
        m = _mk(self.db, "wichita")
        for i in range(10):
            t = _iso(_T0 + timedelta(hours=i))
            m.record_monitored("WichitaDriveway", "driveway", f"mon-{i}", t,
                               is_day=True)
            m.decide("WichitaDriveway", "truck", (712.0, 266.0, 1000.0, 374.0),
                     now=t, fingerprint=b"resident-key", region="driveway",
                     guid=f"mon-{i}")
        m.close()
        m2 = _mk(self.db, "wichita",
                 fp=_TrivialFingerprinter(b"stranger-key"))
        s2 = mi.process_event(self._event("20260914_130000_WichitaDriveway_3"),
                              _FixedDetector([stranger]), _FakeDescriber(),
                              m2, self.notifier, self.dashboard,
                              Path(self.tmp.name) / "sub",
                              baseline_mode=False,
                              local_data_dir=self.tmp.name)
        self.assertEqual(s2["alerts"], 1)
        self.assertEqual(len(self.notifier.alerts), 1)
        m2.close()

    # -- deliverable 1: scope isolation through the adapter ------------------

    def test_scope_leak_guard_in_adapter(self):
        road_car = _mk_detection("car", 0.72, (500.0, 259.0, 576.0, 288.0))
        m_wi, s_wi = self._run([road_car])
        self.assertEqual(s_wi["notes"], 1)
        m_wi.close()
        m_th = _mk(self.db, "taylor_hill")
        import stealthcam.motion_ingest as mi
        mi.process_event(self._event(), _FixedDetector([road_car]), None,
                         m_th, self.notifier, self.dashboard,
                         Path(self.tmp.name) / "sub", baseline_mode=False,
                         local_data_dir=self.tmp.name)
        m_th.close()
        conn = sqlite3.connect(self.db)
        wi_dec = conn.execute(
            "SELECT COUNT(*) FROM decisions WHERE scope='wichita'"
        ).fetchone()[0]
        th_dec = conn.execute(
            "SELECT COUNT(*) FROM decisions WHERE scope='taylor_hill'"
        ).fetchone()[0]
        self.assertGreater(wi_dec, 0)
        self.assertGreater(th_dec, 0)
        # wichita road NOTE must not have touched taylor's rows and vice versa
        wi_road = conn.execute(
            "SELECT COUNT(*) FROM decisions WHERE scope='wichita' "
            "AND decision='note'").fetchone()[0]
        self.assertEqual(wi_road, 1)
        conn.close()

    # -- ingestor correctness: frames, seen-set, monitored denominator -------

    def test_missing_frame_not_seen_and_flagged(self):
        import stealthcam.motion_ingest as mi
        m = _mk(self.db, "wichita")
        event = self._event()
        event.frame_relpath = "frames/missing_f0.jpg"
        s = mi.process_event(event, _FixedDetector([]), None, m,
                             self.notifier, self.dashboard,
                             Path(self.tmp.name) / "sub",
                             local_data_dir=self.tmp.name)
        self.assertTrue(s.get("no_frame"))
        self.assertFalse(m.has_seen(event.event_name))
        self.assertEqual(self.notifier.alerts, [])
        m.close()

    def test_event_marked_seen_after_processing(self):
        road_car = _mk_detection("car", 0.72, (500.0, 259.0, 576.0, 288.0))
        m, s = self._run([road_car])
        self.assertTrue(m.has_seen(self._event().event_name))
        m.close()

    def test_monitored_recorded_with_road_region_when_pass_by(self):
        road_car = _mk_detection("car", 0.72, (500.0, 259.0, 576.0, 288.0))
        m, s = self._run([road_car])
        conn = sqlite3.connect(self.db)
        region = conn.execute(
            "SELECT region FROM monitored_captures WHERE scope='wichita'"
        ).fetchone()[0]
        self.assertEqual(region, "road")
        conn.close()
        m.close()

    def test_degenerate_bbox_skipped_loudly(self):
        bad = _mk_detection("car", 0.9, (300.0, 300.0, 300.0, 300.0))
        m, s = self._run([bad])
        self.assertEqual(s["detections"], 0)
        m.close()


# -- 7. dashboard labeling write path -----------------------------------------


class TestWriteEventLabeling(unittest.TestCase):
    def test_write_event_labeling_sql(self):
        from stealthcam.dashboard import DashboardWriter
        import stealthcam.dashboard as dash_mod
        captured = {}

        class _Cur:
            description = []

            def execute(self, sql, params):
                captured["sql"] = sql
                captured["params"] = params

            def fetchone(self):
                return (77,)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        class _Conn:
            def cursor(self):
                return _Cur()

            def commit(self):
                captured["committed"] = True

            def rollback(self):
                captured["rolled_back"] = True

        with patch.object(dash_mod.psycopg2, "connect",
                          return_value=_Conn()):
            dw = DashboardWriter.__new__(DashboardWriter)
            dw.db_url = "postgresql://fake"
            dw.local_data_dir = "/data/video/watcher"
            dw.base_static_url = "http://mira.local"
            dw.git_version = "test"
            dw._conn = _Conn()
            lid = dw.write_event_labeling(42, "vehicle", 0.7,
                                          "[traffic] vehicle pass-by")
        self.assertEqual(lid, 77)
        self.assertIn("INSERT INTO labelings", captured["sql"])
        self.assertEqual(captured["params"][0], "yolo11n+vlm")
        self.assertEqual(captured["params"][-1], 42)
        self.assertIn("[traffic]", captured["params"][4])
        self.assertTrue(captured.get("committed"))


# -- 8. config surface ---------------------------------------------------------


class TestWichitaScopeConfig(unittest.TestCase):
    def test_scope_setting_resolves_wichita_block(self):
        from stealthcam.config import Config
        cfg = Config()
        cfg.scopes = {"wichita": {
            "event_scenes": ["WichitaDriveway"],
            "detector_conf_threshold": 0.45,
            "road_band_y_max": 0.42,
        }}
        self.assertEqual(
            cfg.scope_setting("wichita", "event_scenes", []),
            ["WichitaDriveway"])
        self.assertEqual(
            cfg.scope_setting("wichita", "detector_conf_threshold", 0.6),
            0.45)
        # taylor scope unaffected
        self.assertEqual(
            cfg.scope_setting("taylor_hill", "detector_conf_threshold", 0.6),
            0.6)

    def test_new_scope_keys_are_whitelisted(self):
        from stealthcam.config import Config
        cfg = Config()
        for key in ("event_scenes", "event_poll_seconds", "event_backlog_max",
                    "road_band_y_max", "road_band_x_min", "road_band_x_max"):
            self.assertIn(key, cfg.SCOPE_KEYS)

    def test_typo_in_scope_block_raises(self):
        from stealthcam.config import Config
        with self.assertRaises(ValueError):
            Config(scopes={"wichita": {"road_band_ymax": 0.42}})

    def test_road_band_helper(self):
        import stealthcam.motion_ingest as mi
        from stealthcam.config import Config
        cfg = Config()
        cfg.scopes = {"wichita": {"road_band_y_max": 0.5}}
        band = mi._road_band(cfg)
        # memory_scope defaults to taylor_hill; set it explicitly for wichita
        cfg.memory_scope = "wichita"
        band = mi._road_band(cfg)
        self.assertEqual(band["y_max"], 0.5)
        self.assertEqual(band["x_min"], 0.30)
        self.assertEqual(band["x_max"], 0.52)


if __name__ == "__main__":
    unittest.main()