import unittest

from watcher.model import EventObservation, WatcherBase
from watcher.tests.utils import test_db_url


class TestBrowserDateRange(unittest.TestCase):
    """Exercise the date-range filter on /browser + /events (GH#19).

    The date filter is server-side on ``EventObservation.capture_time``, driven
    by ``start``/``end`` query params (YYYY-MM-DD, interpreted in the camera's
    timezone). It must compose with the camera/description/filter-mode filters
    and be respected by the /events fetch used by infinite scroll and poll.
    """

    @classmethod
    def setUpClass(cls):
        from api import create_app, db
        cls.app = create_app(db_url=test_db_url(), testing=True)
        cls.app.config['TESTING'] = True
        with cls.app.app_context():
            WatcherBase.metadata.create_all(db.engine)
        cls.client = cls.app.test_client()

    @classmethod
    def tearDownClass(cls):
        from api import db
        with cls.app.app_context():
            WatcherBase.metadata.drop_all(db.engine)

    def _make_event(self, name, capture_time, camera='camA'):
        from api import db
        with self.app.app_context():
            evt = EventObservation(
                event_name=name,
                video_file=f'{name}.mp4',
                capture_time=capture_time,
                scene_name='scene1',
                storage_local=True,
                video_location='/path/to/video',
                camera=camera,
            )
            db.session.add(evt)
            db.session.commit()
            return evt.id

    def _clear_events(self):
        from api import db
        with self.app.app_context():
            db.session.query(EventObservation).delete()
            db.session.commit()

    def test_date_range_filters_events(self):
        self._clear_events()
        # Three events: one inside the range, one before, one after.
        self._make_event('in_range', '2026-08-10T12:00:00')
        self._make_event('before', '2026-08-01T12:00:00')
        self._make_event('after', '2026-08-20T12:00:00')

        r = self.client.get('/events?filter=recent&start=2026-08-05&end=2026-08-15')
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        names = {e['event_name'] for e in data['events']}
        self.assertIn('in_range', names)
        self.assertNotIn('before', names)
        self.assertNotIn('after', names)

    def test_date_range_inclusive_boundaries(self):
        self._clear_events()
        # Boundary events exactly on start/end dates must be included.
        self._make_event('on_start', '2026-08-05T00:00:00')
        self._make_event('on_end', '2026-08-15T23:59:00')
        self._make_event('outside', '2026-08-16T00:00:00')

        r = self.client.get('/events?filter=recent&start=2026-08-05&end=2026-08-15')
        data = r.get_json()
        names = {e['event_name'] for e in data['events']}
        self.assertIn('on_start', names)
        self.assertIn('on_end', names)
        self.assertNotIn('outside', names)

    def test_start_only_and_end_only(self):
        self._clear_events()
        self._make_event('old', '2026-08-01T12:00:00')
        self._make_event('mid', '2026-08-10T12:00:00')
        self._make_event('new', '2026-08-20T12:00:00')

        # start only: everything on/after the start date
        r = self.client.get('/events?filter=recent&start=2026-08-10')
        names = {e['event_name'] for e in r.get_json()['events']}
        self.assertIn('mid', names)
        self.assertIn('new', names)
        self.assertNotIn('old', names)

        # end only: everything on/before the end date
        r = self.client.get('/events?filter=recent&end=2026-08-10')
        names = {e['event_name'] for e in r.get_json()['events']}
        self.assertIn('old', names)
        self.assertIn('mid', names)
        self.assertNotIn('new', names)

    def test_date_range_composes_with_camera(self):
        self._clear_events()
        self._make_event('camA_in', '2026-08-10T12:00:00', camera='camA')
        self._make_event('camB_in', '2026-08-10T12:00:00', camera='camB')

        r = self.client.get('/events?filter=recent&start=2026-08-05&end=2026-08-15&camera=camA')
        names = {e['event_name'] for e in r.get_json()['events']}
        self.assertIn('camA_in', names)
        self.assertNotIn('camB_in', names)

    def test_date_range_overrides_recent_window(self):
        # An explicit start date older than the 60-day "recent" window must
        # still return matching events (the range overrides the implicit cutoff).
        self._clear_events()
        self._make_event('ancient', '2025-01-10T12:00:00')

        r = self.client.get('/events?filter=recent&start=2025-01-01&end=2025-02-01')
        data = r.get_json()
        names = {e['event_name'] for e in data['events']}
        self.assertIn('ancient', names)

    def test_browser_renders_date_inputs(self):
        self._clear_events()
        self._make_event('evt', '2026-08-10T12:00:00')
        r = self.client.get('/browser?filter=recent&start=2026-08-05&end=2026-08-15')
        self.assertEqual(r.status_code, 200)
        html = r.get_data(as_text=True)
        self.assertIn('id="start-input"', html)
        self.assertIn('id="end-input"', html)
        self.assertIn('value="2026-08-05"', html)
        self.assertIn('value="2026-08-15"', html)
        # The sentinel must carry the range so infinite scroll stays consistent.
        self.assertIn('data-start="2026-08-05"', html)
        self.assertIn('data-end="2026-08-15"', html)

    def test_invalid_date_is_ignored(self):
        self._clear_events()
        self._make_event('evt', '2026-08-10T12:00:00')
        # A malformed date must not 500; it is treated as "no filter".
        r = self.client.get('/events?filter=recent&start=not-a-date')
        self.assertEqual(r.status_code, 200)
        self.assertIn('events', r.get_json())


if __name__ == '__main__':
    unittest.main()
