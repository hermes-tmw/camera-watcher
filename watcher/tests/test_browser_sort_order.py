import unittest

from watcher.model import EventObservation, WatcherBase
from watcher.tests.utils import test_db_url


class TestBrowserSortOrder(unittest.TestCase):
    """Exercise the sort-order control on /browser + /events (GH#24).

    The sort is server-side on ``EventObservation.capture_time``, driven by an
    ``order`` query param (asc|desc, default desc). It must compose with the
    camera/description/date-range/filter-mode filters and be respected by the
    /events fetch used by infinite scroll and the 30s poll.
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

    def _names(self, r):
        return [e['event_name'] for e in r.get_json()['events']]

    def test_default_is_descending(self):
        self._clear_events()
        self._make_event('oldest', '2026-08-01T12:00:00')
        self._make_event('middle', '2026-08-10T12:00:00')
        self._make_event('newest', '2026-08-20T12:00:00')

        r = self.client.get('/events?filter=recent')
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self._names(r), ['newest', 'middle', 'oldest'])

    def test_ascending_reorders_newest_last(self):
        self._clear_events()
        self._make_event('oldest', '2026-08-01T12:00:00')
        self._make_event('middle', '2026-08-10T12:00:00')
        self._make_event('newest', '2026-08-20T12:00:00')

        r = self.client.get('/events?filter=recent&order=asc')
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self._names(r), ['oldest', 'middle', 'newest'])

    def test_explicit_desc_matches_default(self):
        self._clear_events()
        self._make_event('oldest', '2026-08-01T12:00:00')
        self._make_event('newest', '2026-08-20T12:00:00')

        r = self.client.get('/events?filter=recent&order=desc')
        self.assertEqual(self._names(r), ['newest', 'oldest'])

    def test_invalid_order_falls_back_to_desc(self):
        self._clear_events()
        self._make_event('oldest', '2026-08-01T12:00:00')
        self._make_event('newest', '2026-08-20T12:00:00')

        r = self.client.get('/events?filter=recent&order=bogus')
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self._names(r), ['newest', 'oldest'])

    def test_sort_composes_with_camera(self):
        self._clear_events()
        self._make_event('camA_new', '2026-08-20T12:00:00', camera='camA')
        self._make_event('camA_old', '2026-08-01T12:00:00', camera='camA')
        self._make_event('camB_new', '2026-08-20T12:00:00', camera='camB')

        r = self.client.get('/events?filter=recent&order=asc&camera=camA')
        self.assertEqual(self._names(r), ['camA_old', 'camA_new'])

    def test_sort_composes_with_date_range(self):
        self._clear_events()
        self._make_event('in_old', '2026-08-06T12:00:00')
        self._make_event('in_new', '2026-08-14T12:00:00')
        self._make_event('out', '2026-08-20T12:00:00')

        r = self.client.get('/events?filter=recent&order=asc&start=2026-08-05&end=2026-08-15')
        self.assertEqual(self._names(r), ['in_old', 'in_new'])

    def test_sort_composes_with_filter_mode(self):
        self._clear_events()
        # 'all' mode still respects the sort order.
        self._make_event('oldest', '2026-08-01T12:00:00')
        self._make_event('newest', '2026-08-20T12:00:00')

        r = self.client.get('/events?filter=all&order=asc')
        self.assertEqual(self._names(r), ['oldest', 'newest'])

    def test_browser_renders_sort_control_default_desc(self):
        self._clear_events()
        self._make_event('evt', '2026-08-10T12:00:00')
        r = self.client.get('/browser?filter=recent')
        self.assertEqual(r.status_code, 200)
        html = r.get_data(as_text=True)
        self.assertIn('id="order-select"', html)
        # Default is desc (Newest first selected).
        self.assertIn('<option value="desc" selected>Newest first</option>', html)
        self.assertIn('<option value="asc" >Oldest first</option>', html)

    def test_browser_renders_sort_control_asc_selected(self):
        self._clear_events()
        self._make_event('evt', '2026-08-10T12:00:00')
        r = self.client.get('/browser?filter=recent&order=asc')
        html = r.get_data(as_text=True)
        self.assertIn('<option value="asc" selected>Oldest first</option>', html)

    def test_sentinel_carries_order(self):
        # Seed 25 events so page 1 has_more=True and the sentinel renders.
        self._clear_events()
        for i in range(25):
            self._make_event(f'evt_{i}', f'2026-08-{i % 28 + 1:02d}T12:00:00')
        r = self.client.get('/browser?filter=recent&order=asc')
        html = r.get_data(as_text=True)
        self.assertIn('id="scroll-sentinel"', html)
        self.assertIn('data-order="asc"', html)

    def test_poll_since_id_respects_order(self):
        self._clear_events()
        self._make_event('old', '2026-08-01T12:00:00')
        self._make_event('new', '2026-08-20T12:00:00')

        # since_id path returns events newer than the id, in the chosen order.
        r = self.client.get('/events?filter=recent&order=asc&since_id=0')
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self._names(r), ['old', 'new'])


if __name__ == '__main__':
    unittest.main()
