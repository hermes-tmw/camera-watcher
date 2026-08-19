import unittest

from watcher.model import EventObservation, WatcherBase
from watcher.tests.utils import test_db_url


class TestBrowserInfiniteScroll(unittest.TestCase):
    """Exercise the /browser + /events routes that infinite scroll depends on.

    The infinite-scroll change (GH#12) replaced the "Load more" button with a
    #scroll-sentinel div + IntersectionObserver. The server-side contract is
    unchanged: /browser renders the sentinel (with data-page/data-filter/
    data-camera/data-description) when has_more is true, and /events returns
    {events, has_more, page} for the next page. This test pins that contract.
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

    def _make_event(self, name, capture_time):
        from api import db
        with self.app.app_context():
            evt = EventObservation(
                event_name=name,
                video_file=f'{name}.mp4',
                capture_time=capture_time,
                scene_name='scene1',
                storage_local=True,
                video_location='/path/to/video',
            )
            db.session.add(evt)
            db.session.commit()
            return evt.id

    def _clear_events(self):
        from api import db
        with self.app.app_context():
            db.session.query(EventObservation).delete()
            db.session.commit()

    def test_browser_renders_sentinel_when_has_more(self):
        # PAGE_SIZE is 24; seed 25 recent events so page 1 has_more=True.
        self._clear_events()
        for i in range(25):
            self._make_event(f'scroll_evt_{i}', f'2026-08-{i % 28 + 1:02d}T12:00:00')
        r = self.client.get('/browser?filter=recent')
        self.assertEqual(r.status_code, 200)
        html = r.get_data(as_text=True)
        self.assertIn('id="scroll-sentinel"', html)
        self.assertIn('data-page="2"', html)
        self.assertIn('IntersectionObserver', html)
        self.assertNotIn('load-more-btn', html)
        # The app is served under /watcher (nginx rewrites /watcher/* -> /*).
        # Every fetch() must carry the prefix or it 404s against nginx's
        # static `location /`. Pin the API_BASE constant and that no bare
        # absolute-path fetch survives.
        self.assertIn("const API_BASE = '/watcher'", html)
        self.assertNotIn("fetch('/events", html)
        self.assertNotIn("fetch(`/events", html)
        self.assertNotIn("fetch('/feedback", html)

    def test_events_returns_next_page(self):
        self._clear_events()
        for i in range(25):
            self._make_event(f'scroll_page_{i}', f'2026-08-{i % 28 + 1:02d}T12:00:00')
        r = self.client.get('/events?page=1&filter=recent')
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertEqual(len(data['events']), 24)
        self.assertTrue(data['has_more'])
        self.assertEqual(data['page'], 1)

        r2 = self.client.get('/events?page=2&filter=recent')
        data2 = r2.get_json()
        self.assertFalse(data2['has_more'])
        self.assertEqual(data2['page'], 2)


if __name__ == '__main__':
    unittest.main()
