import unittest

from watcher.model import EventObservation, StealthcamFeedback, WatcherBase
from watcher.tests.utils import test_db_url


class TestFeedbackRoute(unittest.TestCase):
    """Exercise the /feedback route's toggle semantics end-to-end.

    Uses a real Flask test client against the test database. The route reads
    `from api import db`, so we create the app with the test URL and drive it
    through the client.
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

    def _make_event(self, name):
        from api import db
        with self.app.app_context():
            evt = EventObservation(
                event_name=name,
                video_file=f'{name}.mp4',
                capture_time='2021-01-01T12:00:00',
                scene_name='scene1',
                storage_local=True,
                video_location='/path/to/video',
            )
            db.session.add(evt)
            db.session.commit()
            return evt.id

    def _feedback(self, event_id, label, reason=None):
        body = {'event_id': event_id, 'label': label}
        if reason is not None:
            body['reason'] = reason
        return self.client.post('/feedback', json=body)

    def test_set_good_then_undo(self):
        eid = self._make_event('fb_route_good')
        r = self._feedback(eid, 'good')
        self.assertEqual(r.status_code, 201)
        self.assertEqual(r.get_json()['feedback']['label'], 'good')

        # same label again = undo
        r = self._feedback(eid, 'good')
        self.assertEqual(r.status_code, 200)
        self.assertIsNone(r.get_json()['feedback'])

    def test_set_bad_with_reason_then_switch_to_good(self):
        eid = self._make_event('fb_route_bad')
        r = self._feedback(eid, 'bad', reason='false alarm')
        self.assertEqual(r.status_code, 201)
        self.assertEqual(r.get_json()['feedback']['reason'], 'false alarm')

        # switching to good replaces the bad row
        r = self._feedback(eid, 'good')
        self.assertEqual(r.status_code, 201)
        self.assertEqual(r.get_json()['feedback']['label'], 'good')

        from api import db
        with self.app.app_context():
            fb = db.session.query(StealthcamFeedback).filter_by(event_id=eid).one()
            self.assertEqual(fb.label, 'good')
            self.assertIsNone(fb.reason)

    def test_invalid_label(self):
        eid = self._make_event('fb_route_invalid')
        r = self._feedback(eid, 'meh')
        self.assertEqual(r.status_code, 400)

    def test_missing_event(self):
        r = self._feedback(999999999, 'good')
        self.assertEqual(r.status_code, 404)


if __name__ == '__main__':
    unittest.main()
