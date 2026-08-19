import unittest

from datetime import datetime

from watcher.model import EventObservation, StealthcamFeedback, WatcherBase
from watcher.tests.utils import TransactionalTestCase


class TestStealthcamFeedback(TransactionalTestCase, unittest.TestCase):
    def _make_event(self, name='event1'):
        evt = EventObservation(
            event_name=name,
            video_file='video1.mp4',
            capture_time='2021-01-01T12:00:00',
            scene_name='scene1',
            storage_local=True,
            video_location='/path/to/video',
        )
        self.session.add(evt)
        self.session.commit()
        return evt

    def test_feedback_roundtrip(self):
        evt = self._make_event()
        fb = StealthcamFeedback(event_id=evt.id, label='bad', reason='false alarm')
        self.session.add(fb)
        self.session.commit()

        loaded = self.session.get(StealthcamFeedback, fb.id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.label, 'bad')
        self.assertEqual(loaded.reason, 'false alarm')
        self.assertEqual(loaded.event_id, evt.id)
        self.assertIsNotNone(loaded.created_at)

    def test_feedback_unique_per_event(self):
        from sqlalchemy.exc import IntegrityError
        from sqlalchemy import text
        evt = self._make_event()
        self.session.add(StealthcamFeedback(event_id=evt.id, label='good'))
        self.session.commit()
        self.session.add(StealthcamFeedback(event_id=evt.id, label='bad'))
        with self.assertRaises(IntegrityError):
            self.session.commit()
        # The failed commit aborts the transaction (and its savepoint), so
        # re-establish the savepoint the harness tearDown expects.
        self.session.rollback()
        self._conn.execute(text('SAVEPOINT test_sp'))

    def test_feedback_relationship(self):
        evt = self._make_event()
        self.session.add(StealthcamFeedback(event_id=evt.id, label='good'))
        self.session.commit()
        # relationship is populated on the event
        self.assertEqual(len(evt.feedback), 1)
        self.assertEqual(evt.feedback[0].label, 'good')


if __name__ == '__main__':
    unittest.main()
