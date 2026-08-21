import unittest

from watcher.model import EventObservation, WatcherBase
from watcher.tests.utils import test_db_url


class TestBrowserEnlarge(unittest.TestCase):
    """Pin the whole-cell enlarge contract (GH#18).

    Clicking anywhere in an .event-card toggles the inline media player, so the
    "Enlarge" affordance works even when the user clicks the card body rather
    than the thumbnail or button. The card carries onclick="cardClick(event)";
    cardClick() ignores clicks on the feedback row and the video player so 👍/👎
    and play/pause don't also collapse the card.
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

    def _make_event(self, name, video_file, capture_time):
        from api import db
        with self.app.app_context():
            evt = EventObservation(
                event_name=name,
                video_file=video_file,
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

    def test_card_has_whole_cell_click_handler(self):
        self._clear_events()
        # A photo event (video_file ends in .JPG) so the card renders a
        # .photo-player and the 🔍 Enlarge button.
        self._make_event('enlarge_photo', 'frame_1.JPG', '2026-08-01T12:00:00')
        r = self.client.get('/browser?filter=recent')
        self.assertEqual(r.status_code, 200)
        html = r.get_data(as_text=True)

        # Whole-cell click handler is wired on the card.
        self.assertIn('onclick="cardClick(event)"', html)
        # cardClick() is defined and guards the feedback row + video player.
        self.assertIn('function cardClick(event)', html)
        self.assertIn("event.target.closest('[data-feedback-row]')", html)
        self.assertIn("event.target.closest('.video-player')", html)
        # The Enlarge button no longer carries its own onclick (the whole-cell
        # handler covers it) — avoids double-toggle.
        self.assertNotIn('onclick="toggleMedia', html)

    def test_video_card_uses_whole_cell_handler(self):
        self._clear_events()
        self._make_event('enlarge_video', 'clip.mp4', '2026-08-01T12:00:00')
        r = self.client.get('/browser?filter=recent')
        html = r.get_data(as_text=True)
        self.assertIn('onclick="cardClick(event)"', html)
        self.assertIn('class="video-player hidden"', html)


if __name__ == '__main__':
    unittest.main()
