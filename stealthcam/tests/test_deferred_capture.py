"""Tests for deferred-capture visibility + requeue-deferred backfill.

Covers the two PR #31 follow-ups:
  1. A capture deferred (all sub-images failed) across N consecutive polls
     posts a one-time Mattermost alert (tracked in the memory DB).
  2. `requeue_deferred` re-processes unseen captures older than the retry
     window, skipping seen and still-recent captures.
"""
from __future__ import annotations

import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from stealthcam.client import StealthcamError
from stealthcam.memory import Memory
from stealthcam.worker import process_capture, requeue_deferred


class _FailingClient:
    """download_subimage always raises StealthcamError."""

    def download_subimage(self, guid, n, dest):
        raise StealthcamError(f"download {guid}_{n} failed: HTTP 403")


class _EmptyDetector:
    def detect(self, image, conf_threshold=None):
        return []


class _FakeNotifier:
    def __init__(self):
        self.posts = []

    def post_alert(self, text, image_path=None):
        self.posts.append(text)
        return "post-id"


def _capture(created="2026-09-08T11:03:00"):
    return {
        "imageGuid": str(uuid.uuid4()),
        "imageDirections": ["East", "East", "East", "East", "East", "East"],
        "createdDateTime": created,
    }


class TestDeferredAlert(unittest.TestCase):
    def test_alerts_once_after_threshold_deferrals(self):
        memory = Memory(":memory:", scope="taylor_hill")
        notifier = _FakeNotifier()
        capture = _capture()
        guid = capture["imageGuid"]

        # Defer the same capture 3 times (threshold=3).
        for _ in range(3):
            process_capture(
                capture, _FailingClient(), _EmptyDetector(), None, memory,
                notifier, None, Path("/tmp"), baseline_mode=True,
                deferred_alert_threshold=3,
            )

        # Exactly one alert fired, on the 3rd deferral.
        self.assertEqual(len(notifier.posts), 1)
        self.assertIn(guid, notifier.posts[0])
        self.assertTrue(memory.is_deferred_alerted(guid))
        memory.close()

    def test_no_alert_below_threshold(self):
        memory = Memory(":memory:", scope="taylor_hill")
        notifier = _FakeNotifier()
        capture = _capture()

        for _ in range(2):
            process_capture(
                capture, _FailingClient(), _EmptyDetector(), None, memory,
                notifier, None, Path("/tmp"), baseline_mode=True,
                deferred_alert_threshold=3,
            )

        self.assertEqual(len(notifier.posts), 0)
        self.assertFalse(memory.is_deferred_alerted(capture["imageGuid"]))
        memory.close()

    def test_does_not_realert_after_alerted(self):
        memory = Memory(":memory:", scope="taylor_hill")
        notifier = _FakeNotifier()
        capture = _capture()

        for _ in range(5):
            process_capture(
                capture, _FailingClient(), _EmptyDetector(), None, memory,
                notifier, None, Path("/tmp"), baseline_mode=True,
                deferred_alert_threshold=3,
            )

        # Still exactly one alert despite 5 deferrals.
        self.assertEqual(len(notifier.posts), 1)
        memory.close()

    def test_mark_seen_clears_deferral(self):
        memory = Memory(":memory:", scope="taylor_hill")
        notifier = _FakeNotifier()
        capture = _capture()
        guid = capture["imageGuid"]

        # Defer once, then mark seen (simulating a later successful download).
        process_capture(
            capture, _FailingClient(), _EmptyDetector(), None, memory,
            notifier, None, Path("/tmp"), baseline_mode=True,
            deferred_alert_threshold=3,
        )
        self.assertEqual(memory.deferred_guids(), [guid])

        memory.mark_seen(guid)
        self.assertEqual(memory.deferred_guids(), [])
        memory.close()


class _ListImagesClient:
    """Fake client whose list_images returns a fixed page set."""

    def __init__(self, pages):
        self._pages = list(pages)
        self.calls = []

    def list_images(self, take_count=20, cursor=None):
        self.calls.append((take_count, cursor))
        if self._pages:
            return self._pages.pop(0)
        return []


class _RecordingProcessor:
    """Stands in for process_capture in requeue_deferred tests."""

    def __init__(self):
        self.processed = []

    def __call__(self, capture, client, detector, describer, memory,
                 notifier, dashboard, subimage_dir, **kwargs):
        self.processed.append(capture["imageGuid"])
        memory.mark_seen(capture["imageGuid"])


def _old_capture(guid, hours_ago):
    created = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    return {"imageGuid": guid, "createdDateTime": created}


class TestRequeueDeferred(unittest.TestCase):
    def test_requeues_only_unseen_old_captures(self):
        memory = Memory(":memory:", scope="taylor_hill")
        old_unseen = str(uuid.uuid4())
        old_seen = str(uuid.uuid4())
        recent_unseen = str(uuid.uuid4())

        # old_seen is already marked seen; recent_unseen is inside the window.
        memory.mark_seen(old_seen)

        client = _ListImagesClient([
            [
                _old_capture(recent_unseen, 1),   # recent -> skip
                _old_capture(old_seen, 48),       # seen -> skip
                _old_capture(old_unseen, 48),     # old + unseen -> process
            ],
        ])
        processor = _RecordingProcessor()

        with mock.patch("stealthcam.worker.process_capture", side_effect=processor):
            n = requeue_deferred(
                None, client, None, None, memory, None, None, Path("/tmp"),
                older_than_hours=24.0,
            )

        self.assertEqual(n, 1)
        self.assertEqual(processor.processed, [old_unseen])
        self.assertTrue(memory.has_seen(old_unseen))
        memory.close()

    def test_pages_until_exhausted(self):
        memory = Memory(":memory:", scope="taylor_hill")
        # Two full pages (50 each) then an empty page, so the loop pages twice.
        page1 = [_old_capture(str(uuid.uuid4()), 48) for _ in range(50)]
        page2 = [_old_capture(str(uuid.uuid4()), 48) for _ in range(50)]
        client = _ListImagesClient([page1, page2, []])
        processor = _RecordingProcessor()

        with mock.patch("stealthcam.worker.process_capture", side_effect=processor):
            n = requeue_deferred(
                None, client, None, None, memory, None, None, Path("/tmp"),
                older_than_hours=24.0,
            )

        self.assertEqual(n, 100)
        self.assertEqual(len(processor.processed), 100)
        # Cursor advanced across pages: 2 full pages + 1 empty terminator.
        self.assertEqual(len(client.calls), 3)
        self.assertIsNotNone(client.calls[1][1])  # second call carried a cursor
        memory.close()


if __name__ == "__main__":
    unittest.main()
