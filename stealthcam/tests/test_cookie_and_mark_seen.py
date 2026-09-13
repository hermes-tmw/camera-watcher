"""Tests for the stealthcam client cookie-refresh and worker mark-seen fixes.

Covers the two defects fixed in this change:
  1. client.download_subimage refreshes the signed CloudFront cookie on 403
     and retries once.
  2. worker.process_capture only marks a capture seen when >=1 sub-image
     downloaded successfully.
"""
from __future__ import annotations

import unittest
import uuid
from pathlib import Path
from unittest import mock

from stealthcam.client import StealthcamClient, StealthcamError
from stealthcam.memory import Memory
from stealthcam.worker import process_capture


def _make_client() -> StealthcamClient:
    return StealthcamClient(secrets={
        "app_base": "https://app.example.com",
        "cdn_base": "https://cdn.example.com",
        "device_id": "402_867490078575292",
        "email": "x@example.com",
        "password": "pw",
    })


class _FakeResponse:
    def __init__(self, status_code: int, content: bytes = b""):
        self.status_code = status_code
        self.content = content


class TestDownloadSubimageCookieRefresh(unittest.TestCase):
    def test_403_then_200_refreshes_cookie_and_returns_image(self):
        client = _make_client()
        guid = str(uuid.uuid4())
        dest = Path("/tmp") / f"{guid}_1.JPG"

        # Simulate a cached (stale) cookie and a fresh one after refresh.
        client._cookie_header = "stale=cookie"

        # First CDN GET 403s (stale cookie), retry 200s with fresh cookie.
        responses = [
            _FakeResponse(403, b"x" * 111),
            _FakeResponse(200, b"y" * 2000),
        ]
        seen_cookies = []

        def fake_get(url, headers=None, timeout=None):
            seen_cookies.append(headers["Cookie"])
            return responses.pop(0)

        with mock.patch("stealthcam.client.requests.get", side_effect=fake_get), \
             mock.patch.object(client, "_refresh_cookie",
                               return_value=".CloudFront.Cookies=fresh=cookie") as refresh:
            result = client.download_subimage(guid, 1, dest)

        self.assertEqual(result, dest)
        self.assertTrue(dest.exists())
        self.assertEqual(dest.read_bytes(), b"y" * 2000)
        # First request used the stale cookie, retry used the fresh one.
        self.assertEqual(seen_cookies, ["stale=cookie", ".CloudFront.Cookies=fresh=cookie"])
        # The refresh path was exercised exactly once.
        refresh.assert_called_once_with()

    def test_403_then_403_raises(self):
        client = _make_client()
        guid = str(uuid.uuid4())
        dest = Path("/tmp") / f"{guid}_1.JPG"
        client._cookie_header = "stale=cookie"

        responses = [_FakeResponse(403, b"x" * 111), _FakeResponse(403, b"x" * 111)]

        def fake_get(url, headers=None, timeout=None):
            return responses.pop(0)

        with mock.patch("stealthcam.client.requests.get", side_effect=fake_get), \
             mock.patch.object(client, "_refresh_cookie",
                               return_value=".CloudFront.Cookies=fresh=cookie"):
            with self.assertRaises(StealthcamError):
                client.download_subimage(guid, 1, dest)

    def test_refresh_cookie_invalidates_cache_and_refetches(self):
        client = _make_client()
        client._cookie_header = "stale=cookie"

        # Stub the cloudfront-token POST + the session cookie jar so
        # _ensure_cookie sees a fresh cookie after the re-fetch.
        with mock.patch.object(client.session, "post") as fake_post, \
             mock.patch.object(client.session, "cookies") as fake_jar:
            fake_post.return_value = _FakeResponse(200)
            fake_jar.items.return_value = [(".CloudFront.Cookies", "fresh=cookie")]
            result = client._refresh_cookie()

        self.assertEqual(result, ".CloudFront.Cookies=fresh=cookie")
        self.assertEqual(client._cookie_header, ".CloudFront.Cookies=fresh=cookie")
        # The cloudfront-token endpoint was re-hit (not served from cache).
        fake_post.assert_called_once()


class _FailingClient:
    """download_subimage always raises StealthcamError."""

    def download_subimage(self, guid, n, dest):
        raise StealthcamError(f"download {guid}_{n} failed: HTTP 403")


class _PartialClient:
    """download_subimage succeeds for n==1, fails for n>1."""

    def download_subimage(self, guid, n, dest):
        if n == 1:
            dest.parent.mkdir(parents=True, exist_ok=True)
            # A valid 1x1 JPEG so the worker's Image.open() succeeds.
            from PIL import Image
            Image.new("RGB", (1, 1)).save(dest, format="JPEG")
            return dest
        raise StealthcamError(f"download {guid}_{n} failed: HTTP 403")


class _EmptyDetector:
    def detect(self, image, conf_threshold=None):
        return []


class TestProcessCaptureMarkSeen(unittest.TestCase):
    def _capture(self):
        return {
            "imageGuid": str(uuid.uuid4()),
            "imageDirections": ["East", "East", "East", "East", "East", "East"],
            "createdDateTime": "2026-09-08T11:03:00",
        }

    def test_all_downloads_fail_does_not_mark_seen(self):
        memory = Memory(":memory:", scope="taylor_hill")
        capture = self._capture()
        guid = capture["imageGuid"]

        summary = process_capture(
            capture, _FailingClient(), _EmptyDetector(), None, memory,
            None, None, Path("/tmp"), baseline_mode=True,
        )

        self.assertFalse(memory.has_seen(guid))
        self.assertEqual(summary["detections"], 0)
        memory.close()

    def test_partial_download_marks_seen(self):
        memory = Memory(":memory:", scope="taylor_hill")
        capture = self._capture()
        guid = capture["imageGuid"]

        summary = process_capture(
            capture, _PartialClient(), _EmptyDetector(), None, memory,
            None, None, Path("/tmp"), baseline_mode=True,
        )

        self.assertTrue(memory.has_seen(guid))
        memory.close()


if __name__ == "__main__":
    unittest.main()
