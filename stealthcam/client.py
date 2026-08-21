"""Stealthcam Revolver REST client — corrected auth flow (verified live 2026-08-18).

Auth flow:
  login:        POST {app_base}/api/v1/login  {email, password, useAuthCookie: false}
                -> {accessToken (JWT), refreshToken, ...}
  list images:  POST {app_base}/api/v8/file-manager/images-list
                body {takeCount, createdDateTime?, uploadedDateTime?} (cursor pagination)
  cdn token:    POST {app_base}/api/v1/cloudfront-token  {useAuthCookie: true}
                -> sets a signed cookie named `.CloudFront.Cookies` (leading dot)
  download:     GET {cdn_base}/{deviceId}/{guid}_{n}.JPG
                with explicit `Cookie: .CloudFront.Cookies=<val>` header
                (NOT `?token=` — that returns 401).

The leading-dot cookie name is mangled by requests' cookie jar, so we send the
Cookie header explicitly.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import requests

SECRETS_PATH = Path.home() / ".hermes" / "secrets" / "stealthcam.json"


class StealthcamError(RuntimeError):
    """Raised on any API/auth/download failure."""


def validate_guid(guid: str) -> str:
    """Validate a capture guid is a UUID before it reaches any file path.

    The guid comes from the Stealthcam API response (a semi-trusted boundary).
    It is interpolated into sub-image file paths and dashboard event names, so
    a crafted guid could otherwise traverse out of the sub-image directory.
    Returns the guid unchanged on success; raises StealthcamError otherwise.
    """
    try:
        uuid.UUID(guid)
    except (ValueError, AttributeError, TypeError):
        raise StealthcamError(f"invalid imageGuid: {guid!r}")
    return guid


def load_secrets(path: str | None = None) -> dict:
    p = Path(path) if path else SECRETS_PATH
    with open(p) as f:
        return json.load(f)


class StealthcamClient:
    def __init__(self, secrets: dict | None = None, secrets_path: str | None = None):
        self.s = secrets or load_secrets(secrets_path)
        self.app_base = self.s["app_base"].rstrip("/")
        self.cdn_base = self.s["cdn_base"].rstrip("/")
        self.device_id = self.s["device_id"]
        # The iot/status/on-demand endpoints key on the physical device
        # identifier (e.g. "402_867490078575292"), NOT the numeric device_id.
        self.physical_device_identifier = self.s.get(
            "physical_device_identifier", str(self.device_id)
        )
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "stealthcam-pipeline/0.1"})
        self.access_token: str | None = None
        self._cookie_header: str | None = None

    # -- auth ----------------------------------------------------------------

    def login(self) -> dict:
        r = self.session.post(
            f"{self.app_base}/api/v1/login",
            json={
                "email": self.s["email"],
                "password": self.s["password"],
                "useAuthCookie": False,
            },
            timeout=30,
        )
        if r.status_code != 200:
            raise StealthcamError(f"login failed: HTTP {r.status_code}")
        j = r.json()
        self.access_token = j.get("accessToken")
        if not self.access_token:
            raise StealthcamError(f"login: no accessToken in {list(j.keys())}")
        self.session.headers["Authorization"] = f"Bearer {self.access_token}"
        return j

    def _ensure_cookie(self) -> str:
        if self._cookie_header:
            return self._cookie_header
        r = self.session.post(
            f"{self.app_base}/api/v1/cloudfront-token",
            json={"useAuthCookie": True},
            timeout=30,
        )
        if r.status_code != 200:
            raise StealthcamError(f"cloudfront-token failed: HTTP {r.status_code}")
        cookies = list(self.session.cookies.items())
        if not cookies:
            raise StealthcamError("cloudfront-token: no cookie set")
        name, val = cookies[0]
        self._cookie_header = f"{name}={val}"
        return self._cookie_header

    # -- data ----------------------------------------------------------------

    def list_images(self, take_count: int = 20, cursor: dict | None = None) -> list[dict]:
        """Return the newest `take_count` images (cursor-paginated back in time).

        `cursor` is {createdDateTime, uploadedDateTime} of the last image of the
        previous page (to page further back). Returns the raw image records.
        """
        body: dict[str, Any] = {"takeCount": take_count}
        if cursor:
            body["createdDateTime"] = cursor["createdDateTime"]
            body["uploadedDateTime"] = cursor["uploadedDateTime"]
        r = self.session.post(
            f"{self.app_base}/api/v8/file-manager/images-list",
            json=body,
            timeout=30,
        )
        if r.status_code != 200:
            raise StealthcamError(f"images-list failed: HTTP {r.status_code}")
        return r.json().get("images", [])

    def get_device_status(self) -> dict:
        """Return the camera's live telemetry/status record.

        This is the endpoint the webapp's battery/signal/disk icons are fed
        from (discovered from the frontend bundle's XHR, verified live
        2026-08-21): `GET /api/v4/device-management/devices/status` with the
        array-form query param `physicalDeviceIdentifier[]=<pid>`.

        Returns the `cameraStatuses[0]` dict, e.g.:
          {physicalDeviceIdentifier, sdCardFreeSpace (pct free), batteryLevel
           (pct), batteryVolt, externalBatteryLevel, externalBatteryVolt,
           rssi, signalStrength, batteryLevelBar, externalBatteryLevelBar,
           firmwareVersion, lastSyncDateUnixTime (ms), deviceState
           {onDemandState, newPhotoCounter, onDemandError}, errors [...],
           rawErrors, powerLvlVolts [...]}

        Raises StealthcamError if the request fails or no status is returned.
        """
        r = self.session.get(
            f"{self.app_base}/api/v4/device-management/devices/status"
            f"?physicalDeviceIdentifier[]={self.physical_device_identifier}",
            timeout=30,
        )
        if r.status_code != 200:
            raise StealthcamError(f"device status failed: HTTP {r.status_code}")
        statuses = r.json().get("cameraStatuses") or []
        if not statuses:
            raise StealthcamError("device status: no cameraStatuses returned")
        return statuses[0]

    def trigger_on_demand(self, media_type: str = "Photo") -> dict:
        """Request a one-shot on-demand capture (manual health check only).

        Mirrors the webapp's exact call: `POST /api/v1/device-management/iot/
        <pid>/ondemand` with body `{type: <media_type>}` (the frontend sends
        `{type, force}`; `force` is optional). Returns the parsed response.

        VERIFIED LIVE 2026-08-21: this works on the Revolver 402. A successful
        trigger moves the device state OnDemandEnabled -> PhotoRequested ->
        PhotoUploaded, increments newPhotoCounter, and produces a new image
        with isOnDemand=True. The 400 `ODStateNotSupported` error is
        STATE-DEPENDENT, not a hard "unsupported": it is returned when a
        request is already in flight (the device is no longer in
        OnDemandEnabled state). Callers should check get_device_status()'s
        onDemandState and retry once the state returns to OnDemandEnabled.

        On-demand is a manual one-shot only — it must NEVER be used as a
        substitute for PIR motion detection (a single snapshot cannot catch a
        passerby between triggers). The error is surfaced as a StealthcamError
        so callers fail loudly rather than silently assuming a photo was taken.
        """
        r = self.session.post(
            f"{self.app_base}/api/v1/device-management/iot/{self.physical_device_identifier}/ondemand",
            json={"type": media_type},
            timeout=30,
        )
        if r.status_code not in (200, 204):
            raise StealthcamError(
                f"on-demand failed: HTTP {r.status_code} {r.text[:200]}"
            )
        return r.json() if r.content else {}

    def download_subimage(self, guid: str, n: int, dest: Path) -> Path:
        """Download sub-image `n` (1..6) of a capture to `dest` (a file path)."""
        validate_guid(guid)
        cookie = self._ensure_cookie()
        url = f"{self.cdn_base}/{self.device_id}/{guid}_{n}.JPG"
        # NOTE: use a fresh requests.get, NOT self.session.get — the session's
        # Authorization header + cookie jar cause the CDN to return 401. The
        # signed cookie must be sent as the ONLY auth on a clean request.
        r = requests.get(url, headers={"Cookie": cookie}, timeout=60)
        if r.status_code != 200 or len(r.content) < 1000:
            raise StealthcamError(
                f"download {guid}_{n} failed: HTTP {r.status_code}, {len(r.content)} bytes"
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        return dest
