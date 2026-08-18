"""Notify: post an alert to Mattermost with the highlighted sub-image."""
from __future__ import annotations

import json
from pathlib import Path

import requests


class Notifier:
    def __init__(self, base_url: str, channel_id: str, bot: str = "cody",
                 secrets_path: str = "~/.hermes/secrets/mattermost_persona_tokens.json"):
        self.base_url = base_url.rstrip("/")
        self.channel_id = channel_id
        self.bot = bot
        self.secrets_path = Path(secrets_path).expanduser()
        self._token: str | None = None

    def _get_token(self) -> str:
        if self._token:
            return self._token
        with open(self.secrets_path) as f:
            data = json.load(f)
        entry = data.get(self.bot)
        if not entry or not entry.get("token"):
            raise RuntimeError(f"no Mattermost token for bot '{self.bot}'")
        self._token = entry["token"]
        return self._token

    def post_alert(self, text: str, image_path: Path | None = None) -> str:
        """Post an alert message (optionally with an image) to the channel.

        Returns the post id.
        """
        token = self._get_token()
        headers = {"Authorization": f"Bearer {token}"}

        if image_path is not None and image_path.exists():
            # upload the file first
            with open(image_path, "rb") as f:
                up = requests.post(
                    f"{self.base_url}/api/v4/files",
                    headers=headers,
                    files={"files": (image_path.name, f, "image/jpeg")},
                    data={"channel_id": self.channel_id},
                    timeout=60,
                )
            up.raise_for_status()
            file_ids = [fi["id"] for fi in up.json().get("file_infos", [])]
        else:
            file_ids = []

        r = requests.post(
            f"{self.base_url}/api/v4/posts",
            headers=headers,
            json={"channel_id": self.channel_id, "message": text, "file_ids": file_ids},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["id"]
