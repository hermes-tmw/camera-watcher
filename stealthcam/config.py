"""Configuration loading for the stealthcam pipeline.

Config is a YAML file (default: ./config.yaml, override with STEALTHCAM_CONFIG).
Secrets (Stealthcam creds, Mattermost tokens) are NOT in the config file — they
live in ~/.hermes/secrets/ and are referenced by path.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG = "config.yaml"
LOCAL_CONFIG = "config.local.yaml"
PLACEHOLDER_PASSWORD = "***"


@dataclass
class Config:
    secrets_stealthcam: str = "~/.hermes/secrets/stealthcam.json"
    secrets_mattermost: str = "~/.hermes/secrets/mattermost_persona_tokens.json"

    mattermost_base_url: str = "http://mira:8065"
    mattermost_channel_id: str = ""
    mattermost_bot: str = "cody"

    pipeline_db: str = "state/stealthcam.db"
    subimage_dir: str = "/data/video/watcher/stealthcam"

    detector_weights: str = "/home/claw/stealthcam-eval/yolo11n.pt"
    detector_conf_threshold: float = 0.6
    detector_device: str = "0"

    describer_model: str = "llama3.2-vision:latest"
    describer_ollama_host: str = "http://localhost:11434"
    describer_timeout: int = 180

    memory_static_n_captures: int = 5
    memory_static_min_hours: float = 1.0
    memory_iou_threshold: float = 0.5

    dashboard_db_url: str = ""
    dashboard_base_static_url: str = "http://mira.local"
    dashboard_local_data_dir: str = "/data/video/watcher"

    poll_interval_seconds: int = 60
    poll_backoff_base: float = 2.0
    poll_backoff_max: int = 300
    poll_take_count: int = 20

    def __post_init__(self):
        self.secrets_stealthcam = os.path.expanduser(self.secrets_stealthcam)
        self.secrets_mattermost = os.path.expanduser(self.secrets_mattermost)


def load_config(path: str | None = None) -> Config:
    # Resolution order: explicit arg > STEALTHCAM_CONFIG env > config.local.yaml
    # (if present) > config.yaml. The local file carries the real DB password
    # and is gitignored; the committed config.yaml uses a `***` placeholder.
    if path is None:
        path = os.environ.get("STEALTHCAM_CONFIG")
    if path is None:
        path = LOCAL_CONFIG if os.path.isfile(LOCAL_CONFIG) else DEFAULT_CONFIG
    cfg = Config()
    if path and os.path.isfile(path):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        _apply(cfg, data)
    # Environment overrides (STEALTHCAM_<FLAT_KEY>) take precedence over the file.
    _apply_env(cfg)
    # In the camera-watcher compose stack, derive the dashboard DB URL from the
    # mounted watcher.cfg (gitignored) when no real URL was configured (empty or
    # still the `***` placeholder).
    if not cfg.dashboard_db_url or PLACEHOLDER_PASSWORD in cfg.dashboard_db_url:
        derived = _db_url_from_watcher_cfg()
        if derived:
            cfg.dashboard_db_url = derived
    # A `***` placeholder password must never reach a real connection attempt:
    # treat it as "dashboard not configured" so the pipeline degrades to
    # alert-only instead of crashing every cycle.
    if PLACEHOLDER_PASSWORD in (cfg.dashboard_db_url or ""):
        cfg.dashboard_db_url = ""
    return cfg


def _apply(cfg: Config, data: dict) -> None:
    flat = _flatten(data)
    for key, val in flat.items():
        attr = key.replace(".", "_")
        if hasattr(cfg, attr):
            setattr(cfg, attr, val)
    cfg.__post_init__()


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _db_url_from_watcher_cfg() -> str:
    """Build a Postgres URL from the camera-watcher `watcher.cfg` [database] section.

    When the pipeline runs inside the camera-watcher compose stack, the DB
    password lives in `etc/watcher.cfg` (gitignored, mounted read-only at
    /usr/local/etc/watcher.cfg) — the same file every other service reads. We
    derive the dashboard DB URL from it so the password never appears in the
    stealthcam repo or the compose file. Returns "" if the file is absent or
    the section is incomplete.
    """
    import configparser

    path = os.environ.get("WATCHER_CONFIG", "")
    if not path or not os.path.isfile(path):
        return ""
    try:
        p = configparser.ConfigParser()
        p.read(path)
        db = p["database"]
        user = db.get("DB_USER", "")
        pw = db.get("DB_PASS", "")
        host = db.get("DB_HOST", "")
        name = db.get("DB_NAME", "")
        if not (user and host and name):
            return ""
        return f"postgresql://{user}:{pw}@{host}/{name}"
    except (configparser.Error, KeyError):
        return ""


def _apply_env(cfg: Config) -> None:
    """Apply STEALTHCAM_<FLAT_KEY> environment overrides (highest precedence).

    e.g. STEALTHCAM_DASHBOARD_DB_URL -> dashboard.db_url -> dashboard_db_url.
    Lets the compose service inject the DB URL, secrets paths, weights path,
    and device without a committed config file.
    """
    for env, val in os.environ.items():
        if not env.startswith("STEALTHCAM_"):
            continue
        key = env[len("STEALTHCAM_"):].lower()
        # flat key uses dots (dashboard.db_url); attr uses underscores.
        attr = key.replace(".", "_")
        if hasattr(cfg, attr):
            setattr(cfg, attr, val)
