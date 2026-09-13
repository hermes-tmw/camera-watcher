"""Configuration loading for the stealthcam pipeline.

Config is a YAML file (default: ./config.yaml, override with STEALTHCAM_CONFIG).
Secrets (Stealthcam creds, Mattermost tokens) are NOT in the config file — they
live in ~/.hermes/secrets/ and are referenced by path.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG = "config.yaml"
LOCAL_CONFIG = "config.local.yaml"
PLACEHOLDER_PASSWORD = "***"


def _validate_scopes(scopes: dict, valid_keys: tuple[str, ...]) -> None:
    """Whitelist-check scope blocks eagerly (config load must fail loudly on a
    typo'd param, not silently ignore it at the first decision)."""
    for scope_name, params in (scopes or {}).items():
        if not isinstance(params, dict):
            raise ValueError(
                f"scopes.{scope_name} must be a mapping of param -> value, "
                f"got {type(params).__name__}")
        bad = set(params) - set(valid_keys)
        if bad:
            raise ValueError(
                f"scopes.{scope_name} has unknown param(s) {sorted(bad)}; "
                f"valid params: {sorted(valid_keys)}")


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

    # Night confidence gate (settled decision 4, data-backed 2026-09-13): at
    # night the detector confidence bar is raised above the day value so IR
    # background noise (<0.3 median) is suppressed. The eval over 59 night
    # captures confirmed 0.75 eliminates ~96.8% of IR noise while keeping clear
    # signals (2 real SUVs @ 0.86/0.85). Sub-threshold detections are dropped.
    detector_night_conf_threshold: float = 0.75

    # Day window (wall hours, US/Mountain) used to classify a capture as
    # day/night for the night gate AND the presence-baseline denominator
    # (settled decision 4: "configured dark hours"). Everything outside
    # [day_start_hour, day_end_hour) is night. Defaults mirror the historical
    # 7-20 window.
    day_start_hour: int = 7
    day_end_hour: int = 20

    describer_model: str = "llama3.2-vision:latest"
    describer_ollama_host: str = "http://localhost:11434"
    describer_timeout: int = 180

    memory_static_n_captures: int = 5
    memory_static_min_hours: float = 1.0
    memory_iou_threshold: float = 0.5

    # Entity identity (settled decisions 1 & 2). The re-id fingerprint is the
    # identity KEY; the describer VLM supplies make/model + lettering
    # attributes. Two crops are the same entity when their fingerprints match
    # within ``reid_similarity_threshold`` (normalized-L2 similarity, 1 - dist;
    # lowered to 0.80 after the v2 descriptor separated lookalikes).
    reid_similarity_threshold: float = 0.80
    reid_model: str = "cv2-silhouette-v2"

    # Expected-presence baseline: per entity_id x direction, an entity is
    # "typically here" when it is present in >= this fraction of monitored
    # captures over the rolling window. Default >=50% over 7 days (settled
    # decision 2), surfaced as config so it can be tuned. The denominator
    # counts DAY captures only (night captures inflate it without adding
    # numerator, since the night confidence gate drops most night detections).
    entity_expected_presence_ratio: float = 0.5
    entity_baseline_window_days: float = 7.0

    # Hysteresis: once an entity is "expected" it stays expected (no alert)
    # until it has been absent >= this many days, so a familiar vehicle at
    # ~threshold presence does not flip between alert/note on every minor miss
    # (occlusion, weather). Default 2 days.
    entity_expected_absence_days: float = 2.0

    # Location policy (settled decision 3). ``road`` = pass-by (note only, no
    # alert); ``driveway`` = identity-expectedness eval (unfamiliar -> alert).
    location_road: str = "road"
    location_driveway: str = "driveway"

    # Direction -> region mapping. The camera's sub-images are named by
    # COMPASS bearing (e.g. "East", "South_SouthWest"); which bearing faces the
    # driveway vs Wichita Rd is deployment knowledge we cannot infer. A
    # comma-separated "bearing=region" list, e.g. "east=driveway,west=road".
    # Direction names are matched case-insensitively by their leading token
    # (before any underscore). Unmapped directions -> unclassified region ->
    # novel-vehicle fallback policy.
    direction_regions: str = ""

    dashboard_db_url: str = ""
    dashboard_base_static_url: str = "http://mira.local"
    dashboard_local_data_dir: str = "/data/video/watcher"

    # Per-location scope (2026-09-13 refactor): which location this process
    # serves. The Memory engine keys every table row on this scope, so two
    # sites sharing one DB never leak identity/state across each other.
    # Default = taylor_hill: the ONLY live scope on the pipeline DB as of the
    # refactor, so the existing deploy needs zero config change.
    memory_scope: str = "taylor_hill"

    # Per-scope parameter overrides: {scope_name: {param: value}}. Params not
    # listed fall back to the legacy flat attrs above (direction_regions,
    # detector_night_conf_threshold, day_start_hour/day_end_hour,
    # entity_expected_presence_ratio, entity_baseline_window_days,
    # entity_expected_absence_days, reid_similarity_threshold, reid_model,
    # describer_model). Load from YAML `scopes:` or STEALTHCAM_SCOPES (JSON).
    # Use scope_setting(scope, key) rather than reading either layer directly.
    scopes: dict = field(default_factory=dict)

    poll_interval_seconds: int = 60
    poll_backoff_base: float = 2.0
    poll_backoff_max: int = 300
    poll_take_count: int = 20

    deferred_alert_threshold: int = 3
    requeue_older_than_hours: float = 24.0

    # Keys that may be overridden per scope (whitelist — a typo'd key in a
    # scopes block must raise, not silently do nothing).
    SCOPE_KEYS = (
        "direction_regions",
        "detector_night_conf_threshold",
        "day_start_hour",
        "day_end_hour",
        "entity_expected_presence_ratio",
        "entity_baseline_window_days",
        "entity_expected_absence_days",
        "reid_similarity_threshold",
        "reid_model",
        "describer_model",
    )

    def __post_init__(self):
        self.secrets_stealthcam = os.path.expanduser(self.secrets_stealthcam)
        self.secrets_mattermost = os.path.expanduser(self.secrets_mattermost)
        # Whitelist-check every scope block eagerly so a bogus param fails at
        # config load, not at the first decision. (load_config re-validates
        # after merging STEALTHCAM_SCOPES into self.scopes.)
        if self.scopes and not isinstance(self.scopes, dict):
            raise ValueError(
                f"scopes must be a mapping of scope name -> params, got "
                f"{type(self.scopes).__name__}")
        _validate_scopes(self.scopes, self.SCOPE_KEYS)

    def scope_setting(self, scope: str, key: str, default=None):
        """Per-scope config lookup: scopes.<scope>.<key> -> flat attr <key> -> default.

        The flat-attr fallback is what keeps the existing Taylor Hill deploy
        byte-identical: its settings arrive as legacy env/flat keys and apply
        to its scope unchanged, while a new scope (e.g. wichita) overrides any
        SCOPE_KEYS param in its own block without touching other scopes.
        """
        params = (self.scopes or {}).get(scope)
        if params and key in params:
            return params[key]
        return getattr(self, key, default)


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
    # STEALTHCAM_SCOPES: JSON map {scope: {param: value}} for per-scope
    # overrides (compose can inject them without a config file).
    env_scopes = os.environ.get("STEALTHCAM_SCOPES", "").strip()
    if env_scopes:
        try:
            parsed = json.loads(env_scopes)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"STEALTHCAM_SCOPES is not valid JSON: {e}; value was "
                f"{env_scopes[:200]!r}") from e
        if not isinstance(parsed, dict):
            raise ValueError(
                f"STEALTHCAM_SCOPES must be a JSON object mapping scope name "
                f"-> params, got {type(parsed).__name__}")
        cfg.scopes.update(parsed)
        _validate_scopes(cfg.scopes, cfg.SCOPE_KEYS)
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
        # STEALTHCAM_SCOPES is parsed as JSON by load_config (it is a nested
        # {scope: params} map, not a flat string attr) — never setattr it raw.
        if key == "scopes":
            continue
        # flat key uses dots (dashboard.db_url); attr uses underscores.
        attr = key.replace(".", "_")
        if hasattr(cfg, attr):
            setattr(cfg, attr, val)
