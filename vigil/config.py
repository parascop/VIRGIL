"""YAML configuration loading, merging and validation.

A *profile* is ``configs/defaults.yaml`` deep-merged with a hardware profile
(``configs/profile_*.yaml``). Cameras come from ``configs/cameras.yaml`` and may
override heuristic thresholds per camera. Invalid configuration raises
``ConfigError`` with a list of every problem, before the pipeline starts.
"""
from __future__ import annotations

import copy
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

log = logging.getLogger("vigil.config")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"


class ConfigError(ValueError):
    """Raised for missing or invalid configuration."""


def deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand_env(value: Any) -> Any:
    """Expand ``${VAR}`` in strings (used for RTSP URLs / secrets)."""
    if isinstance(value, str):
        def repl(m: "re.Match[str]") -> str:
            v = os.environ.get(m.group(1))
            if v is None:
                raise ConfigError(f"environment variable {m.group(1)} is not set")
            return v
        return _ENV.sub(repl, value)
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    return value


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in {path}: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping")
    return data


def _num(errs: List[str], cfg: Dict[str, Any], dotted: str, lo: float = None,
         hi: float = None, integer: bool = False) -> None:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            errs.append(f"missing required setting '{dotted}'")
            return
        cur = cur[part]
    if isinstance(cur, bool) or not isinstance(cur, (int, float)):
        errs.append(f"'{dotted}' must be a number, got {cur!r}")
        return
    if integer and int(cur) != cur:
        errs.append(f"'{dotted}' must be an integer, got {cur!r}")
    if lo is not None and cur < lo:
        errs.append(f"'{dotted}' must be >= {lo}, got {cur}")
    if hi is not None and cur > hi:
        errs.append(f"'{dotted}' must be <= {hi}, got {cur}")


def validate_profile(cfg: Dict[str, Any]) -> None:
    errs: List[str] = []
    for sect in ("paths", "detector", "ingest", "vlm", "confidence", "embeddings",
                 "retrieval", "categories", "heuristics"):
        if not isinstance(cfg.get(sect), dict):
            errs.append(f"missing section '{sect}'")
    if errs:
        raise ConfigError("; ".join(errs))
    for k in ("sqlite", "chroma", "keyframes", "live"):
        if not cfg["paths"].get(k):
            errs.append(f"paths.{k} is required")
    _num(errs, cfg, "detector.imgsz", 64, 2048, True)
    _num(errs, cfg, "detector.conf", 0.0, 1.0)
    _num(errs, cfg, "ingest.target_fps", 0.1, 120)
    _num(errs, cfg, "ingest.queue_maxsize", 1, 10000, True)
    _num(errs, cfg, "ingest.ring_seconds", 1, 60)
    _num(errs, cfg, "vlm.timeout_s", 1, 3600)
    _num(errs, cfg, "vlm.connect_timeout_s", 0.1, 600)
    _num(errs, cfg, "vlm.queue_maxsize", 1, 1000, True)
    _num(errs, cfg, "vlm.retries", 0, 1, True)
    _num(errs, cfg, "vlm.composite_max_side", 128, 2048, True)
    _num(errs, cfg, "vlm.composite_frames", 1, 4, True)
    _num(errs, cfg, "vlm.num_ctx", 256, 131072, True)
    _num(errs, cfg, "confidence.heuristic_weight", 0, 1)
    _num(errs, cfg, "confidence.vlm_weight", 0, 1)
    _num(errs, cfg, "retrieval.image_weight", 0, 1)
    _num(errs, cfg, "retrieval.summary_weight", 0, 1)
    _num(errs, cfg, "cooldown_s", 0, 86400)
    if cfg["retrieval"].get("mode") not in ("image", "summary", "hybrid"):
        errs.append("retrieval.mode must be image | summary | hybrid")
    if cfg["vlm"].get("queue_overflow") not in ("lowest_priority", "drop_oldest", "reject_new"):
        errs.append("vlm.queue_overflow must be lowest_priority | drop_oldest | reject_new")
    if not str(cfg["vlm"].get("base_url", "")).startswith(("http://", "https://")):
        errs.append("vlm.base_url must start with http:// or https://")
    if not cfg["vlm"].get("model"):
        errs.append("vlm.model is required")
    try:
        from zoneinfo import ZoneInfo
        ZoneInfo(str(cfg.get("timezone", "")))
    except Exception as e:  # noqa: BLE001 - tzdata missing or unknown zone
        errs.append(f"timezone '{cfg.get('timezone')}' is invalid ({e}); on Windows `pip install tzdata`")
    for trig, cat in cfg["categories"].items():
        from .types import CATEGORIES
        if cat not in CATEGORIES:
            errs.append(f"categories.{trig} maps to unknown category '{cat}'")
    if errs:
        raise ConfigError("invalid profile:\n  - " + "\n  - ".join(errs))


def load_profile(profile_path: Optional[str | Path], configs_dir: Path = CONFIG_DIR) -> Dict[str, Any]:
    """Load defaults + profile, validate, resolve relative paths."""
    defaults = _read_yaml(configs_dir / "defaults.yaml")
    prof: Dict[str, Any] = {}
    if profile_path:
        p = Path(profile_path)
        if not p.is_absolute() and not p.exists():
            p = PROJECT_ROOT / p
        prof = _read_yaml(p)
    cfg = deep_merge(defaults, prof)
    cfg = expand_env(cfg)
    env_url = os.environ.get("VIGIL_OLLAMA_URL")
    if env_url:
        cfg["vlm"]["base_url"] = env_url
    validate_profile(cfg)
    cfg["_profile_path"] = str(profile_path or "")
    cfg["_root"] = str(PROJECT_ROOT)
    for k, v in list(cfg["paths"].items()):
        pv = Path(v)
        cfg["paths"][k] = str(pv if pv.is_absolute() else PROJECT_ROOT / pv)
    return cfg


@dataclass
class CameraConfig:
    id: str
    source: Any  # str path/URL or int webcam index
    name: str = ""
    target_fps: float = 10.0
    description: str = ""
    zones: List[Dict[str, Any]] = field(default_factory=list)
    entry_lines: List[Dict[str, Any]] = field(default_factory=list)
    thresholds: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_webcam(self) -> bool:
        return isinstance(self.source, int)

    @property
    def is_network(self) -> bool:
        return isinstance(self.source, str) and "://" in self.source

    @property
    def is_file(self) -> bool:
        return not (self.is_webcam or self.is_network)


_ID = re.compile(r"^[A-Za-z0-9_\-]{1,32}$")


def _validate_polygon(errs: List[str], where: str, poly: Any) -> None:
    if not (isinstance(poly, list) and len(poly) >= 3
            and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in poly)):
        errs.append(f"{where}: polygon needs >= 3 [x, y] points")
        return
    for x, y in poly:
        if not (isinstance(x, (int, float)) and isinstance(y, (int, float)) and 0 <= x <= 1 and 0 <= y <= 1):
            errs.append(f"{where}: polygon coordinates must be normalised 0..1, got {[x, y]}")
            return


def parse_camera(raw: Dict[str, Any], default_fps: float, resolve_paths: bool = True) -> CameraConfig:
    errs: List[str] = []
    cid = str(raw.get("id", ""))
    if not _ID.match(cid):
        errs.append(f"camera id {cid!r} must match [A-Za-z0-9_-]{{1,32}}")
    src = raw.get("source")
    if src is None or src == "":
        errs.append(f"camera {cid}: 'source' is required")
    if isinstance(src, str):
        src = expand_env(src)
        if src.isdigit():
            src = int(src)
        elif "://" not in src and resolve_paths:
            p = Path(src)
            src = str(p if p.is_absolute() else PROJECT_ROOT / p)
    zones = raw.get("zones") or []
    for z in zones:
        if z.get("type") not in ("restricted", "prohibited", "crowd"):
            errs.append(f"camera {cid}: zone {z.get('name')} has invalid type {z.get('type')!r}")
        _validate_polygon(errs, f"camera {cid} zone {z.get('name')}", z.get("polygon"))
    lines = raw.get("entry_lines") or []
    for ln in lines:
        for k in ("p1", "p2"):
            pt = ln.get(k)
            if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
                errs.append(f"camera {cid}: entry line {ln.get('name')} needs {k}: [x, y]")
        if ln.get("allowed_direction", "positive") not in ("positive", "negative"):
            errs.append(f"camera {cid}: entry line {ln.get('name')} allowed_direction must be positive|negative")
    fps = float(raw.get("target_fps", default_fps))
    if fps <= 0:
        errs.append(f"camera {cid}: target_fps must be > 0")
    if errs:
        raise ConfigError("invalid camera config:\n  - " + "\n  - ".join(errs))
    return CameraConfig(id=cid, source=src, name=str(raw.get("name", cid)), target_fps=fps,
                        description=str(raw.get("description", "")), zones=list(zones),
                        entry_lines=list(lines), thresholds=dict(raw.get("thresholds") or {}))


def load_cameras(path: str | Path, default_fps: float = 10.0) -> List[CameraConfig]:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = PROJECT_ROOT / p
    data = _read_yaml(p)
    raw_cams = data.get("cameras")
    if not isinstance(raw_cams, list) or not raw_cams:
        raise ConfigError(f"{p}: 'cameras' must be a non-empty list")
    cams = [parse_camera(c, default_fps) for c in raw_cams]
    ids = [c.id for c in cams]
    if len(set(ids)) != len(ids):
        raise ConfigError(f"{p}: duplicate camera ids {ids}")
    return cams


def heuristic_params(profile: Dict[str, Any], camera: Optional[CameraConfig]) -> Dict[str, Any]:
    """Profile heuristics merged with the per-camera ``thresholds`` override."""
    base = profile["heuristics"]
    return deep_merge(base, camera.thresholds) if camera else copy.deepcopy(base)


def to_local_dirs(profile: Dict[str, Any]) -> None:
    """Create the persistence directories (idempotent)."""
    for k in ("keyframes", "live", "clips", "jobs", "cache"):
        if k in profile["paths"]:
            Path(profile["paths"][k]).mkdir(parents=True, exist_ok=True)
    Path(profile["paths"]["sqlite"]).parent.mkdir(parents=True, exist_ok=True)
    Path(profile["paths"]["chroma"]).parent.mkdir(parents=True, exist_ok=True)
