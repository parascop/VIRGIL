"""Ollama client + VLM event verifier.

The VLM is a skeptical *verifier and summariser*, not a rubber stamp. Every
response is validated in Python even when schema-constrained decoding is used.
Failures never raise out of ``verify``: they yield ``VLMResult(ok=False)`` so the
pipeline can store the event as ``unverified``.
"""
from __future__ import annotations

import base64
import json
import logging
import math
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from .types import CATEGORIES, NONE_CATEGORY, VLM_CATEGORIES, EventRecord, VLMRequest, VLMResult
from .util import encode_jpeg

log = logging.getLogger("vigil.vlm")

MAX_SUMMARY_WORDS = 25

# JSON Schema passed to Ollama's structured-output ``format`` field (>= 0.5.0).
VLM_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": list(VLM_CATEGORIES)},
        "vlm_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "summary": {"type": "string"},
        "objects": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
    "required": ["category", "vlm_confidence", "summary", "objects", "reason"],
}

SYSTEM_PROMPT = """You are a skeptical video-surveillance analyst. An automatic heuristic raised an alert and you must VERIFY it from the images, not agree with it.

Rules:
- The image is a composite of consecutive video frames in CHRONOLOGICAL order: left-to-right, then top-to-bottom (tile 1 is the oldest). Evaluate the SEQUENCE of events; do not treat the composite as one simultaneous scene.
- The heuristic's claim is only a hypothesis. Decide whether the frames directly show it.
- Do not invent people, objects, actions or context that are not clearly visible. If the evidence is unclear or absent, answer "none".
- Separate what is directly visible from what is uncertain; lower vlm_confidence when you are unsure.
- summary: at most 25 words, describing only what is visible.
- objects: short list of visible objects/people relevant to the event.
- reason: one short sentence justifying the category.

Categories:
- "person down": a person falls or lies on the floor/ground and stays down (not sitting, bending, crouching, kneeling or exercising).
- "fighting": two or more people in physical violence (punching, kicking, grappling). Not hugging, dancing, playing or walking close together.
- "tailgating": a second person follows closely behind another through a controlled entry so both pass on one authorisation.
- "crowd incident": an unusually dense gathering or sudden mass movement/dispersal/panic. Ordinary calm busy areas or queues are "none".
- "suspicious activity": a person loitering in or entering a restricted area, or an unattended object left behind.
- "none": the claim is not supported by the images (false alarm).

Reply with ONLY a JSON object: {"category", "vlm_confidence" (0..1), "summary", "objects", "reason"}."""


class OllamaError(RuntimeError):
    def __init__(self, msg: str, kind: str = "error"):
        super().__init__(msg)
        self.kind = kind  # connect | timeout | http | error


def _parse_version(v: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])


class OllamaClient:
    """Minimal HTTP client for the local Ollama service."""

    def __init__(self, base_url: str = "http://localhost:11434", connect_timeout: float = 3.0,
                 read_timeout: float = 45.0):
        self.base_url = base_url.rstrip("/")
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self._version: Optional[str] = None

    def _url(self, path: str) -> str:
        return self.base_url + path

    def version(self) -> Optional[str]:
        try:
            r = requests.get(self._url("/api/version"), timeout=(self.connect_timeout, 5))
            r.raise_for_status()
            self._version = r.json().get("version")
            return self._version
        except Exception:  # noqa: BLE001
            return None

    def supports_schema(self) -> bool:
        v = self._version or self.version()
        return bool(v) and _parse_version(v) >= (0, 5, 0)

    def tags(self) -> List[str]:
        r = requests.get(self._url("/api/tags"), timeout=(self.connect_timeout, 10))
        r.raise_for_status()
        return [m.get("name", "") for m in r.json().get("models", [])]

    def has_model(self, tag: str) -> bool:
        names = self.tags()
        return tag in names or (":" not in tag and f"{tag}:latest" in names)

    def ps(self) -> List[Dict[str, Any]]:
        """Models currently loaded (with ``size`` and ``size_vram``)."""
        try:
            r = requests.get(self._url("/api/ps"), timeout=(self.connect_timeout, 5))
            r.raise_for_status()
            return r.json().get("models", [])
        except Exception:  # noqa: BLE001
            return []

    def tag_sizes(self) -> Dict[str, int]:
        """name -> size in bytes of every pulled model (used to prefer smaller models)."""
        r = requests.get(self._url("/api/tags"), timeout=(self.connect_timeout, 10))
        r.raise_for_status()
        return {m.get("name", ""): int(m.get("size") or 0) for m in r.json().get("models", [])}

    def unload(self, model: str) -> None:
        """Ask Ollama to free the model now (keep_alive=0). Best effort."""
        try:
            requests.post(self._url("/api/generate"), json={"model": model, "keep_alive": 0},
                          timeout=(self.connect_timeout, 30))
        except Exception:  # noqa: BLE001
            pass

    def chat(self, model: str, messages: List[Dict[str, Any]], schema: Optional[Dict[str, Any]] = None,
             options: Optional[Dict[str, Any]] = None, keep_alive: Optional[str] = None,
             timeout: Optional[float] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        if schema is not None:
            body["format"] = schema if self.supports_schema() else "json"
        if options:
            body["options"] = options
        if keep_alive is not None:
            body["keep_alive"] = keep_alive
        try:
            r = requests.post(self._url("/api/chat"), json=body,
                              timeout=(self.connect_timeout, timeout or self.read_timeout))
        except requests.exceptions.ConnectTimeout as e:
            raise OllamaError(f"connect timeout: {e}", "connect") from e
        except requests.exceptions.ConnectionError as e:
            raise OllamaError(f"cannot connect to Ollama at {self.base_url}: {e}", "connect") from e
        except requests.exceptions.ReadTimeout as e:
            raise OllamaError(f"inference timed out after {timeout or self.read_timeout}s", "timeout") from e
        except requests.exceptions.RequestException as e:
            raise OllamaError(f"request failed: {e}", "error") from e
        if r.status_code != 200:
            raise OllamaError(f"HTTP {r.status_code}: {r.text[:300]}", "http")
        try:
            return r.json()
        except ValueError as e:
            raise OllamaError(f"non-JSON HTTP body: {r.text[:200]}", "http") from e


# ---------------------------------------------------------------- validation
def normalize_category(c: Any) -> Optional[str]:
    if not isinstance(c, str):
        return None
    s = re.sub(r"[\s_\-]+", " ", c.strip().lower())
    return s if s in VLM_CATEGORIES else None


def validate_vlm_output(text: str) -> Tuple[Optional[Dict[str, Any]], str, bool]:
    """Parse + validate. Returns (clean dict | None, error message, summary_truncated)."""
    if not isinstance(text, str) or not text.strip():
        return None, "empty response", False
    t = text.strip()
    if t.startswith("```"):  # tolerate fenced JSON
        t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    try:
        d = json.loads(t)
    except json.JSONDecodeError as e:
        return None, f"malformed JSON: {e}", False
    if not isinstance(d, dict):
        return None, "JSON root is not an object", False
    cat = normalize_category(d.get("category"))
    if cat is None:
        return None, f"invalid category {d.get('category')!r}", False
    conf = d.get("vlm_confidence")
    if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not math.isfinite(float(conf)):
        return None, f"invalid vlm_confidence {conf!r}", False
    conf = float(conf)
    if not 0.0 <= conf <= 1.0:
        return None, f"vlm_confidence out of range: {conf}", False
    summary = d.get("summary", "")
    if not isinstance(summary, str):
        return None, "summary is not a string", False
    words = summary.split()
    truncated = len(words) > MAX_SUMMARY_WORDS
    if truncated:
        summary = " ".join(words[:MAX_SUMMARY_WORDS])
    objs = d.get("objects", [])
    if not isinstance(objs, list) or not all(isinstance(o, str) for o in objs):
        return None, "objects must be a list of strings", False
    reason = d.get("reason", "")
    if not isinstance(reason, str):
        return None, "reason is not a string", False
    return ({"category": cat, "vlm_confidence": conf, "summary": summary.strip(), "objects": objs[:20],
             "reason": reason.strip()}, "", truncated)


# ------------------------------------------------------------------ verifier
class VLMVerifier:
    """Calls the selected Ollama model once per triggered event."""

    def __init__(self, cfg: Dict[str, Any], client: Optional[OllamaClient] = None):
        self.cfg = cfg
        self.model = cfg["model"]
        self.client = client or OllamaClient(cfg["base_url"], cfg["connect_timeout_s"], cfg["timeout_s"])
        self._down_until = 0.0
        self._lock = threading.Lock()

    def options(self) -> Dict[str, Any]:
        o: Dict[str, Any] = {"temperature": self.cfg.get("temperature", 0), "num_ctx": self.cfg["num_ctx"],
                             "num_predict": self.cfg["num_predict"]}
        if self.cfg.get("num_gpu") is not None:
            o["num_gpu"] = self.cfg["num_gpu"]
        return o

    def check(self) -> Dict[str, Any]:
        """Startup diagnostics: service reachable? model pulled?"""
        v = self.client.version()
        out: Dict[str, Any] = {"reachable": v is not None, "version": v, "model": self.model,
                               "model_available": False, "schema_supported": False, "error": ""}
        if v is None:
            out["error"] = (f"Ollama is not reachable at {self.client.base_url}. Install it from "
                            "https://ollama.com, start it (`ollama serve` or the desktop app) and run "
                            f"`ollama pull {self.model}`.")
            return out
        try:
            out["model_available"] = self.client.has_model(self.model)
        except Exception as e:  # noqa: BLE001
            out["error"] = f"could not list models: {e}"
        if not out["model_available"] and not out["error"]:
            out["error"] = f"model '{self.model}' is not pulled: run `ollama pull {self.model}`"
        out["schema_supported"] = self.client.supports_schema()
        return out

    def _messages(self, req: VLMRequest, b64: str, retry_note: str = "") -> List[Dict[str, Any]]:
        n = req.sample_count
        order = ("a single frame" if n == 1 else
                 f"{n} frames in chronological order (tile 1 oldest, tile {n} newest)")
        user = (f"Camera {req.trigger.camera_id}. The image is a composite of {order}, laid out "
                f"left-to-right then top-to-bottom. Tile labels show seconds relative to the last frame.\n"
                f"A heuristic detector claims: \"{req.trigger.category}\" because: {req.trigger.reason}\n"
                f"Treat this as an unverified claim. Verify it from the frames and answer with the JSON object only.")
        if retry_note:
            user += "\n" + retry_note
        return [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user, "images": [b64]}]

    def verify(self, req: VLMRequest) -> VLMResult:
        t0 = time.time()
        now = time.monotonic()
        with self._lock:
            down = self._down_until > now
        if down:
            return VLMResult(ok=False, model=self.model, t_start=t0, t_end=time.time(),
                             error="ollama unavailable (in back-off after a connection failure)")
        try:
            b64 = base64.b64encode(encode_jpeg(req.composite, 85)).decode("ascii")
        except Exception as e:  # noqa: BLE001
            return VLMResult(ok=False, model=self.model, t_start=t0, t_end=time.time(),
                             error=f"could not encode composite: {e}")
        raw_all: List[str] = []
        err = ""
        attempts = 0
        for attempt in range(1 + int(self.cfg["retries"])):
            attempts += 1
            note = "" if attempt == 0 else ("Your previous reply was not valid. Reply with ONLY a JSON object "
                                            "with keys category, vlm_confidence, summary, objects, reason.")
            try:
                resp = self.client.chat(self.model, self._messages(req, b64, note), schema=VLM_SCHEMA,
                                        options=self.options(), keep_alive=self.cfg["keep_alive"],
                                        timeout=self.cfg["timeout_s"])
            except OllamaError as e:
                err = f"{e.kind}: {e}"
                if e.kind == "connect":
                    with self._lock:
                        self._down_until = time.monotonic() + float(self.cfg["unavailable_cooldown_s"])
                log.warning("[%s] VLM request failed: %s", req.event_id, err)
                return VLMResult(ok=False, model=self.model, t_start=t0, t_end=time.time(), error=err,
                                 raw="\n---\n".join(raw_all), attempts=attempts)
            content = (resp.get("message") or {}).get("content", "")
            raw_all.append(content if isinstance(content, str) else repr(content))
            clean, verr, trunc = validate_vlm_output(content)
            if clean is not None:
                return VLMResult(ok=True, category=clean["category"], vlm_confidence=clean["vlm_confidence"],
                                 summary=clean["summary"], objects=clean["objects"], reason=clean["reason"],
                                 raw="\n---\n".join(raw_all), model=self.model, t_start=t0,
                                 t_end=time.time(), attempts=attempts, summary_truncated=trunc)
            err = f"invalid structured output: {verr}"
            log.warning("[%s] %s (attempt %d)", req.event_id, err, attempt + 1)
        return VLMResult(ok=False, model=self.model, t_start=t0, t_end=time.time(), error=err,
                         raw="\n---\n".join(raw_all), attempts=attempts)


# ------------------------------------------------------------------- verdicts
def confidence_score(heuristic: float, vlm_support: float, weights: Dict[str, float]) -> float:
    """0.4*heuristic + 0.6*vlm (weights from YAML). Uncalibrated ranking score."""
    return float(weights["heuristic_weight"]) * heuristic + float(weights["vlm_weight"]) * vlm_support


def apply_verdict(req: VLMRequest, res: Optional[VLMResult], vlm_cfg: Dict[str, Any],
                  weights: Dict[str, float], unverified_reason: str = "") -> Dict[str, Any]:
    """Map a (trigger, VLM result) pair to verdict/category/confidence fields.

    * valid result, category none            -> rejected (kept for FP evaluation)
    * valid result, category == trigger's    -> verified
    * valid result, different non-none cat   -> verified as VLM's category when
      ``allow_reclassify`` else rejected
    * no valid result                        -> unverified, heuristic category
    """
    h = req.trigger.heuristic_score
    if res is None or not res.ok:
        reason = unverified_reason or (res.error if res else "VLM not run")
        return {"verdict": "unverified", "category": req.trigger.category, "vlm_confidence": 0.0,
                "summary": "", "error": reason, "confidence_score": confidence_score(h, 0.0, weights),
                "category_changed": False}
    if res.category == NONE_CATEGORY:
        return {"verdict": "rejected", "category": NONE_CATEGORY, "vlm_confidence": res.vlm_confidence,
                "summary": res.summary, "error": "", "category_changed": False,
                "confidence_score": confidence_score(h, 1.0 - res.vlm_confidence, weights)}
    if res.category != req.trigger.category and not vlm_cfg.get("allow_reclassify", True):
        return {"verdict": "rejected", "category": NONE_CATEGORY, "vlm_confidence": res.vlm_confidence,
                "summary": res.summary, "error": "VLM category differs from heuristic and reclassification is off",
                "confidence_score": confidence_score(h, 1.0 - res.vlm_confidence, weights), "category_changed": True}
    return {"verdict": "verified", "category": res.category, "vlm_confidence": res.vlm_confidence,
            "summary": res.summary, "error": "", "category_changed": res.category != req.trigger.category,
            "confidence_score": confidence_score(h, res.vlm_confidence, weights)}
