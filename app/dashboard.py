"""Vigil Streamlit dashboard (Process 2).

Reads SQLite events, ChromaDB vectors, data/live/cam{N}.jpg and status.json.
It never loads YOLO and, while the pipeline runs, never loads CLIP either: query
text embeddings come from the pipeline process over a small file RPC.

Launch: streamlit run app/dashboard.py
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from vigil.config import CONFIG_DIR, ConfigError, load_cameras, load_profile  # noqa: E402
from vigil.ingest import probe_video  # noqa: E402
from vigil.media import browser_playable, extract_clip, extract_frame  # noqa: E402
from vigil.query import LLMQueryParser, SearchEngine, SearchResponse, SearchResult  # noqa: E402
from vigil.storage import EventStore, resolve_keyframe  # noqa: E402
from vigil.types import CATEGORIES, EventRecord  # noqa: E402
from vigil.util import atomic_write_json, fmt_ts  # noqa: E402

st.set_page_config(page_title="Vigil - local video event intelligence", page_icon="🛡️", layout="wide")

CAT_COLOURS = {"person down": "#ef476f", "fighting": "#f78c6b", "tailgating": "#ffd166", "crowd incident": "#06d6a0",
               "suspicious activity": "#118ab2", "none": "#6c757d"}
EXAMPLES = ["Show all tailgating events from camera 2 today", "Find possible person-down events in the last hour",
            "Show suspicious activity near the restricted entrance", "Find frames containing a person wearing a red shirt",
            "When did someone enter through the door?", "Show possible fighting events that occurred yesterday",
            "Find crowd incidents between 2 PM and 4 PM"]

st.markdown("""
<style>
.block-container {padding-top: 1.2rem;}
.vg-badge {display:inline-block;padding:2px 9px;border-radius:999px;font-size:0.76rem;font-weight:600;margin-right:6px;color:#0e1117;}
.vg-muted {color:#9aa4b2;font-size:0.82rem;}
.vg-unver {border-left:4px solid #ffd166;padding-left:8px;}
.vg-ok {color:#06d6a0;font-weight:600} .vg-warn {color:#ffd166;font-weight:600} .vg-bad {color:#ef476f;font-weight:600}
</style>""", unsafe_allow_html=True)


# --------------------------------------------------------------------- state
def esc(s: Any) -> str:
    return html.escape(str(s))


def _profile_path() -> str:
    env = os.environ.get("VIGIL_PROFILE")
    if env:
        return env
    st_file = _live_dir_guess() / "status.json"
    try:
        name = json.loads(st_file.read_text()).get("profile")
        cand = CONFIG_DIR / f"profile_{name}.yaml"
        if name and cand.exists():
            return str(cand)
    except Exception:  # noqa: BLE001
        pass
    for n in ("profile_rtx2050.yaml", "profile_mac_m4.yaml", "profile_gtx1650.yaml"):
        if (CONFIG_DIR / n).exists():
            return str(CONFIG_DIR / n)
    return ""


def _live_dir_guess() -> Path:
    return ROOT / "data" / "live"


@st.cache_resource(show_spinner="Opening event database...")
def backend(profile_path: str) -> Dict[str, Any]:
    prof = load_profile(profile_path)
    store = EventStore(prof["paths"]["sqlite"])
    cams = []
    try:
        cams = load_cameras(CONFIG_DIR / "cameras.yaml", prof["ingest"]["target_fps"])
    except ConfigError:
        pass
    index = None
    index_error = ""
    try:
        from vigil.embeddings import VectorIndex

        index = VectorIndex(prof["paths"]["chroma"], f'{prof["embeddings"]["model"]}/{prof["embeddings"]["pretrained"]}')
    except Exception as e:  # noqa: BLE001
        index_error = f"{type(e).__name__}: {e}"
    return {"prof": prof, "store": store, "cams": cams, "index": index, "index_error": index_error}


class UnavailableIndex:
    def __init__(self, why: str):
        self.why = why

    def scores(self, *a: Any, **k: Any) -> Dict[str, float]:
        raise RuntimeError(f"ChromaDB unavailable ({self.why}). Run `python -m vigil.embeddings --rebuild`.")


class UIEmbedder:
    """Query-text embedder for the UI: asks the pipeline process first (no model
    in this process); only if the pipeline is down does it lazily load CLIP
    locally (text tower on CPU) so search keeps working."""

    is_stub = False
    tag = "ui"

    def __init__(self, prof: Dict[str, Any]):
        from vigil.embeddings import RpcTextEmbedder

        self.prof = prof
        self.rpc = RpcTextEmbedder(Path(prof["paths"]["jobs"]).parent / "rpc", timeout_s=4.0)
        self.local = None

    def embed_texts(self, texts: List[str]):
        status = read_status(self.prof)
        if pipeline_alive(status) and status.get("embedder_ok"):
            try:
                return self.rpc.embed_texts(texts)
            except Exception:  # noqa: BLE001
                pass
        if self.local is None:
            from vigil.embeddings import make_embedder

            self.local = make_embedder(self.prof, "cpu")
        return self.local.embed_texts(texts)


@st.cache_resource(show_spinner=False)
def search_engine(profile_path: str) -> SearchEngine:
    b = backend(profile_path)
    prof = b["prof"]
    llm = None
    if prof["query"]["llm_parser"]:
        from vigil.vlm import OllamaClient

        llm = LLMQueryParser(OllamaClient(prof["vlm"]["base_url"], prof["vlm"]["connect_timeout_s"], prof["query"]["llm_timeout_s"]),
                             prof["vlm"]["model"], prof["query"]["llm_timeout_s"], prof["vlm"]["keep_alive"])
    index = b["index"] or UnavailableIndex(b["index_error"])
    return SearchEngine(b["store"], index, UIEmbedder(prof), prof["retrieval"], prof["timezone"], llm,
                        [c.id for c in b["cams"]])


def read_status(prof: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        return json.loads((Path(prof["paths"]["live"]) / "status.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def pipeline_alive(status: Optional[Dict[str, Any]]) -> bool:
    return bool(status) and status.get("health") != "stopped" and time.time() - status.get("updated_at", 0) < 6


PROFILE_PATH = _profile_path()
try:
    B = backend(PROFILE_PATH)
except ConfigError as e:
    st.error(f"Configuration problem: {e}")
    st.stop()
PROF, STORE, TZ = B["prof"], B["store"], B["prof"]["timezone"]
KEYFRAMES = PROF["paths"]["keyframes"]
if "selected" not in st.session_state:
    st.session_state.selected = None  # {"id":..., "t":...}
    st.session_state.chat = []
    st.session_state.processed_uploads = {}


def select_event(event_id: str) -> None:
    st.session_state.selected = {"id": event_id}


# ----------------------------------------------------------------- rendering
def badge(text: str, colour: str) -> str:
    return f'<span class="vg-badge" style="background:{colour}">{esc(text)}</span>'


def verdict_badge(v: str) -> str:
    return badge(v, {"verified": "#06d6a0", "rejected": "#6c757d", "unverified": "#ffd166"}.get(v, "#999"))


def keyframe_path(ev: EventRecord) -> Optional[Path]:
    return resolve_keyframe(KEYFRAMES, ev.keyframe_path)


def event_header_html(ev: EventRecord) -> str:
    cat = ev.category if ev.verdict != "unverified" else ev.category
    return (badge(cat, CAT_COLOURS.get(cat, "#999")) + verdict_badge(ev.verdict) +
            f'<span class="vg-muted">{esc(ev.camera_id)} · {esc(fmt_ts(ev.timestamp, TZ))}</span>')


def confidence_text(ev: EventRecord) -> str:
    return (f"ranking score {ev.confidence_score:.2f} (0.4·heuristic {ev.heuristic_score:.2f} + 0.6·VLM {ev.vlm_confidence:.2f}; "
            "uncalibrated, not a probability)")


def render_card(ev: EventRecord, sr: Optional[SearchResult] = None, key: str = "") -> None:
    with st.container(border=True):
        c1, c2 = st.columns([1, 2])
        kp = keyframe_path(ev)
        with c1:
            if kp:
                st.image(str(kp), use_container_width=True, caption="2×2 chronological composite (tiles 1→4)")
            else:
                st.warning("keyframe file missing")
        with c2:
            st.markdown(event_header_html(ev), unsafe_allow_html=True)
            cls = "vg-unver" if ev.verdict == "unverified" else ""
            st.markdown(f'<div class="{cls}">{esc(ev.summary or "(no summary)")}</div>', unsafe_allow_html=True)
            if ev.verdict == "unverified":
                st.caption(f"⚠ Not verified by the VLM: {ev.error or 'verification unavailable'}")
            if ev.verdict == "rejected":
                st.caption("Rejected by the VLM (kept for evaluation).")
            st.caption(confidence_text(ev) + (f" · duration {ev.duration_s:.1f}s" if ev.duration_s else ""))
            if sr is not None:
                parts = []
                if sr.hybrid_score is not None:
                    parts.append(f"match score {sr.hybrid_score:.2f}")
                if sr.image_score is not None:
                    parts.append(f"image cos {sr.image_score:.3f}")
                if sr.summary_score is not None:
                    parts.append(f"summary cos {sr.summary_score:.3f}")
                if sr.lexical_score is not None:
                    parts.append(f"keyword overlap {sr.lexical_score:.2f}")
                if parts:
                    st.caption("similarity (not a probability): " + " · ".join(parts))
            has_src = bool(ev.source_path) and Path(ev.source_path).exists()
            st.button("▶ Inspect clip / jump to time" if has_src else "🔎 Inspect event", key=f"sel_{key}_{ev.id}",
                      on_click=select_event, args=(ev.id,))


def render_player() -> None:
    sel = st.session_state.selected
    if not sel:
        return
    ev = STORE.get_event(sel["id"])
    if ev is None:
        st.warning("That event no longer exists.")
        st.session_state.selected = None
        return
    with st.container(border=True):
        top = st.columns([6, 1])
        top[0].markdown("**Selected event** " + event_header_html(ev), unsafe_allow_html=True)
        if top[1].button("Close", key="close_player"):
            st.session_state.selected = None
            st.rerun()
        src = ev.source_path
        t = max(0.0, (ev.media_time_s or 0.0) - 3.0)
        st.session_state.selected["t"] = t
        a, b = st.columns(2)
        with a:
            if src and Path(src).exists() and ev.media_time_s is not None:
                playable = browser_playable(src)
                st.video(src, start_time=int(t))
                if playable is False:
                    st.warning("This file's codec is probably not playable in browsers; use the frame preview or the "
                               "extracted clip below. Exact seeking is only verified for H.264 MP4 files.")
                elif playable is None:
                    st.info("Codec could not be determined; seeking accuracy is unverified for this file.")
                st.caption(f"Starts at {t:.0f}s (event at {ev.media_time_s:.1f}s into the file).")
            else:
                st.info("No source video is available for this event (live source or file moved). Showing the stored keyframe.")
        with b:
            kp = keyframe_path(ev)
            if src and Path(src).exists() and ev.media_time_s is not None:
                frame = extract_frame(src, ev.media_time_s)
                if frame is not None:
                    st.image(frame[:, :, ::-1], caption=f"Source frame at {ev.media_time_s:.1f}s (timestamp preview)", use_container_width=True)
                if st.button("Extract 8 s clip around the event", key="extract_clip"):
                    out, note = extract_clip(src, ev.media_time_s - 4, ev.media_time_s + 4, Path(PROF["paths"]["clips"]) / "extracted" / f"{ev.id}.mp4")
                    if out:
                        st.video(str(out))
                    st.caption(note)
            elif kp:
                st.image(str(kp), use_container_width=True)
        with st.expander("Event details"):
            st.json({"id": ev.id, "trigger": ev.trigger_type, "heuristic_reason": ev.meta.get("heuristic_reason"),
                     "vlm_reason": ev.meta.get("vlm_reason"), "objects": ev.meta.get("objects"), "vlm_model": ev.vlm_model,
                     "track_ids": ev.track_ids, "index_status": ev.index_status, "sample_count": ev.meta.get("sample_count"),
                     "t_trigger→t_stored_s": round(ev.t_stored - ev.t_trigger, 3) if ev.t_stored and ev.t_trigger else None})


# ------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## 🛡️ Vigil")
    st.caption("Local AI video event intelligence")
    show_rejected = st.checkbox("Show rejected candidates", value=False, help="Candidates the VLM rejected (debug/evaluation).")
    n_feed = st.slider("Events in feed", 5, 50, 15)
    st.markdown("### Live event feed")

    @st.fragment(run_every="2s")
    def event_feed() -> None:
        try:
            verdicts = ("verified", "unverified", "rejected") if show_rejected else ("verified", "unverified")
            evs = STORE.query(verdicts=verdicts, limit=n_feed)
        except Exception as e:  # noqa: BLE001
            st.error(f"Event database unavailable: {e}")
            return
        if not evs:
            st.info("No events yet. Start the pipeline or upload a video.")
        for ev in evs:
            with st.container(border=True):
                st.markdown(event_header_html(ev), unsafe_allow_html=True)
                st.markdown(f'<div class="{"vg-unver" if ev.verdict == "unverified" else ""}">{esc(ev.summary or "(no summary)")}</div>',
                            unsafe_allow_html=True)
                st.caption(f"score {ev.confidence_score:.2f} · VLM {ev.vlm_confidence:.2f}" + (" · ⚠ unverified" if ev.verdict == "unverified" else ""))
                with st.expander("keyframe"):
                    kp = keyframe_path(ev)
                    if kp:
                        st.image(str(kp), use_container_width=True)
                    else:
                        st.caption("keyframe missing")
                    st.button("Inspect", key=f"feed_{ev.id}", on_click=select_event, args=(ev.id,))

    event_feed()

st.title("Vigil")
status0 = read_status(PROF)
render_player()
tab_live, tab_chat, tab_metrics, tab_upload = st.tabs(["📹 Live", "💬 Chat search", "📊 Metrics", "⬆ Upload video"])


# ---------------------------------------------------------------------- live
with tab_live:
    cam_ids = list((status0 or {}).get("cameras", {}).keys()) or [c.id for c in B["cams"]]
    cam_sel = st.selectbox("Camera", cam_ids or ["(none)"], key="live_cam")

    @st.fragment(run_every="1s")
    def live_view(cam: str) -> None:
        status = read_status(PROF)
        alive = pipeline_alive(status)
        if not status:
            st.warning("Pipeline status not found. Start it with: `python -m vigil.pipeline --config configs/cameras.yaml`")
        elif not alive:
            st.warning("Pipeline is not running (status is stale). Events and search still work. Start it with: "
                       "`python -m vigil.pipeline --config configs/cameras.yaml`")
        left, right = st.columns([3, 2])
        with left:
            jpg = Path(PROF["paths"]["live"]) / f"{cam}.jpg"
            try:
                if jpg.exists() and (time.time() - jpg.stat().st_mtime < 3600):
                    st.image(jpg.read_bytes(), use_container_width=True, caption=f"{cam}: annotated frame (boxes, track ids, zones, entry line, trigger banners)")
                else:
                    st.info("No live frame yet for this camera.")
            except OSError:
                st.info("Live frame is being updated...")
        with right:
            if status:
                h = status.get("health", "unknown")
                cls = {"ok": "vg-ok", "degraded": "vg-warn"}.get(h, "vg-bad")
                st.markdown(f'Pipeline health: <span class="{cls}">{esc(h)}</span> · profile **{esc(status.get("profile"))}** · detector `{esc(status.get("detector_device"))}`',
                            unsafe_allow_html=True)
                c = status.get("cameras", {}).get(cam, {})
                if c:
                    st.markdown(f'Camera state: **{esc(c.get("state"))}** · FPS **{c.get("fps", 0)}** · queue {c.get("queue_depth")} · dropped frames {c.get("dropped_frames")}')
                    if c.get("error"):
                        st.error(c["error"])
                vs = status.get("vlm_state")
                lat = status.get("latest_vlm_latency_s")
                st.markdown(f'VLM `{esc(status.get("vlm_model"))}`: **{esc(vs)}** · latest latency **{lat if lat is not None else "n/a"} s** · '
                            f'queue {status.get("vlm_queue_depth")} (dropped triggers {status.get("dropped_triggers")})')
                if status.get("vlm_info", {}).get("error"):
                    st.warning(status["vlm_info"]["error"])
                if not status.get("embedder_ok", True):
                    st.warning("CLIP/ChromaDB unavailable in the pipeline: new events are stored without embeddings (search falls back to keywords).")
                st.markdown("**Events today by category**")
                today0 = datetime.now(ZoneInfo(TZ)).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
                try:
                    counts = STORE.counts_by_category(since=today0)
                    st.table({"category": list(CATEGORIES), "events": [counts.get(k, 0) for k in CATEGORIES]})
                except Exception as e:  # noqa: BLE001
                    st.error(f"count failed: {e}")

    live_view(cam_sel)


# ---------------------------------------------------------------------- chat
def run_query(text: str, mode: str) -> None:
    try:
        resp = search_engine(PROFILE_PATH).ask(text, mode)
        st.session_state.chat.append({"q": text, "resp": resp, "err": None})
    except Exception as e:  # noqa: BLE001
        st.session_state.chat.append({"q": text, "resp": None, "err": f"{type(e).__name__}: {e}"})


with tab_chat:
    st.caption("Ask in plain language. Camera/category/time filters are exact SQL filters; visual concepts are ranked with CLIP "
               "(image + summary embeddings) over the filtered events.")
    mode = st.radio("Search mode", ["hybrid", "image", "summary"], horizontal=True, key="mode",
                    help="hybrid = 0.6·image + 0.4·summary on normalised similarities (starting weights, not tuned).")
    with st.expander("Example queries"):
        cols = st.columns(2)
        for i, ex in enumerate(EXAMPLES):
            if cols[i % 2].button(ex, key=f"ex{i}"):
                run_query(ex, mode)
    prompt = st.chat_input("e.g. Show all tailgating events from camera 2 today")
    if prompt:
        run_query(prompt, mode)
    for i, turn in enumerate(st.session_state.chat[-6:]):
        with st.chat_message("user"):
            st.write(turn["q"])
        with st.chat_message("assistant"):
            if turn["err"]:
                st.error(f"Search failed: {turn['err']}")
                continue
            resp: SearchResponse = turn["resp"]
            for w in resp.warnings:
                st.warning(w)
            with st.expander("How I interpreted this"):
                st.json(resp.filter.to_dict(ZoneInfo(TZ)))
                st.caption(f"mode: {resp.mode} · {resp.n_candidates} candidate events after structured filters · latency ms: {resp.latency_ms}")
            if not resp.results:
                st.info("No matching events. Try widening the time range, or tick 'Show rejected candidates' in the sidebar "
                        "to include events the VLM rejected.")
            else:
                st.write(f"{len(resp.results)} result(s):")
                last_turn = i == len(st.session_state.chat[-6:]) - 1
                for r in resp.results:
                    render_card(r.event, r, key=f"{i}_{r.rank}" if last_turn else f"old{i}_{r.rank}")


# ------------------------------------------------------------------- metrics
with tab_metrics:
    @st.fragment(run_every="3s")
    def live_metrics() -> None:
        st.subheader("Live pipeline metrics (preliminary, current run)")
        status = read_status(PROF)
        if not status:
            st.info("No status file yet.")
            return
        m = status.get("metrics", {})
        stages = m.get("stages", {})
        if stages:
            st.table([{"stage": k, "n": v["n"], "p50 ms": v["p50_ms"], "p95 ms": v["p95_ms"], "last ms": v["last_ms"]} for k, v in stages.items()])
        mem = m.get("memory", {})
        c = st.columns(4)
        c[0].metric("Process RSS MB", f"{mem.get('rss_mb') or 0:.0f}")
        c[1].metric("Peak GPU MB (torch)", "n/a" if mem.get("torch_peak_mb") is None else f"{mem['torch_peak_mb']:.0f}")
        c[2].metric("GPU used MB (nvidia-smi)", "n/a" if mem.get("smi_used_mb") is None else f"{mem['smi_used_mb']:.0f}")
        c[3].metric("Dropped triggers", status.get("dropped_triggers", 0))
        st.caption(f"Verdicts this run: {status.get('verdict_counts')} · warm-up/model load (not steady-state): {m.get('warmup_s')}")
        for e in (m.get("errors") or [])[-5:]:
            st.caption(f"⚠ {fmt_ts(e['t'], TZ)} {e['source']}: {e['message']}")

    live_metrics()
    st.subheader("Historical latency (from stored events)")
    try:
        rows = STORE.timing_rows(500)
        import pandas as pd

        recs = []
        for r in rows:
            if r["t_stored"] and r["t_trigger"]:
                recs.append({"time": datetime.fromtimestamp(r["timestamp"], ZoneInfo(TZ)),
                             "trigger→store s": r["t_stored"] - r["t_trigger"],
                             "VLM s": (r["t_vlm_end"] - r["t_vlm_start"]) if r["t_vlm_end"] else None})
        if recs:
            df = pd.DataFrame(recs).sort_values("time").set_index("time")
            st.line_chart(df)
            st.caption(f"{len(df)} events; p50/p95 trigger→store: {df['trigger→store s'].quantile(.5):.2f}/{df['trigger→store s'].quantile(.95):.2f} s")
        else:
            st.info("No timed events yet.")
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read timing history: {e}")
    st.subheader("Offline evaluation (REPORT.md)")
    results = ROOT / "eval" / "results"
    report = ROOT / "REPORT.md"
    if report.exists():
        st.caption("Generated by eval/make_report.py: preliminary hackathon measurements, not real-world accuracy guarantees.")
        for png in sorted(results.glob("*.png")) if results.exists() else []:
            st.image(str(png), caption=png.stem.replace("_", " "))
        st.markdown(report.read_text(encoding="utf-8"))
    else:
        st.info("REPORT.md not found. Run `python eval/run_eval.py` then `python eval/make_report.py`.")


# -------------------------------------------------------------------- upload
with tab_upload:
    st.caption("Upload an MP4 to run it through the same pipeline (the running pipeline process picks it up; "
               "no model is loaded in this UI process).")
    up = st.file_uploader("Video file", type=["mp4", "mov", "avi", "mkv", "m4v"], key="uploader")
    cam_opts = [c.id for c in B["cams"]] or ["cam1"]
    c1, c2 = st.columns(2)
    cam_pick = c1.selectbox("Use zones / entry line of camera", cam_opts, key="up_cam")
    speed = c2.selectbox("Playback speed", [0, 1, 4], format_func=lambda s: {0: "as fast as possible", 1: "real time", 4: "4× real time"}[s], key="up_speed")
    if up is not None:
        data = up.getvalue()
        digest = hashlib.sha1(data).hexdigest()[:12]
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in up.name)[:80]
        dest = Path(PROF["paths"]["clips"]) / "uploads" / f"{digest}_{safe}"
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(data)
            os.replace(tmp, dest)
        info = probe_video(str(dest))
        if not info["ok"]:
            st.error(f"This video cannot be processed: {info['error']}.")
        else:
            st.success(f"{up.name}: {info['width']}×{info['height']}, {info['fps']:.1f} fps, {info['duration_s']:.1f} s, {info['frame_count']} frames")
            st.video(data)
            job_id = f"job_{digest}_{cam_pick}"
            jf = Path(PROF["paths"]["jobs"]) / f"{job_id}.json"
            existing = json.loads(jf.read_text()) if jf.exists() else None
            alive = pipeline_alive(read_status(PROF))
            if not alive:
                st.warning("The pipeline is not running, so the job would wait. Start it with `python -m vigil.pipeline --config configs/cameras.yaml`.")
            busy = existing and existing.get("state") in ("queued", "running")
            if st.button("Process through pipeline", disabled=bool(busy), key="process_btn"):
                jf.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_json(jf, {"job_id": job_id, "state": "queued", "camera_id": cam_pick, "source": str(dest),
                                       "speed": speed, "created_at": time.time(), "message": "waiting for the pipeline"})
                st.rerun()

            @st.fragment(run_every="2s")
            def job_status(path: str) -> None:
                p = Path(path)
                if p.exists():
                    try:
                        j = json.loads(p.read_text())
                        st.info(f"Job `{j['job_id']}`: **{j['state']}**: {j.get('message', '')}")
                    except Exception:  # noqa: BLE001
                        pass

            job_status(str(jf))
