"""CLIP embeddings (open-clip-torch, ViT-B/32) + ChromaDB vector index.

* ``ClipEmbedder``: image and text encoders sharing one semantic space; models
  are loaded lazily (never at import), once per process, behind a lock.
* ``VectorIndex``: persistent ChromaDB, two cosine collections (keyframe image
  vectors, summary-text vectors) keyed by the SQLite ``event_id``. SQLite stays
  the source of truth; the index can be rebuilt (`python -m vigil.embeddings
  --rebuild`).
* ``EmbedRpcServer`` / ``RpcTextEmbedder``: lets the Streamlit process obtain
  query-text embeddings from the pipeline process through small files so the
  UI does not need a second copy of CLIP while the pipeline runs.
* ``HashEmbedder``: deterministic **test stub**. It carries no semantics and is
  only ever selected explicitly (unit tests); results using it are not real.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # no network calls from ChromaDB

log = logging.getLogger("vigil.embeddings")

IMAGE_COLLECTION = "vigil_image"
SUMMARY_COLLECTION = "vigil_summary"


def _normalize(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.maximum(n, 1e-12)


class EmbedderUnavailable(RuntimeError):
    pass


class ClipEmbedder:
    is_stub = False

    def __init__(self, model: str = "ViT-B-32", pretrained: str = "openai", device: str = "cpu",
                 cache_dir: Optional[str] = None):
        self.model_name, self.pretrained, self.device, self.cache_dir = model, pretrained, device, cache_dir
        self._model = self._preprocess = self._tok = None
        self._lock = threading.Lock()
        self.dim: Optional[int] = None
        self.load_seconds = 0.0

    @property
    def tag(self) -> str:
        return f"{self.model_name}/{self.pretrained}"

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            t0 = time.time()
            try:
                import open_clip
                import torch

                if self.cache_dir:
                    Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
                model, pre = self._create(open_clip)
                model.eval()
                self._model, self._preprocess = model, pre
                self._tok = open_clip.get_tokenizer(self.model_name)
                with torch.no_grad():
                    probe = model.encode_text(self._tok(["probe"]).to(self.device))
                self.dim = int(probe.shape[-1])
            except Exception as e:  # noqa: BLE001
                self._model = None
                raise EmbedderUnavailable(
                    f"CLIP {self.tag} could not be loaded ({type(e).__name__}: {e}). The weights are downloaded "
                    "on first use; run scripts/download_models.* once while online.") from e
            self.load_seconds = time.time() - t0
            log.info("CLIP %s on %s ready in %.1fs (dim=%s)", self.tag, self.device, self.load_seconds, self.dim)

    def _create(self, open_clip: Any):
        """Prefer the local cache (works fully offline, no slow network retries);
        only if the checkpoint is not cached yet, try downloading it."""
        last: Optional[Exception] = None
        try:
            from huggingface_hub import constants as hfc
        except Exception:  # noqa: BLE001
            hfc = None
        prev_env, prev_const = os.environ.get("HF_HUB_OFFLINE"), getattr(hfc, "HF_HUB_OFFLINE", None)
        try:
            for offline in (True, False):
                if offline:
                    os.environ["HF_HUB_OFFLINE"] = "1"
                    if hfc is not None:
                        hfc.HF_HUB_OFFLINE = True
                else:
                    if prev_env is None:
                        os.environ.pop("HF_HUB_OFFLINE", None)
                    else:
                        os.environ["HF_HUB_OFFLINE"] = prev_env
                    if hfc is not None:
                        hfc.HF_HUB_OFFLINE = bool(prev_const)
                try:
                    model, _, pre = open_clip.create_model_and_transforms(
                        self.model_name, pretrained=self.pretrained, device=self.device, cache_dir=self.cache_dir)
                    return model, pre
                except Exception as e:  # noqa: BLE001
                    last = e
        finally:
            if prev_env is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = prev_env
            if hfc is not None and prev_const is not None:
                hfc.HF_HUB_OFFLINE = prev_const
        raise last  # type: ignore[misc]

    def embed_images(self, images: Sequence[Any]) -> np.ndarray:
        """images: BGR numpy arrays or PIL images -> (n, dim) L2-normalised."""
        self.load()
        import torch
        from PIL import Image

        tensors = []
        for im in images:
            if isinstance(im, np.ndarray):
                im = Image.fromarray(im[:, :, ::-1])
            tensors.append(self._preprocess(im.convert("RGB")))
        with self._lock, torch.no_grad():
            feats = self._model.encode_image(torch.stack(tensors).to(self.device))
        return _normalize(feats.float().cpu().numpy())

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        self.load()
        import torch

        with self._lock, torch.no_grad():
            feats = self._model.encode_text(self._tok(list(texts)).to(self.device))
        return _normalize(feats.float().cpu().numpy())


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder. TEST STUB, no real vision."""

    is_stub = True
    tag = "hash-stub/none"
    dim = 64

    def _vec(self, tokens: List[str]) -> np.ndarray:
        v = np.zeros(self.dim, np.float32)
        for t in tokens:
            h = int(hashlib.md5(t.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        return v if v.any() else np.ones(self.dim, np.float32)

    def load(self) -> None:
        return None

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        return _normalize(np.stack([self._vec(t.lower().split()) for t in texts]))

    def embed_images(self, images: Sequence[Any]) -> np.ndarray:
        out = []
        for im in images:
            a = np.asarray(im)
            m = a.reshape(-1, a.shape[-1]).mean(0) if a.ndim == 3 else np.array([a.mean()])
            tokens = [f"c{int(x) // 32}_{i}" for i, x in enumerate(m)]
            out.append(self._vec(tokens))
        return _normalize(np.stack(out))


class VectorIndex:
    """ChromaDB persistent index; embeddings are retrieved by id and ranked
    exactly (cosine) among the candidates chosen by the SQLite filter."""

    def __init__(self, path: str | Path, embed_tag: str = "", dim: Optional[int] = None):
        import chromadb
        from chromadb.config import Settings

        self.path = str(path)
        Path(self.path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self.path, settings=Settings(anonymized_telemetry=False))
        self.embed_tag = embed_tag
        self.dim = dim
        self._lock = threading.Lock()
        self._cols: Dict[str, Any] = {}

    def _col(self, name: str):
        c = self._cols.get(name)
        if c is None:
            c = self._client.get_or_create_collection(
                name=name, metadata={"hnsw:space": "cosine", "embed_tag": self.embed_tag})
            self._cols[name] = c
        return c

    def reset(self) -> None:
        with self._lock:
            for n in (IMAGE_COLLECTION, SUMMARY_COLLECTION):
                try:
                    self._client.delete_collection(n)
                except Exception:  # noqa: BLE001 - missing collection is fine
                    pass
            self._cols.clear()

    def counts(self) -> Dict[str, int]:
        try:
            return {"image": self._col(IMAGE_COLLECTION).count(), "summary": self._col(SUMMARY_COLLECTION).count()}
        except Exception as e:  # noqa: BLE001
            log.error("vector index unavailable: %s", e)
            return {"image": -1, "summary": -1}

    def upsert(self, event_id: str, image_vec: Optional[np.ndarray], text_vec: Optional[np.ndarray],
               meta: Dict[str, Any]) -> None:
        with self._lock:
            for name, vec in ((IMAGE_COLLECTION, image_vec), (SUMMARY_COLLECTION, text_vec)):
                if vec is None:
                    continue
                v = np.asarray(vec, np.float32).reshape(-1)
                if self.dim is not None and v.shape[0] != self.dim:
                    raise ValueError(f"embedding dimension {v.shape[0]} != expected {self.dim}")
                if not np.all(np.isfinite(v)):
                    raise ValueError("embedding contains NaN/inf")
                self._col(name).upsert(ids=[event_id], embeddings=[v.tolist()], metadatas=[meta])

    def scores(self, kind: str, query_vec: np.ndarray, ids: Sequence[str], batch: int = 400) -> Dict[str, float]:
        """Cosine similarity of ``query_vec`` to the stored vectors of ``ids``.
        Ids without a vector are simply absent from the result."""
        col = self._col(IMAGE_COLLECTION if kind == "image" else SUMMARY_COLLECTION)
        q = _normalize(np.asarray(query_vec, np.float32).reshape(1, -1))[0]
        out: Dict[str, float] = {}
        ids = list(ids)
        for i in range(0, len(ids), batch):
            chunk = ids[i:i + batch]
            got = col.get(ids=chunk, include=["embeddings"])
            gids = got.get("ids", [])
            embs = got.get("embeddings")
            if embs is None or len(gids) == 0:
                continue
            m = _normalize(np.asarray(embs, np.float32))
            if m.shape[1] != q.shape[0]:
                raise ValueError(f"stored embedding dim {m.shape[1]} != query dim {q.shape[0]} (rebuild the index)")
            for gid, s in zip(gids, m @ q):
                out[gid] = float(s)
        return out


# ------------------------------------------------------------- file RPC (UI)
class EmbedRpcServer(threading.Thread):
    """Runs in the pipeline process; answers text-embedding requests that the
    Streamlit process drops into ``<dir>/req``."""

    def __init__(self, embedder: Any, rpc_dir: str | Path, stop: threading.Event):
        super().__init__(name="embed-rpc", daemon=True)
        self.embedder, self.stop_event = embedder, stop
        self.req = Path(rpc_dir) / "req"
        self.res = Path(rpc_dir) / "res"
        self.req.mkdir(parents=True, exist_ok=True)
        self.res.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        from .util import atomic_write_json

        while not self.stop_event.wait(0.05):
            try:
                for f in list(self.req.glob("*.json")):
                    try:
                        text = json.loads(f.read_text(encoding="utf-8"))["text"]
                        vec = self.embedder.embed_texts([text])[0]
                        atomic_write_json(self.res / f.name, {"vec": [float(x) for x in vec]})
                    except Exception as e:  # noqa: BLE001
                        atomic_write_json(self.res / f.name, {"error": str(e)})
                    finally:
                        try:
                            f.unlink()
                        except OSError:
                            pass
                for old in self.res.glob("*.json"):  # garbage-collect unclaimed answers
                    if time.time() - old.stat().st_mtime > 60:
                        old.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                log.exception("embed rpc loop error")


class RpcTextEmbedder:
    """UI-side text embedder that asks the pipeline process (no model here)."""

    is_stub = False
    tag = "rpc"

    def __init__(self, rpc_dir: str | Path, timeout_s: float = 5.0):
        self.req = Path(rpc_dir) / "req"
        self.res = Path(rpc_dir) / "res"
        self.timeout_s = timeout_s

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        from .util import atomic_write_json

        vecs = []
        for t in texts:
            rid = uuid.uuid4().hex + ".json"
            self.req.mkdir(parents=True, exist_ok=True)
            atomic_write_json(self.req / rid, {"text": t})
            deadline = time.time() + self.timeout_s
            while time.time() < deadline:
                p = self.res / rid
                if p.exists():
                    d = json.loads(p.read_text(encoding="utf-8"))
                    p.unlink(missing_ok=True)
                    if "error" in d:
                        raise EmbedderUnavailable(d["error"])
                    vecs.append(np.asarray(d["vec"], np.float32))
                    break
                time.sleep(0.03)
            else:
                (self.req / rid).unlink(missing_ok=True)
                raise EmbedderUnavailable("pipeline did not answer the embedding request in time")
        return np.stack(vecs)

    def embed_images(self, images: Sequence[Any]) -> np.ndarray:  # pragma: no cover
        raise EmbedderUnavailable("image embedding is done by the pipeline process")


def make_embedder(profile: Dict[str, Any], device: Optional[str] = None) -> ClipEmbedder:
    e = profile["embeddings"]
    return ClipEmbedder(e["model"], e["pretrained"], device or e.get("device", "cpu"),
                        cache_dir=str(Path(profile["paths"]["cache"]) / "open_clip"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI: ``python -m vigil.embeddings --rebuild --profile configs/profile_mac_m4.yaml``"""
    import argparse

    from .config import load_profile
    from .storage import EventStore, rebuild_index, reindex_pending
    from .util import setup_logging

    ap = argparse.ArgumentParser(description="Maintain the ChromaDB index (SQLite is the source of truth)")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--rebuild", action="store_true", help="drop and rebuild the whole vector index")
    ap.add_argument("--repair", action="store_true", help="only index events that are not yet indexed")
    ap.add_argument("--stub", action="store_true", help="use the deterministic TEST stub embedder")
    a = ap.parse_args(argv)
    prof = load_profile(a.profile)
    setup_logging(prof["logging"]["level"])
    emb: Any = HashEmbedder() if a.stub else make_embedder(prof)
    emb.load()
    store = EventStore(prof["paths"]["sqlite"])
    idx = VectorIndex(prof["paths"]["chroma"], emb.tag, emb.dim)
    if a.rebuild:
        print(json.dumps(rebuild_index(store, emb, idx, prof["paths"]["keyframes"]), indent=1))
    elif a.repair:
        print(f"repaired {reindex_pending(store, emb, idx, prof['paths']['keyframes'])} events")
    else:
        print(json.dumps(idx.counts()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
