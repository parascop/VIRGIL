"""Exercises ClipEmbedder's real code path (preprocessing, batching, normalisation,
dimension checks, RPC) with a RANDOMLY INITIALISED ViT-B/32. The pretrained OpenAI
weights need a network download, so retrieval *quality* is NOT tested here; only that
the plumbing around the model is correct."""
import threading

import numpy as np
import pytest

open_clip = pytest.importorskip("open_clip")

from vigil.embeddings import ClipEmbedder, EmbedRpcServer, EmbedderUnavailable, RpcTextEmbedder, VectorIndex


@pytest.fixture(scope="module")
def emb():
    class RandomClip(ClipEmbedder):
        def _create(self, oc):  # no download: random weights, real architecture + real preprocessing
            m, _, pre = oc.create_model_and_transforms("ViT-B-32", pretrained=None, device="cpu")
            return m, pre

    e = RandomClip("ViT-B-32", "none", "cpu")
    e.load()
    return e


def test_shapes_normalisation_and_determinism(emb):
    img = np.random.RandomState(0).randint(0, 255, (120, 200, 3), np.uint8)  # BGR array
    v = emb.embed_images([img, img[::-1]])
    t = emb.embed_texts(["a person wearing a red shirt", "an empty hallway"])
    assert v.shape == (2, 512) and t.shape == (2, 512) and emb.dim == 512
    assert np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-4) and np.allclose(np.linalg.norm(t, axis=1), 1, atol=1e-4)
    assert np.allclose(emb.embed_images([img]), v[:1], atol=1e-5)
    assert not np.allclose(v[0], v[1])


def test_long_text_is_truncated_not_crashing(emb):
    assert emb.embed_texts(["word " * 400]).shape == (1, 512)


def test_missing_weights_raise_clear_error(tmp_path):
    class Missing(ClipEmbedder):
        def _create(self, oc):
            raise FileNotFoundError("no network and no cache")

    with pytest.raises(EmbedderUnavailable, match="download_models"):
        Missing().load()


def test_rpc_text_embedding_roundtrip(emb, tmp_path):
    stop = threading.Event()
    srv = EmbedRpcServer(emb, tmp_path / "rpc", stop)
    srv.start()
    try:
        client = RpcTextEmbedder(tmp_path / "rpc", timeout_s=10)
        v = client.embed_texts(["red shirt"])
        assert v.shape == (1, 512) and np.allclose(v, emb.embed_texts(["red shirt"]), atol=1e-5)
    finally:
        stop.set()
        srv.join(2)
    with pytest.raises(EmbedderUnavailable):
        RpcTextEmbedder(tmp_path / "rpc2", timeout_s=0.3).embed_texts(["x"])  # nobody answering


def test_index_accepts_clip_dim_and_rejects_wrong_dim(emb, tmp_path):
    idx = VectorIndex(tmp_path / "c", emb.tag, emb.dim)
    v = emb.embed_texts(["a", "b"])
    idx.upsert("e1", v[0], v[1], {"camera_id": "c"})
    assert idx.counts() == {"image": 1, "summary": 1}
    with pytest.raises(ValueError):
        idx.upsert("e2", np.ones(64, np.float32), None, {"camera_id": "c"})
