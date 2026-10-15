#!/usr/bin/env python
"""MOCK Ollama server for UI/pipeline testing without a model. NOT a VLM.

It answers /api/version, /api/tags, /api/ps and /api/chat with canned JSON whose
summary starts with "[MOCK VLM]" so the output can never be mistaken for a real
model result. Use only to exercise the application plumbing.

  python scripts/mock_ollama.py --port 11500      # then set VIGIL_OLLAMA_URL=http://127.0.0.1:11500
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/api/version":
            self._send({"version": "0.9.0-mock"})
        elif self.path == "/api/tags":
            self._send({"models": [{"name": self.server.model, "size": 1_700_000_000}]})
        else:
            self._send({"models": []})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        user = " ".join(m.get("content", "") for m in body.get("messages", []) if m["role"] == "user")
        if "images" not in json.dumps(body.get("messages", [])):  # text-only (query parser)
            self._send({"message": {"content": json.dumps({"category": None, "camera_id": None, "time_from": None,
                                                           "time_to": None, "visual_query": None, "limit": 20})}})
            return
        cat = "none"
        for c in ("person down", "fighting", "tailgating", "crowd incident", "suspicious activity"):
            if f'claims: "{c}"' in user:
                cat = c if self.server.mode == "agree" else "none"
        self._send({"message": {"content": json.dumps({
            "category": cat, "vlm_confidence": 0.75, "summary": f"[MOCK VLM] canned answer for a {cat} candidate",
            "objects": ["person"], "reason": "mock server: no real analysis"})}})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=11500)
    ap.add_argument("--model", default="moondream:latest")
    ap.add_argument("--mode", choices=["agree", "reject"], default="agree")
    a = ap.parse_args()
    srv = HTTPServer(("127.0.0.1", a.port), H)
    srv.model, srv.mode = a.model, a.mode
    print(f"MOCK Ollama on http://127.0.0.1:{a.port} (canned answers, mode={a.mode})")
    srv.serve_forever()
