#!/usr/bin/env python3
"""Confirm a survivor in a robot camera picture with a vision LLM (runs on the Mac).

The robot sends a JPEG; this asks Cohere's vision model whether the picture shows one of the
mission's survivor stand-ins and answers in JSON. The API key stays on the Mac (config.json,
cohere_api_key); the robots never see it. (The OpenAI key had no credit left on 2026-11-24.)

  python3 survivor_vision.py picture.jpg            # one picture, prints the verdict
  python3 survivor_vision.py --serve 8770           # HTTP service for mappo_agent.py:
      POST /confirm?robot=TURTLEBOT3_1&x=..&y=..&yaw=..  (body = JPEG)  -> verdict JSON
"""

from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
MODEL = "command-a-vision-07-2025"
RECORD_DIR = HERE / "rl_point_nav" / "results" / "survivor_pictures"

# What the survivors look like in this drill (the camera picture is taken ~0.5 m away, ~10 cm above the floor).
SURVIVORS = {
    "BOB": "a yellow package or bag on the floor with a cartoon face and the name BOB drawn on it in black marker",
    "BOB2": "a red bean bag lying on the floor",
}
PROMPT = (
    "You are the vision module of a search-and-rescue robot in a disaster drill. The picture was taken by a "
    "small ground robot's front camera, close to the floor. In this drill two survivors are represented by objects: "
    + "; ".join(f"{name} = {text}" for name, text in SURVIVORS.items())
    + ". Fallen office chairs, boxes, bins, desks and other furniture are rubble, not survivors. Look for EACH "
    "survivor separately and judge only by what is visible (colour and shape). Answer ONLY with JSON: "
    "{\"BOB\": {\"visible\": true|false, \"confidence\": 0.0-1.0, \"position_in_image\": \"left\"|\"center\"|\"right\"|null}, "
    "\"BOB2\": {\"visible\": true|false, \"confidence\": 0.0-1.0, \"position_in_image\": \"left\"|\"center\"|\"right\"|null}, "
    "\"reason\": \"one short sentence\"}"
)


def summarize(answer: dict) -> dict:
    """Per-survivor answer -> verdict: the most confident visible survivor (confidence >= 0.6)."""
    seen = [(float(answer.get(n, {}).get("confidence") or 0.0), n) for n in SURVIVORS
            if isinstance(answer.get(n), dict) and answer[n].get("visible")]
    best = max(seen, default=(0.0, None))
    which = best[1] if best[0] >= 0.6 else None
    return {"survivor": which is not None, "which": which, "confidence": best[0],
            "position_in_image": answer[which].get("position_in_image") if which else None,
            "seen": {n: answer.get(n) for n in SURVIVORS}, "reason": answer.get("reason", "")}


def api_key() -> str:
    return json.loads((REPO / "config.json").read_text())["cohere_api_key"]


def confirm(jpeg: bytes) -> dict:
    body = {
        "model": MODEL,
        "temperature": 0,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}},
        ]}],
    }
    request = urllib.request.Request(
        "https://api.cohere.com/v2/chat", data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key()}"})
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=60) as response:
        reply = json.loads(response.read())
    text = reply["message"]["content"][0]["text"]
    verdict = summarize(json.loads(text[text.index("{"):text.rindex("}") + 1]))
    verdict["model"] = MODEL
    verdict["seconds"] = round(time.monotonic() - started, 1)
    return verdict


def serve(port: int) -> None:
    RECORD_DIR.mkdir(parents=True, exist_ok=True)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            url = urllib.parse.urlparse(self.path)
            if url.path != "/confirm":
                self.send_error(404)
                return
            query = dict(urllib.parse.parse_qsl(url.query))
            jpeg = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            stamp = time.strftime("%Y%m%d_%H%M%S")
            name = f"{stamp}_{query.get('robot', 'robot')}"
            (RECORD_DIR / f"{name}.jpg").write_bytes(jpeg)
            try:
                verdict = confirm(jpeg)
            except Exception as exc:  # report, keep serving
                verdict = {"survivor": False, "which": None, "confidence": 0.0, "reason": f"vision error: {exc}"}
            verdict.update(picture=f"{name}.jpg", robot_pose=query)
            (RECORD_DIR / f"{name}.json").write_text(json.dumps(verdict, indent=1))
            print(f"[SURVIVOR_VISION] {name}: {json.dumps(verdict)}", flush=True)
            data = json.dumps(verdict).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    print(f"[SURVIVOR_VISION] {MODEL} on port {port}; pictures and verdicts in {RECORD_DIR}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("picture", nargs="?", type=Path)
    parser.add_argument("--serve", type=int, metavar="PORT")
    args = parser.parse_args()
    if args.serve:
        serve(args.serve)
    elif args.picture:
        print(json.dumps(confirm(args.picture.read_bytes()), indent=1))
    else:
        parser.error("give a picture or --serve PORT")


if __name__ == "__main__":
    main()
