#!/usr/bin/env python3
# mockllm: scriptable fake openai-compatible chat completions server.
# stdlib only. serves POST /v1/chat/completions and GET /v1/models
# on 127.0.0.1. requests with "stream": true get fake sse chunks.
# see README.md for the scenario format.

import argparse
import json
import random
import re
import sys
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_PORT = 8471
DEFAULT_MODEL = "mockllm"


# mutable scenario state shared across requests
class MockState:
    def __init__(self, steps, rng):
        self.steps = steps
        self.i = 0
        self.waiting_for_tools = False
        self.rng = rng
        self.call_seq = 0
        self.lock = threading.Lock()


def load_scenario(path):
    # scenario file is a step list, or {"steps": [...]}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("steps", [])
    if not isinstance(data, list):
        raise ValueError("scenario must be a list of steps or an object with a steps list")
    return data


def pick_reply(spec, rng):
    # reply can be one string or a list to choose from (seeded)
    if isinstance(spec, list):
        return rng.choice(spec) if spec else ""
    return spec


def echo_reply(body):
    # default mode: describe what the request contained
    msgs = body.get("messages") or []
    last = msgs[-1] if msgs else {}
    tools = body.get("tools") or []
    names = [t.get("function", {}).get("name", "?")
             for t in tools if isinstance(t, dict)]
    text = str(last.get("content", ""))[:200]
    return "mockllm echo: %d message(s), last=%s: %r, tools offered: %s" % (
        len(msgs), last.get("role", "?"), text,
        ", ".join(names) if names else "none")


def chat_payload(model, content, calls, finish):
    msg = {"role": "assistant", "content": content}
    if calls:
        msg["tool_calls"] = calls
    return {
        "id": "chatcmpl-mock-%d" % int(time.time() * 1000),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": msg,
                     "finish_reason": finish}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                  "total_tokens": 0},
    }


def error_payload(err, status):
    if isinstance(err.get("body"), dict):
        return err["body"]
    return {"error": {"message": err.get("message", "mock error %d" % status),
                      "type": "mockllm_error", "code": status}}


def stream_chunks(model, payload):
    # turn a chat completion into openai-style sse chunks. content goes
    # out word by word, tool calls ride one delta chunk, then the final
    # chunk carries the finish reason. keeps exact whitespace.
    msg = payload["choices"][0]["message"]
    head = {
        "id": payload["id"], "object": "chat.completion.chunk",
        "created": payload["created"], "model": model,
    }
    for piece in re.findall(r"\S+\s*|\s+", msg.get("content") or ""):
        yield dict(head, choices=[{"index": 0, "delta": {
            "role": "assistant", "content": piece},
            "finish_reason": None}])
    delta = {"role": "assistant"}
    if msg.get("tool_calls"):
        delta["tool_calls"] = msg["tool_calls"]
    finish = payload["choices"][0].get("finish_reason") or "stop"
    yield dict(head, choices=[{"index": 0, "delta": delta,
                               "finish_reason": finish}])


def decide(state, body, latency_default, model):
    # returns (status, payload, latency_ms). caller sleeps outside the lock.
    msgs = body.get("messages") or []
    has_results = any(isinstance(m, dict) and m.get("role") == "tool"
                      for m in msgs)
    if state.waiting_for_tools and has_results:
        state.i += 1
        state.waiting_for_tools = False
    step = state.steps[state.i] if state.i < len(state.steps) else None
    if step and "error" in step:
        err = step["error"] or {}
        if err.get("once"):
            state.i += 1
        return int(err.get("status", 500)), error_payload(err, int(err.get("status", 500))), 0
    content, calls = None, None
    if step:
        if "reply" in step:
            content = pick_reply(step["reply"], state.rng)
        for tc in step.get("tool_calls") or []:
            state.call_seq += 1
            calls = calls or []
            calls.append({
                "id": "call_%04d" % state.call_seq,
                "type": "function",
                "function": {
                    "name": tc.get("name", ""),
                    "arguments": json.dumps(tc.get("arguments", {})),
                },
            })
    if calls:
        # stay on this step until the client sends tool results back
        state.waiting_for_tools = True
    elif step is not None:
        state.i += 1
    latency = latency_default
    if step and "latency_ms" in step:
        latency = step["latency_ms"]
    if content is None and calls is None:
        content = echo_reply(body)
    finish = "tool_calls" if calls else "stop"
    return 200, chat_payload(model, content, calls, finish), latency


class Handler(BaseHTTPRequestHandler):
    server_version = "mockllm"

    def log_message(self, fmt, *args):
        pass  # stay quiet; --record covers inspection

    def send_json(self, status, obj):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_stream(self, model, payload):
        # fake sse: chunk events, then data: [DONE], then close
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for chunk in stream_chunks(model, payload):
            self.wfile.write(
                ("data: %s\n\n" % json.dumps(chunk)).encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            m = self.server.mock
            self.send_json(200, {"object": "list", "data": [{
                "id": m.model, "object": "model",
                "created": int(time.time()), "owned_by": "mockllm"}]})
        else:
            self.send_json(404, {"error": {"message": "not found",
                                           "type": "not_found"}})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            self.send_json(404, {"error": {"message": "not found",
                                           "type": "not_found"}})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except (ValueError, UnicodeDecodeError):
            self.send_json(400, {"error": {
                "message": "invalid json", "type": "invalid_request"}})
            return
        if not isinstance(body, dict):
            self.send_json(400, {"error": {
                "message": "body must be a json object",
                "type": "invalid_request"}})
            return
        m = self.server.mock
        if m.record_path:
            with m.record_lock:
                with open(m.record_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"path": self.path, "body": body})
                            + "\n")
        with m.state.lock:
            status, payload, latency = decide(m.state, body, m.latency_ms,
                                             m.model)
        if latency:
            time.sleep(latency / 1000.0)
        if body.get("stream") and status == 200:
            # streaming clients get sse; errors stay plain json
            self.send_stream(m.model, payload)
        else:
            self.send_json(status, payload)


def make_server(port, state, record_path=None, latency_ms=0, model=DEFAULT_MODEL):
    # binds 127.0.0.1 only, never public
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    server.mock = types.SimpleNamespace(
        state=state, record_path=record_path, latency_ms=latency_ms,
        model=model, record_lock=threading.Lock())
    return server


def main(argv=None):
    p = argparse.ArgumentParser(
        description="mockllm: scriptable fake openai-compatible chat server")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--scenario", default=None,
                   help="json file with an ordered list of scripted steps")
    p.add_argument("--record", default=None,
                   help="append every request body as jsonl to FILE")
    p.add_argument("--latency", type=int, default=0,
                   help="default artificial latency per reply, in ms")
    p.add_argument("--seed", type=int, default=None,
                   help="seed for deterministic random choices")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="model id reported in replies")
    args = p.parse_args(argv)
    try:
        steps = load_scenario(args.scenario) if args.scenario else []
    except (OSError, ValueError) as e:
        print("mockllm: bad scenario file: %s" % e, file=sys.stderr)
        return 2
    state = MockState(steps, random.Random(args.seed))
    server = make_server(args.port, state, args.record, args.latency,
                         args.model)
    print("mockllm on http://127.0.0.1:%d/ (model %s)"
          % (server.server_address[1], args.model))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
