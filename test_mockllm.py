#!/usr/bin/env python3
# tests for mockllm. spins up the real server on 127.0.0.1,
# no external network, no api keys. run with: python3 test_mockllm.py

import json
import os
import random
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mockllm

SERVERS = []


def serve(steps=None, **kw):
    state = mockllm.MockState(steps or [], random.Random(kw.get("seed")))
    server = mockllm.make_server(
        0, state, record_path=kw.get("record"),
        latency_ms=kw.get("latency", 0),
        model=kw.get("model", "mockllm"))
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    SERVERS.append(server)
    return "http://127.0.0.1:%d" % server.server_address[1]


def post(base, body, path="/v1/chat/completions"):
    req = urllib.request.Request(
        base + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=10) as r:
        return r.status, json.loads(r.read().decode())


def chat(msgs, **extra):
    body = {"model": "mockllm", "messages": msgs}
    body.update(extra)
    return body


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_models():
    base = serve()
    status, data = get(base, "/v1/models")
    check(status == 200, "models status %r" % status)
    check(data["object"] == "list", "models object")
    check(data["data"][0]["id"] == "mockllm", "model id")


def test_custom_model_name():
    base = serve(model="tiny-test")
    status, data = get(base, "/v1/models")
    check(data["data"][0]["id"] == "tiny-test", "custom model in list")
    status, data = post(base, chat([{"role": "user", "content": "hi"}],
                                   model="tiny-test"))
    check(data["model"] == "tiny-test", "custom model in reply")


def test_echo_default():
    base = serve()
    status, data = post(base, chat([
        {"role": "user", "content": "hello world"}],
        tools=[{"type": "function",
                "function": {"name": "read_file"}}]))
    check(status == 200, "echo status")
    text = data["choices"][0]["message"]["content"]
    check("hello world" in text, "echo repeats user text: %r" % text)
    check("read_file" in text, "echo lists offered tools: %r" % text)
    check(data["choices"][0]["finish_reason"] == "stop", "finish reason")


def test_scripted_reply_then_echo():
    base = serve([{"reply": "scripted answer"}])
    status, data = post(base, chat([{"role": "user", "content": "q"}]))
    check(data["choices"][0]["message"]["content"] == "scripted answer",
          "scripted reply")
    status, data = post(base, chat([{"role": "user", "content": "q2"}]))
    check("mockllm echo" in data["choices"][0]["message"]["content"],
          "falls back to echo when steps run out")


def test_tool_call_sequence():
    steps = [
        {"reply": "checking", "tool_calls": [
            {"name": "read_file", "arguments": {"path": "a.py"}}]},
        {"reply": "got it"},
    ]
    base = serve(steps)
    first = chat([{"role": "user", "content": "read a.py"}])
    status, data = post(base, first)
    msg = data["choices"][0]["message"]
    check(data["choices"][0]["finish_reason"] == "tool_calls",
          "tool_calls finish reason")
    check(len(msg["tool_calls"]) == 1, "one tool call")
    fn = msg["tool_calls"][0]["function"]
    check(fn["name"] == "read_file", "tool name")
    check(json.loads(fn["arguments"]) == {"path": "a.py"}, "tool args")
    call_id = msg["tool_calls"][0]["id"]
    # no tool results yet: same step repeats
    status, data = post(base, first)
    check(data["choices"][0]["finish_reason"] == "tool_calls",
          "step waits for tool results")
    # send results back: scenario advances
    with_results = chat([
        {"role": "user", "content": "read a.py"},
        {"role": "assistant", "content": "checking",
         "tool_calls": msg["tool_calls"]},
        {"role": "tool", "tool_call_id": call_id,
         "content": "file contents"},
    ])
    status, data = post(base, with_results)
    check(data["choices"][0]["message"]["content"] == "got it",
          "advanced after tool results")


def test_error_once_then_continue():
    steps = [
        {"error": {"status": 429, "message": "slow down", "once": True}},
        {"reply": "recovered"},
    ]
    base = serve(steps)
    body = chat([{"role": "user", "content": "hi"}])
    status, data = post(base, body)
    check(status == 429, "first request errors, got %r" % status)
    check("slow down" in json.dumps(data), "error message passes through")
    status, data = post(base, body)
    check(status == 200, "second request ok")
    check(data["choices"][0]["message"]["content"] == "recovered",
          "continues after once error")


def test_error_repeats_without_once():
    base = serve([{"error": {"status": 500, "message": "boom"}}])
    body = chat([{"role": "user", "content": "hi"}])
    for _ in range(2):
        status, _ = post(base, body)
        check(status == 500, "persistent error, got %r" % status)


def test_custom_error_body():
    custom = {"error": {"message": "custom", "code": "weird"}}
    base = serve([{"error": {"status": 503, "body": custom}}])
    status, data = post(base, chat([{"role": "user", "content": "hi"}]))
    check(status == 503, "custom status")
    check(data == custom, "custom body passes through verbatim")


def test_per_step_latency():
    base = serve([{"reply": "slow", "latency_ms": 300}])
    start = time.time()
    status, data = post(base, chat([{"role": "user", "content": "hi"}]))
    elapsed = time.time() - start
    check(status == 200, "latency reply ok")
    check(elapsed >= 0.29, "latency applied, only %.2fs" % elapsed)
    check(elapsed < 5, "latency not absurd: %.2fs" % elapsed)


def test_global_latency():
    base = serve(latency=150)
    start = time.time()
    post(base, chat([{"role": "user", "content": "hi"}]))
    check(time.time() - start >= 0.14, "global latency applied")


def test_record():
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    try:
        base = serve(record=path)
        post(base, chat([{"role": "user", "content": "one"}]))
        post(base, chat([{"role": "user", "content": "two"}]))
        lines = open(path, encoding="utf-8").read().strip().split("\n")
        check(len(lines) == 2, "two recorded lines, got %d" % len(lines))
        bodies = [json.loads(l)["body"] for l in lines]
        check(bodies[0]["messages"][0]["content"] == "one", "first body")
        check(bodies[1]["messages"][0]["content"] == "two", "second body")
    finally:
        os.unlink(path)


def test_seed_deterministic_choice():
    steps = [{"reply": ["a", "b", "c", "d"]}]
    b1 = serve([json.loads(json.dumps(s)) for s in steps], seed=7)
    b2 = serve([json.loads(json.dumps(s)) for s in steps], seed=7)
    _, d1 = post(b1, chat([{"role": "user", "content": "pick"}]))
    _, d2 = post(b2, chat([{"role": "user", "content": "pick"}]))
    r1 = d1["choices"][0]["message"]["content"]
    r2 = d2["choices"][0]["message"]["content"]
    check(r1 == r2, "same seed gives same choice: %r vs %r" % (r1, r2))


def test_bad_json_is_400():
    base = serve()
    req = urllib.request.Request(
        base + "/v1/chat/completions", data=b"{not json",
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        urllib.request.urlopen(req, timeout=10)
        raise AssertionError("expected 400")
    except urllib.error.HTTPError as e:
        check(e.code == 400, "bad json gives 400, got %r" % e.code)


def test_unknown_paths_404():
    base = serve()
    try:
        get(base, "/nope")
        raise AssertionError("expected 404")
    except urllib.error.HTTPError as e:
        check(e.code == 404, "unknown get 404")
    status, _ = post(base, chat([]), path="/v1/embeddings")
    check(status == 404, "unknown post path 404")


def test_load_scenario_file():
    fd, path = tempfile.mkstemp(suffix=".json")
    try:
        os.write(fd, json.dumps(
            {"steps": [{"reply": "from file"}]}).encode())
        os.close(fd)
        steps = mockllm.load_scenario(path)
        check(steps == [{"reply": "from file"}], "loads steps object")
        base = serve(steps)
        _, data = post(base, chat([{"role": "user", "content": "hi"}]))
        check(data["choices"][0]["message"]["content"] == "from file",
              "file scenario drives replies")
    finally:
        os.unlink(path)


def test_load_scenario_rejects_junk():
    fd, path = tempfile.mkstemp(suffix=".json")
    try:
        os.write(fd, b'{"steps": "nope"}')
        os.close(fd)
        try:
            mockllm.load_scenario(path)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    finally:
        os.unlink(path)


def main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_")]
    passed, failed = 0, 0
    try:
        for t in tests:
            try:
                t()
                passed += 1
                print("ok %s" % t.__name__)
            except Exception as e:
                failed += 1
                print("FAIL %s: %r" % (t.__name__, e))
    finally:
        for s in SERVERS:
            s.shutdown()
            s.server_close()
    print("%d passed, %d failed" % (passed, failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
