# mockllm

A scriptable fake OpenAI-compatible chat completions server for testing
agents. Every agent project hand-rolls a fake model server in its tests;
this is the reusable one. Stdlib only, no dependencies, no API keys,
no external network. It binds 127.0.0.1 only.

## quick start

```sh
python3 mockllm.py --port 8471
```

With no scenario it runs in echo mode: it replies with a description of
what it received (message count, last message, tools offered). Handy for
quick manual checks.

```sh
curl -s http://127.0.0.1:8471/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mockllm","messages":[{"role":"user","content":"hi"}]}'
```

`GET /v1/models` lists the configured model id.

## scripted scenarios

Pass `--scenario FILE` with a JSON file holding an ordered list of steps.
Each step is served in order:

```json
{
  "steps": [
    {
      "reply": "Let me look that up.",
      "tool_calls": [
        {"name": "read_file", "arguments": {"path": "main.py"}}
      ]
    },
    {"reply": "The file has 42 lines."},
    {"error": {"status": 429, "message": "slow down", "once": true}},
    {"reply": "Done.", "latency_ms": 250}
  ]
}
```

Step kinds:

- `{"reply": "text"}` serves a plain assistant reply, then advances.
- `{"reply": ["a", "b"]}` picks one reply at random (`--seed` makes it
  deterministic).
- `{"tool_calls": [{"name": ..., "arguments": {...}}]}` serves an
  assistant message with OpenAI-style tool calls. The scenario stays on
  this step until the client sends tool results back (messages with
  `"role": "tool"`), then it advances. A step can combine `reply` text
  and `tool_calls`.
- `{"latency_ms": 250}` on any step sleeps before replying.
- `{"error": {"status": 500}}` replies with that HTTP status and an
  OpenAI-style error body, and repeats on every request. Add
  `"once": true` to serve the error a single time and then continue.
  Use `"body": {...}` to send a custom error payload verbatim.

When the steps run out, the server falls back to echo mode.

## streaming

Send `"stream": true` in the request and mockllm answers with
server-sent events instead of json: one chunk per word, a final
chunk carrying `finish_reason`, then `data: [DONE]`. Tool calls
stream as a single delta chunk with `finish_reason: tool_calls`.
Errors still come back as plain json, same as real APIs.

```sh
curl -N http://127.0.0.1:8471/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mockllm","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

## cli

```sh
python3 mockllm.py [--port N] [--scenario FILE] [--record FILE]
                   [--latency MS] [--seed N] [--model NAME]
```

- `--port` default 8471.
- `--record FILE` appends every chat request body as one JSON object
  per line (jsonl) for later inspection.
- `--latency MS` adds a default artificial delay to every reply.
- `--seed N` seeds the random choices so runs are reproducible.
- `--model NAME` sets the model id reported in replies (default
  `mockllm`).

## testing

```sh
python3 test_mockllm.py
```

Spins up the real server on localhost and checks scripted replies,
tool call sequences, error injection, latency, request recording,
and seeded determinism.
