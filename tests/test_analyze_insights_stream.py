"""Client-side assembly of streamed chat completions."""

import json
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import analyze_insights  # noqa: E402


def _sse(*events: str) -> list[str]:
    lines = []
    for event in events:
        lines.append(f"data: {event}")
        lines.append("")
    lines.append("data: [DONE]")
    return lines


def test_consume_sse_joins_delta_text():
    lines = _sse(
        json.dumps({"choices": [{"delta": {"role": "assistant"}}]}),
        json.dumps({"choices": [{"delta": {"content": "Hello"}}]}),
        json.dumps({"choices": [{"delta": {"content": " week"}}]}),
    )
    assert analyze_insights._consume_chat_sse(lines) == "Hello week"


def test_consume_sse_raises_on_stream_error_and_logs_it(capsys):
    lines = ["data: " + json.dumps({"error": {"code": "server_error", "message": "boom"}})]
    with pytest.raises(RuntimeError, match="server_error"):
        analyze_insights._consume_chat_sse(lines)
    err = capsys.readouterr().err
    assert "server_error" in err
    assert "Authorization" not in err


def test_chat_completion_streams_and_logs_error_body(monkeypatch, capsys):
    captured = {}

    class Response:
        def __init__(self, status, body=b"", lines=None):
            self.status_code = status
            self._body = body
            self._lines = lines or []
            self.request = httpx.Request("POST", "https://sentinel.knowlyr.com/v1/chat/completions")

        def read(self):
            return self._body

        def iter_lines(self):
            yield from self._lines

        def raise_for_status(self):
            if self.status_code >= 400:
                raise httpx.HTTPStatusError("bad", request=self.request, response=self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_stream(method, url, headers=None, json=None, timeout=None):
        captured["method"] = method
        captured["json"] = json
        captured["auth"] = (headers or {}).get("Authorization")
        if json and json.get("model") == "fail":
            return Response(502, b'{"error":{"code":"server_is_overloaded"}}')
        return Response(
            200,
            lines=_sse(json_dumps_chunk("ok")),
        )

    def json_dumps_chunk(text):
        return json.dumps({"choices": [{"delta": {"content": text}}]})

    monkeypatch.setattr(httpx, "stream", fake_stream)

    assert (
        analyze_insights._openai_chat_completion(
            "https://sentinel.knowlyr.com/v1",
            "kly-proxy-SECRET",
            "gpt-6.1-sol",
            "sys",
            "user",
            128,
        )
        == "ok"
    )
    assert captured["method"] == "POST"
    assert captured["json"]["stream"] is True
    assert captured["auth"] == "Bearer kly-proxy-SECRET"
    assert "kly-proxy-SECRET" not in capsys.readouterr().err

    with pytest.raises(httpx.HTTPStatusError):
        analyze_insights._openai_chat_completion(
            "https://sentinel.knowlyr.com/v1",
            "kly-proxy-SECRET",
            "fail",
            "sys",
            "user",
            128,
        )
    err = capsys.readouterr().err
    assert "HTTP 502" in err
    assert "server_is_overloaded" in err
    assert "kly-proxy-SECRET" not in err
    assert "Authorization" not in err
