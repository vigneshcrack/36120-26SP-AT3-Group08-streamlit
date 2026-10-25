import requests

from app import main as core


class FakeResponse:
    def __init__(self, status, body):
        self.status_code, self._body = status, body
        self.ok = status < 400

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def test_success_returns_data(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(200, {"prediction": 14.7}))
    result = core.call_api("https://x", "/predict", {"token": "BTC"})
    assert result.ok and result.data == {"prediction": 14.7} and result.status == 200


def test_client_error_is_not_retried_and_message_is_kept(monkeypatch):
    calls = []

    def post(*a, **k):
        calls.append(1)
        return FakeResponse(400, {"error": "Unsupported cryptocurrency", "supported_tokens": ["BTC"]})

    monkeypatch.setattr(requests, "post", post)
    result = core.call_api("https://x", "/predict", {"token": "ADA"}, retries=2)
    assert not result.ok and result.status == 400
    assert result.error == "Unsupported cryptocurrency (HTTP 400)"
    assert len(calls) == 1


def test_timeout_is_retried_then_reported(monkeypatch):
    calls = []
    monkeypatch.setattr(core.time, "sleep", lambda s: None)

    def get(*a, **k):
        calls.append(1)
        raise requests.Timeout

    monkeypatch.setattr(requests, "get", get)
    result = core.call_api("https://x", "/health", retries=1)
    assert not result.ok and "too long" in result.error
    assert len(calls) == 2


def test_server_error_then_success(monkeypatch):
    responses = iter([FakeResponse(503, {"error": "Market data source unavailable"}), FakeResponse(200, {"ok": 1})])
    monkeypatch.setattr(core.time, "sleep", lambda s: None)
    monkeypatch.setattr(requests, "get", lambda *a, **k: next(responses))
    assert core.call_api("https://x", "/model-metadata").ok


def test_invalid_json_and_missing_url(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResponse(200, ValueError("bad")))
    assert "invalid response" in core.call_api("https://x", "/", retries=0).error
    assert "not been deployed" in core.call_api(None, "/").error


def test_failures_are_not_cached(monkeypatch):
    core.clear_cache()
    outcomes = iter([requests.ConnectionError(), requests.ConnectionError(), FakeResponse(200, {"v": 1})])

    def get(*a, **k):
        item = next(outcomes)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(core.time, "sleep", lambda s: None)
    monkeypatch.setattr(requests, "get", get)
    first = core.cached_call("https://cache-test", "/model-metadata")
    second = core.cached_call("https://cache-test", "/model-metadata")
    assert not first.ok and second.ok and second.data == {"v": 1}
