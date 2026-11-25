import asyncio
import json
import time

import httpx
import openai
import pytest

import mloom
from mloom import track_run
from mloom.sdk import patcher
from mloom.sdk.cost import calculate_cost
from mloom.sdk.patcher import patch_openai

MODEL = "gpt-4o-mini"
PROMPT_TOKENS = 12
COMPLETION_TOKENS = 8
DELAY_SECONDS = 0.05


def _completion_body(model: str) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Paris"},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": PROMPT_TOKENS,
            "completion_tokens": COMPLETION_TOKENS,
            "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
        },
    }


def _ok_handler(request: httpx.Request) -> httpx.Response:
    time.sleep(DELAY_SECONDS)
    model = json.loads(request.content)["model"]
    return httpx.Response(200, json=_completion_body(model))


def _error_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(500, json={"error": {"message": "upstream down", "type": "server_error"}})


def _client(handler=_ok_handler) -> openai.OpenAI:
    return openai.OpenAI(
        api_key="test-key",
        base_url="http://mock-openai.local/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def _ask(client, model=MODEL):
    return client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Capital of France?"}],
    )


@pytest.fixture(autouse=True)
def patched():
    patch_openai()


def test_call_inside_track_run_records_tokens_cost_latency(local_project, fetch_runs):
    client = _client()

    @track_run(name="qa")
    def qa():
        return _ask(client).choices[0].message.content

    assert qa() == "Paris"

    [run] = fetch_runs()
    assert run["run_name"] == "qa"
    [metric] = run["metrics"]
    assert metric["model_name"] == MODEL
    assert metric["response"] == "Paris"
    assert "Capital of France?" in metric["prompt"]
    assert metric["input_tokens"] == PROMPT_TOKENS
    assert metric["output_tokens"] == COMPLETION_TOKENS
    assert metric["total_cost"] == pytest.approx(calculate_cost(MODEL, PROMPT_TOKENS, COMPLETION_TOKENS))
    assert metric["total_cost"] > 0
    assert metric["latency"] >= DELAY_SECONDS * 1000 * 0.8
    assert metric["start_time"] is not None and metric["end_time"] is not None


def test_standalone_call_is_recorded_as_its_own_run(local_project, fetch_runs):
    response = _ask(_client())
    assert response.usage.total_tokens == PROMPT_TOKENS + COMPLETION_TOKENS

    [run] = fetch_runs()
    assert run["run_name"] == "OpenAI API Call"
    assert run["tags"] == {"provider": "openai"}
    assert run["start_time"] is not None
    [metric] = run["metrics"]
    assert metric["input_tokens"] == PROMPT_TOKENS
    assert metric["output_tokens"] == COMPLETION_TOKENS
    assert metric["total_cost"] > 0


def test_patching_twice_records_each_call_once(local_project, fetch_runs):
    patch_openai()
    patch_openai()

    _ask(_client())

    runs = fetch_runs()
    assert len(runs) == 1
    assert len(runs[0]["metrics"]) == 1


def test_async_client_is_tracked(local_project, fetch_runs):
    client = openai.AsyncOpenAI(
        api_key="test-key",
        base_url="http://mock-openai.local/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(_ok_handler)),
    )

    @track_run(name="async-qa")
    async def qa():
        response = await client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": "hi"}]
        )
        return response.choices[0].message.content

    assert asyncio.run(qa()) == "Paris"

    [run] = fetch_runs()
    [metric] = run["metrics"]
    assert metric["input_tokens"] == PROMPT_TOKENS
    assert metric["output_tokens"] == COMPLETION_TOKENS


def test_api_error_is_recorded_and_reraised(local_project, fetch_runs):
    client = _client(_error_handler)

    with pytest.raises(openai.APIStatusError):
        _ask(client)

    [run] = fetch_runs()
    [metric] = run["metrics"]
    assert metric["response"].startswith("ERROR:")
    assert metric["input_tokens"] == 0
    assert metric["total_cost"] == 0.0


def test_telemetry_failure_does_not_break_the_call(local_project, fetch_runs, monkeypatch, capsys):
    def broken_cost(*args, **kwargs):
        raise RuntimeError("pricing bug")

    monkeypatch.setattr(patcher, "calculate_cost", broken_cost)

    response = _ask(_client())

    assert response.choices[0].message.content == "Paris"
    assert fetch_runs() == []
    assert "pricing bug" in capsys.readouterr().err


def test_dated_model_snapshot_is_priced(local_project, fetch_runs):
    _ask(_client(), model="gpt-4o-mini-2024-07-18")

    [run] = fetch_runs()
    assert run["metrics"][0]["total_cost"] == pytest.approx(
        calculate_cost(MODEL, PROMPT_TOKENS, COMPLETION_TOKENS)
    )
