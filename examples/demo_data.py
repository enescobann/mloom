"""
Fill a mloom database with realistic demo runs, e.g. to try the dashboard.

Everything goes through the real SDK (track_run, track_metric, autotrack) and
a real OpenAI client. Only the HTTP transport is fake (httpx.MockTransport),
so no API key or network is needed. Tokens, costs, call sites and error traces
(including secret redaction) are recorded exactly as in a real app.

Because the fake model answers instantly, the script afterwards spreads the
runs over the last few days and gives each call a realistic latency, so the
charts look like real usage.

Write into a throwaway directory instead of ~/.mloom:

    MLOOM_HOME=/tmp/mloom-demo python examples/demo_data.py
    MLOOM_HOME=/tmp/mloom-demo mloom ui
"""

import argparse
import json
import random
from datetime import datetime, timedelta, timezone

import httpx
import openai

import mloom
from mloom import track_metric, track_run

# Median latency (ms) and spread for each fake model.
MODEL_LATENCY = {
    "gpt-4o-mini": (650, 0.35),
    "gpt-4o": (1600, 0.35),
    "gpt-3.5-turbo": (700, 0.30),
    "llama3.1:8b": (1100, 0.40),
}

rng = random.Random(42)


# ─── Fake OpenAI-compatible backend ────────────────────────────────────

class FakeBackend:
    """Answers chat completions; can be told to fail the next request."""

    def __init__(self):
        self.next_status = 200
        self.next_content = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        status, self.next_status = self.next_status, 200
        if status != 200:
            return httpx.Response(status, json={"error": {
                "message": "Rate limit reached for requests", "type": "rate_limit_exceeded",
            }})

        prompt = " ".join(m["content"] for m in body["messages"])
        content, self.next_content = self.next_content, None
        if content is None:
            content = rng.choice([
                "You can reset your password from Settings > Security. A link is valid for 24 hours.",
                "Refunds are processed within 5 business days after the return is received.",
                "The export runs nightly; large workspaces may take up to an hour.",
                "Based on the retrieved documents, the API limit is 600 requests per minute.",
            ])
        return httpx.Response(200, json={
            "id": f"chatcmpl-{rng.randrange(10**8)}",
            "object": "chat.completion",
            "created": 0,
            "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {
                # Pretend the retrieved context makes prompts much longer than the question.
                "prompt_tokens": len(prompt) // 4 + rng.randint(150, 1400),
                "completion_tokens": rng.randint(40, 450),
                "total_tokens": 0,
            },
        })


backend = FakeBackend()
client = openai.OpenAI(
    api_key="demo-key",
    base_url="http://demo.invalid/v1",
    max_retries=0,
    http_client=httpx.Client(transport=httpx.MockTransport(backend)),
)


def chat(model: str, prompt: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


# ─── Project: support-bot ──────────────────────────────────────────────

TICKETS = [
    "How do I reset my password?",
    "I was charged twice this month.",
    "Can I export my data to CSV?",
    "The mobile app crashes when I open settings.",
    "How long do refunds take?",
]


@track_metric("retrieve_kb")
def retrieve_kb(question: str) -> list:
    return [f"kb-{rng.randint(100, 999)}" for _ in range(3)]


@track_run(name="answer_ticket")
def answer_ticket(ticket: str, escalate: bool) -> str:
    docs = retrieve_kb(ticket)
    draft = chat("gpt-4o-mini", f"Docs: {docs}\nTicket: {ticket}")
    if escalate:
        draft = chat("gpt-4o", f"Improve this answer for a frustrated customer: {draft}")
    return draft


@track_run(name="summarize_thread")
def summarize_thread(messages: list) -> str:
    return chat("gpt-4o-mini", "Summarize: " + " | ".join(messages))


# ─── Project: doc-search ───────────────────────────────────────────────

INDEXES = {"handbook": 1200, "api-reference": 640}


@track_metric("embed_query")
def embed_query(query: str) -> list:
    return [rng.random() for _ in range(4)]


@track_metric("vector_search")
def vector_search(vector: list, index_name: str) -> list:
    vector_db_api_key = "vdb-live-7f3a9c2e"
    size = INDEXES[index_name]  # KeyError for unknown indexes
    return [f"{index_name}/chunk-{rng.randrange(size)}" for _ in range(5)]


@track_run(name="search_docs")
def search_docs(query: str, index_name: str) -> str:
    hits = vector_search(embed_query(query), index_name)
    ranked = chat("gpt-3.5-turbo", f"Rank these passages for '{query}': {hits}")
    return chat("gpt-4o-mini", f"Answer '{query}' using: {ranked}")


# ─── Project: eval-harness ─────────────────────────────────────────────

@track_metric("parse_score")
def parse_score(raw_output: str) -> int:
    try:
        payload = json.loads(raw_output)
    except json.JSONDecodeError:
        raise ValueError(f"Judge returned non-JSON output: {raw_output!r}") from None
    return int(payload["score"])


@track_run(name="grade_answer", run_type="CLASSIC")
def grade_answer(question: str, answer: str, malformed: bool) -> int:
    judge_token = "hf_demo_judge_token"
    backend.next_content = "Score: 4/5" if malformed else json.dumps({"score": rng.randint(1, 5)})
    raw_output = chat("llama3.1:8b", f"Grade 1-5.\nQ: {question}\nA: {answer}")
    return parse_score(raw_output)


# ─── Scenario ──────────────────────────────────────────────────────────

def run_one():
    """Run one randomly chosen pipeline; roughly 15% of runs fail."""
    roll = rng.random()
    try:
        if roll < 0.35:
            mloom.init(project_id="support-bot", verbose=False)
            if rng.random() < 0.12:
                backend.next_status = 429
            answer_ticket(rng.choice(TICKETS), escalate=rng.random() < 0.3)
        elif roll < 0.45:
            mloom.init(project_id="support-bot", verbose=False)
            summarize_thread(rng.sample(TICKETS, 3))
        elif roll < 0.75:
            mloom.init(project_id="doc-search", verbose=False)
            index = "legacy-wiki" if rng.random() < 0.15 else rng.choice(list(INDEXES))
            search_docs(rng.choice(["rate limits", "SSO setup", "webhooks", "pagination"]), index)
        elif roll < 0.93:
            mloom.init(project_id="eval-harness", verbose=False)
            grade_answer("What is the refund window?", "30 days", malformed=rng.random() < 0.2)
        else:
            # A plain OpenAI call outside any @track_run is still recorded.
            mloom.init(project_id="support-bot", verbose=False)
            chat("gpt-4o-mini", "Classify the sentiment of: 'Thanks, that fixed it!'")
    except (openai.APIStatusError, KeyError, ValueError):
        pass  # The failure is recorded on the run; keep generating.


def spread_over_time(first_run_id: int, days: int):
    """Give the new runs realistic start times and latencies over the last `days` days."""
    from mloom.db import database
    from mloom.db.classes import Run

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    with database.get_session() as db:
        runs = db.query(Run).filter(Run.id >= first_run_id).order_by(Run.id).all()
        starts = sorted(
            now - timedelta(days=days) + timedelta(seconds=rng.uniform(0, days * 86400 - 600))
            for _ in runs
        )
        for run, start in zip(runs, starts):
            cursor = start + timedelta(milliseconds=rng.randint(2, 15))
            for metric in sorted(run.metrics, key=lambda m: m.id):
                if metric.model_name in MODEL_LATENCY:
                    median, sigma = MODEL_LATENCY[metric.model_name]
                    latency = int(rng.lognormvariate(0, sigma) * median)
                    if (metric.response or "").startswith("ERROR:"):
                        latency = rng.randint(80, 250)
                else:
                    latency = rng.randint(15, 180)
                metric.start_time = cursor
                metric.end_time = cursor + timedelta(milliseconds=latency)
                metric.timestamp = metric.end_time
                metric.latency = latency
                cursor = metric.end_time + timedelta(milliseconds=rng.randint(3, 40))

            end = cursor + timedelta(milliseconds=rng.randint(5, 60))
            run.start_time, run.end_time, run.timestamp = start, end, end
            run.latency = int((end - start).total_seconds() * 1000)
        db.commit()
        failed = sum(1 for r in runs if r.error_trace)
        return len(runs), failed, sorted({r.project_id for r in runs})


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=int, default=40, help="number of runs to generate")
    parser.add_argument("--days", type=int, default=4, help="spread runs over this many days")
    args = parser.parse_args()

    mloom.init(project_id="support-bot", verbose=False)
    mloom.autotrack()

    from mloom.db import database
    from mloom.db.classes import Run
    with database.get_session() as db:
        first_run_id = (db.query(Run.id).order_by(Run.id.desc()).limit(1).scalar() or 0) + 1

    for _ in range(args.runs):
        run_one()
    mloom.flush()

    count, failed, projects = spread_over_time(first_run_id, args.days)
    print(f"Recorded {count} runs ({failed} failed) for {', '.join(projects)}.")
    print(f"Database: {mloom.config.local_db_path}")
    print("Open the dashboard with: mloom ui  (use the same MLOOM_HOME)")


if __name__ == "__main__":
    main()
