"""
mloom quickstart: no API key, no network.

A tiny fake RAG pipeline is tracked with @track_run and @track_metric. One run
fails on purpose so you can see how mloom stores the error, the failing line
and the local variables (with secret-looking values redacted).

    python examples/quickstart.py
    mloom runs
    mloom ui
"""

import time

import mloom
from mloom import track_metric, track_run

mloom.init(project_id="quickstart")

DOCUMENTS = {
    "pricing": "The Pro plan costs $20 per month.",
    "support": "Support is available on weekdays from 9:00 to 17:00.",
}


@track_metric("retrieve")
def retrieve(question: str) -> list:
    time.sleep(0.02)
    return [text for topic, text in DOCUMENTS.items() if topic in question.lower()]


@track_metric("generate")
def generate(question: str, context: list) -> str:
    time.sleep(0.05)
    if not context:
        return "Sorry, I could not find anything about that."
    return f"{context[0]} (answering: {question})"


@track_run(name="answer_question")
def answer_question(question: str) -> str:
    context = retrieve(question)
    return generate(question, context)


@track_run(name="sync_knowledge_base")
def sync_knowledge_base(source_url: str) -> int:
    api_key = "sk-quickstart-not-a-real-key"  # will show up as [REDACTED]
    fetched_pages = 0
    raise ConnectionError(f"Could not reach {source_url}")


if __name__ == "__main__":
    for question in ["What is the pricing?", "When is support open?", "Do you have a mobile app?"]:
        print(answer_question(question))

    try:
        sync_knowledge_base("https://kb.example.com")
    except ConnectionError as error:
        print(f"sync_knowledge_base failed as planned: {error}")

    # Events are written in the background; make sure they are stored before exiting.
    mloom.flush()

    print()
    print(f"Runs stored in {mloom.config.local_db_path}")
    print("Next: run `mloom runs` to list them, or `mloom ui` to open the dashboard.")
