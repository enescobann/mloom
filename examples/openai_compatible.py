"""
Track a real OpenAI-compatible endpoint with autotrack().

Defaults target a local Ollama server; anything that speaks the OpenAI chat
completions API works (OpenAI, LM Studio, vLLM, llama.cpp server, ...).

    ollama pull llama3.2
    python examples/openai_compatible.py

Configuration through environment variables:

    OPENAI_BASE_URL   default http://localhost:11434/v1 (Ollama)
    OPENAI_API_KEY    default "ollama" (local servers ignore it)
    MLOOM_MODEL       default llama3.2

For OpenAI itself:

    OPENAI_BASE_URL=https://api.openai.com/v1 OPENAI_API_KEY=sk-... \
    MLOOM_MODEL=gpt-4o-mini python examples/openai_compatible.py
"""

import os

from openai import OpenAI

import mloom

BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.environ.get("OPENAI_API_KEY", "ollama")
MODEL = os.environ.get("MLOOM_MODEL", "llama3.2")

mloom.init(project_id="openai-compatible")
mloom.autotrack()  # records every chat.completions.create call

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def ask(question: str, system: str = "Answer in one short sentence.") -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": question}],
    )
    return response.choices[0].message.content


@mloom.track_run(name="explain_and_summarize")
def explain_and_summarize(topic: str) -> str:
    # Both calls are grouped under one run, with tokens, cost and latency each.
    explanation = ask(f"Explain {topic}.", system="Answer in three sentences.")
    return ask(f"Summarize in five words: {explanation}")


if __name__ == "__main__":
    print(f"Using {MODEL} at {BASE_URL}")

    # A standalone call outside @track_run is recorded as its own run.
    print("Standalone:", ask("What is the capital of France?"))

    print("Pipeline:  ", explain_and_summarize("vector databases"))

    mloom.flush()
    print("Done. Run `mloom runs` or `mloom ui` to inspect the results.")
