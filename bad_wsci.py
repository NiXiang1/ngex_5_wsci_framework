"""Demonstrate the deliberately inefficient "give the model everything" approach."""

import argparse
import os
from pathlib import Path

from ollama import chat


DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"
DEFAULT_QUESTION = """I changed my university password this morning.
Now my Windows laptop won't connect to campus Wi-Fi,
but my phone still works."""


def load_all_context() -> str:
    """Read every knowledge document, intentionally without selecting."""
    sections = []
    for file in sorted(KNOWLEDGE_DIR.glob("*.txt")):
        sections.append(f"--- {file.name} ---\n{file.read_text(encoding='utf-8').strip()}")
    if not sections:
        raise RuntimeError(f"No knowledge files found in {KNOWLEDGE_DIR}")
    return "\n\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    context = load_all_context()
    response = chat(
        model=args.model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a university IT support assistant. Use the supplied "
                    "knowledge base to give safe, concise, step-by-step help."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"KNOWLEDGE BASE:\n{context}\n\nSTUDENT PROBLEM:\n{args.question}"
                ),
            },
        ],
        options={"temperature": 0},
    )

    print("Model:", args.model)
    print("Context characters:", len(context))
    print(response.message.content)


if __name__ == "__main__":
    main()
