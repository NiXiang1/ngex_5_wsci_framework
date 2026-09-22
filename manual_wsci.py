"""Demonstrate manual SELECT: send only known-relevant files to the model."""

import argparse
import os
from pathlib import Path

from ollama import chat


DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"
DEFAULT_QUESTION = """I changed my university password this morning.
Now my Windows laptop won't connect to campus Wi-Fi,
but my phone still works."""
SELECTED_FILENAMES = (
    "wifi_setup.txt",
    "password_changes.txt",
    "service_status.txt",
)


def read_selected_context(selected_files: list[Path]) -> str:
    """Read only the documents manually selected for this question."""
    missing = [str(file) for file in selected_files if not file.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing selected knowledge files: {missing}")
    return "\n\n".join(
        f"--- {file.name} ---\n{file.read_text(encoding='utf-8').strip()}"
        for file in selected_files
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    selected_files = [KNOWLEDGE_DIR / name for name in SELECTED_FILENAMES]
    context = read_selected_context(selected_files)
    response = chat(
        model=args.model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a university IT support assistant. Answer only from the "
                    "selected support documents. Give concise, step-by-step help."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"SELECTED CONTEXT:\n{context}\n\nSTUDENT PROBLEM:\n{args.question}"
                ),
            },
        ],
        options={"temperature": 0},
    )

    print("Model:", args.model)
    print("Selected files:", ", ".join(file.name for file in selected_files))
    print("Context characters:", len(context))
    print(response.message.content)


if __name__ == "__main__":
    main()
