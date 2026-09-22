"""A complete Write, Select, Compress, Isolate (WSCI) demonstration."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from ollama import chat


DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
DEFAULT_STATE_PATH = BASE_DIR / "state.json"
DEFAULT_QUESTION = """I changed my university password this morning.
Now my Windows laptop won't connect to campus Wi-Fi,
but my phone still works."""

# This transparent keyword selector is intentional for the classroom exercise.
# A production system could replace it with embeddings and semantic search.
FILE_KEYWORDS = {
    "wifi_setup.txt": ("wi-fi", "wifi", "wireless", "eduroam", "windows", "laptop"),
    "password_changes.txt": ("password", "credential", "credentials", "login"),
    "service_status.txt": ("status", "outage", "down", "works", "operational"),
    "email_setup.txt": ("email", "mail", "outlook", "webmail"),
    "printing.txt": ("print", "printer", "printing"),
    "vpn.txt": ("vpn", "remote", "off campus"),
    "classroom_projectors.txt": ("projector", "display", "hdmi", "usb-c"),
}

COMPRESSION_SCHEMA = {
    "type": "object",
    "properties": {
        "relevant_facts": {
            "type": "array",
            "items": {"type": "string", "maxLength": 180},
            "minItems": 1,
            "maxItems": 4,
        },
        "recommended_actions": {
            "type": "array",
            "items": {"type": "string", "maxLength": 180},
            "minItems": 1,
            "maxItems": 3,
        },
        "warnings": {
            "type": "array",
            "items": {"type": "string", "maxLength": 180},
            "maxItems": 2,
        },
    },
    "required": ["relevant_facts", "recommended_actions", "warnings"],
    "additionalProperties": False,
}
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "diagnosis": {"type": "string", "maxLength": 250},
        "steps": {
            "type": "array",
            "items": {"type": "string", "maxLength": 180},
            "minItems": 1,
            "maxItems": 4,
        },
        "escalate_if": {
            "type": "array",
            "items": {"type": "string", "maxLength": 180},
            "maxItems": 3,
        },
    },
    "required": ["diagnosis", "steps", "escalate_if"],
    "additionalProperties": False,
}


def _keyword_present(keyword: str, query: str) -> bool:
    """Match complete keywords so, for example, 'mail' does not match 'email'."""
    return re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", query) is not None


def select_context(user_question: str, limit: int = 3) -> list[Path]:
    """SELECT the highest-scoring knowledge documents for a question."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    query = user_question.casefold()
    scored_files: list[tuple[int, Path]] = []
    for filename, keywords in FILE_KEYWORDS.items():
        score = sum(_keyword_present(keyword, query) for keyword in keywords)
        path = KNOWLEDGE_DIR / filename
        if score and path.is_file():
            scored_files.append((score, path))
    scored_files.sort(key=lambda item: (-item[0], item[1].name))
    return [path for _, path in scored_files[:limit]]


def read_context(selected_files: list[Path]) -> str:
    """Read selected files and keep source names visible to the model."""
    if not selected_files:
        raise ValueError("At least one context file must be selected")
    sections = []
    for file in selected_files:
        if not file.is_file():
            raise FileNotFoundError(f"Knowledge file not found: {file}")
        sections.append(f"--- {file.name} ---\n{file.read_text(encoding='utf-8').strip()}")
    return "\n\n".join(sections)


def parse_json_object(raw_text: str, label: str) -> dict[str, Any]:
    """Parse a model response and require a JSON object."""
    try:
        value = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Qwen returned invalid JSON for {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Qwen returned a non-object JSON value for {label}")
    return value


def validate_string_list_object(
    value: dict[str, Any], required_keys: tuple[str, ...], label: str
) -> dict[str, Any]:
    """Validate structured output even if a model ignores the JSON schema."""
    if set(value) != set(required_keys):
        raise ValueError(f"{label} must contain exactly: {', '.join(required_keys)}")
    for key in required_keys:
        item = value[key]
        if key == "diagnosis":
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"{label}.{key} must be a non-empty string")
        elif not isinstance(item, list) or not all(
            isinstance(entry, str) and entry.strip() for entry in item
        ):
            raise ValueError(f"{label}.{key} must be a list of non-empty strings")
    return value


def compress_context(context: str, user_question: str, model: str) -> dict[str, Any]:
    """COMPRESS selected documents to facts that address this question."""
    response = chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Compress the documents into short support facts for this exact "
                    "problem. Include only facts and actions that directly help solve "
                    "it. A document may mention other services; omit them unless the "
                    "question asks about them. Do not invent menu paths or details. "
                    "Return concise JSON matching the supplied schema."
                ),
            },
            {
                "role": "user",
                "content": f"QUESTION:\n{user_question}\n\nCONTEXT:\n{context}",
            },
        ],
        format=COMPRESSION_SCHEMA,
        options={"temperature": 0},
    )
    result = parse_json_object(response.message.content, "compressed context")
    return validate_string_list_object(
        result,
        ("relevant_facts", "recommended_actions", "warnings"),
        "compressed context",
    )


def classify_problem(selected_files: list[Path]) -> str:
    """Use the best SELECT result as the state partition for ISOLATE."""
    return selected_files[0].stem if selected_files else "general"


def empty_state() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "diagnostic_contexts": {},
        "report_context": {"total_cases": 0, "cases_by_category": {}},
    }


def load_state(state_path: Path) -> dict[str, Any]:
    """Load and minimally validate the state artifact."""
    if not state_path.exists():
        return empty_state()
    try:
        value = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{state_path} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("state.json must contain a JSON object")
    value.setdefault("schema_version", 1)
    value.setdefault("diagnostic_contexts", {})
    value.setdefault("report_context", {})
    if not isinstance(value["diagnostic_contexts"], dict):
        raise ValueError("diagnostic_contexts must be a JSON object")
    if not isinstance(value["report_context"], dict):
        raise ValueError("report_context must be a JSON object")
    return value


def relevant_state_for(category: str, state: dict[str, Any]) -> dict[str, Any]:
    """ISOLATE only one category; never expose the full state to the model."""
    previous = state.get("diagnostic_contexts", {}).get(category)
    if not isinstance(previous, dict):
        return {}
    # Counters, other categories and the previous raw question are deliberately
    # excluded from the model context.
    answer = previous.get("answer")
    return {"previous_answer": answer} if isinstance(answer, dict) else {}


def answer_question(
    user_question: str,
    compressed_context: dict[str, Any],
    isolated_state: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    """Answer from compressed evidence plus the isolated prior partition."""
    response = chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a university IT support assistant. Use only the "
                    "compressed support facts and relevant prior state. Every proposed "
                    "step must be supported by compressed_context.recommended_actions; "
                    "do not add UI paths or unrelated checks. Return concise JSON "
                    "matching the supplied schema. Never request or repeat passwords."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": user_question,
                        "compressed_context": compressed_context,
                        "relevant_prior_state": isolated_state,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        ],
        format=ANSWER_SCHEMA,
        options={"temperature": 0},
    )
    result = parse_json_object(response.message.content, "support answer")
    return validate_string_list_object(
        result, ("diagnosis", "steps", "escalate_if"), "support answer"
    )


def write_state(
    state_path: Path,
    state: dict[str, Any],
    category: str,
    user_question: str,
    selected_files: list[Path],
    compressed_context: dict[str, Any],
    answer: dict[str, Any],
) -> None:
    """WRITE structured reusable state using an atomic file replacement."""
    state["diagnostic_contexts"][category] = {
        "problem": user_question,
        "selected_files": [file.name for file in selected_files],
        "compressed_context": compressed_context,
        "answer": answer,
    }
    report = state["report_context"]
    report["total_cases"] = int(report.get("total_cases", 0)) + 1
    counts = report.setdefault("cases_by_category", {})
    counts[category] = int(counts.get(category, 0)) + 1
    report["last_category"] = category

    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = state_path.with_suffix(state_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary_path.replace(state_path)


def run_pipeline(
    user_question: str, model: str, state_path: Path = DEFAULT_STATE_PATH
) -> dict[str, Any]:
    """Run all four WSCI stages and return metrics useful for comparison."""
    if not user_question.strip():
        raise ValueError("question must not be empty")
    selected_files = select_context(user_question)
    if not selected_files:
        raise RuntimeError("No relevant knowledge files were selected")

    context = read_context(selected_files)
    compressed_context = compress_context(context, user_question, model)
    state = load_state(state_path)
    category = classify_problem(selected_files)
    isolated_state = relevant_state_for(category, state)
    answer = answer_question(user_question, compressed_context, isolated_state, model)
    write_state(
        state_path,
        state,
        category,
        user_question,
        selected_files,
        compressed_context,
        answer,
    )
    return {
        "model": model,
        "category": category,
        "selected_files": [file.name for file in selected_files],
        "original_context_characters": len(context),
        "compressed_context_characters": len(
            json.dumps(compressed_context, ensure_ascii=False)
        ),
        "answer": answer,
        "state_path": str(state_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE_PATH)
    args = parser.parse_args()

    try:
        result = run_pipeline(args.question, args.model, args.state.resolve())
    except Exception as exc:
        raise SystemExit(
            f"WSCI pipeline failed: {exc}\n"
            f"Check that Ollama is running and model '{args.model}' is installed."
        ) from exc

    print("Model:", result["model"])
    print("Category:", result["category"])
    print("Selected files:", ", ".join(result["selected_files"]))
    print("Original context characters:", result["original_context_characters"])
    print("Compressed context characters:", result["compressed_context_characters"])
    print(json.dumps(result["answer"], indent=2, ensure_ascii=False))
    print("State written to:", result["state_path"])


if __name__ == "__main__":
    main()
