"""A small WSCI (Write, Select, Compress, Isolate) demonstration.

The program keeps reusable structured state, selects only relevant knowledge,
compresses it with Qwen, isolates the current task's state, and asks Qwen for a
structured support answer.
"""

import os
from pathlib import Path
from ollama import chat
import json


MODEL = os.getenv("OLLAMA_MODEL", "qwen3:0.6b")
BASE_DIR = Path(__file__).parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
STATE_PATH = BASE_DIR / "state.json"

question = """
I changed my university password this morning.
Now my Windows laptop won't connect to campus Wi-Fi,
but my phone still works.
""".strip()


# Keywords are intentionally simple for this classroom exercise. A production
# version would replace this table with embeddings and semantic search.
FILE_KEYWORDS = {
    "wifi_setup.txt": ("wi-fi", "wifi", "wireless", "eduroam", "windows", "laptop"),
    "password_changes.txt": ("password", "credential", "login", "sign in"),
    "service_status.txt": ("status", "outage", "down", "works", "operational"),
    "email_setup.txt": ("email", "mail", "outlook", "webmail"),
    "printing.txt": ("print", "printer", "printing"),
    "vpn.txt": ("vpn", "remote", "off campus"),
    "classroom_projectors.txt": ("projector", "display", "hdmi", "usb-c"),
}

COMPRESSION_SCHEMA = {
    "type": "object",
    "properties": {
        "relevant_facts": {"type": "array", "items": {"type": "string"}},
        "recommended_actions": {"type": "array", "items": {"type": "string"}},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["relevant_facts", "recommended_actions", "warnings"],
}

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "diagnosis": {"type": "string"},
        "steps": {"type": "array", "items": {"type": "string"}},
        "escalate_if": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["diagnosis", "steps", "escalate_if"],
}


def select_context(user_question: str) -> list[Path]:
    """SELECT knowledge files whose keyword score is non-zero."""
    query = user_question.casefold()
    scored_files = []

    for filename, keywords in FILE_KEYWORDS.items():
        score = sum(keyword in query for keyword in keywords)
        if score:
            scored_files.append((score, KNOWLEDGE_DIR / filename))

    scored_files.sort(key=lambda item: (-item[0], item[1].name))
    return [path for _, path in scored_files]


def read_context(selected_files: list[Path]) -> str:
    sections = []
    for file in selected_files:
        text = file.read_text(encoding="utf-8").strip()
        sections.append(f"--- {file.name} ---\n{text}")
    return "\n\n".join(sections)


def parse_json_object(raw_text: str, label: str) -> dict:
    """Validate that a model response is a JSON object."""
    try:
        value = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Qwen returned invalid JSON for {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Qwen returned a non-object JSON value for {label}")
    return value


def compress_context(context: str, user_question: str) -> dict:
    """COMPRESS the selected documents down to facts relevant to the question."""
    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract only facts that help answer the student's problem. "
                    "Do not invent facts. Return JSON matching the supplied schema."
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
    return parse_json_object(response.message.content, "compressed context")


def classify_problem(user_question: str) -> str:
    """Return the state partition needed for this question (ISOLATE)."""
    selected = select_context(user_question)
    if not selected:
        return "general"
    return selected[0].stem


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {"diagnostic_contexts": {}, "report_context": {}}
    with STATE_PATH.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError("state.json must contain a JSON object")
    value.setdefault("diagnostic_contexts", {})
    value.setdefault("report_context", {})
    return value


def relevant_state_for(category: str, state: dict) -> dict:
    """ISOLATE one relevant diagnostic partition instead of exposing all state."""
    previous = state.get("diagnostic_contexts", {}).get(category)
    return {"previous_case": previous} if previous else {}


def answer_question(
    user_question: str, compressed_context: dict, isolated_state: dict
) -> dict:
    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a university IT support assistant. Use only the "
                    "compressed support facts and relevant prior state. Return JSON "
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
                    indent=2,
                ),
            },
        ],
        format=ANSWER_SCHEMA,
        options={"temperature": 0},
    )
    return parse_json_object(response.message.content, "support answer")


def write_state(
    state: dict,
    category: str,
    user_question: str,
    selected_files: list[Path],
    compressed_context: dict,
    answer: dict,
) -> None:
    """WRITE structured, reusable information to the state artifact."""
    state["diagnostic_contexts"][category] = {
        "problem": user_question,
        "selected_files": [file.name for file in selected_files],
        "compressed_context": compressed_context,
        "answer": answer,
    }
    report = state["report_context"]
    report["total_cases"] = int(report.get("total_cases", 0)) + 1
    report["last_category"] = category

    with STATE_PATH.open("w", encoding="utf-8") as file:
        json.dump(state, file, indent=2, ensure_ascii=False)
        file.write("\n")


def main() -> None:
    selected_files = select_context(question)
    if not selected_files:
        raise RuntimeError("No relevant knowledge files were selected")

    context = read_context(selected_files)
    compressed_context = compress_context(context, question)

    state = load_state()
    category = classify_problem(question)
    isolated_state = relevant_state_for(category, state)
    answer = answer_question(question, compressed_context, isolated_state)
    write_state(
        state,
        category,
        question,
        selected_files,
        compressed_context,
        answer,
    )

    print("Selected files:", ", ".join(file.name for file in selected_files))
    print("Original context characters:", len(context))
    compressed_text = json.dumps(compressed_context, ensure_ascii=False)
    print("Compressed context characters:", len(compressed_text))
    print(json.dumps(answer, indent=2, ensure_ascii=False))
    print(f"State written to: {STATE_PATH}")


if __name__ == "__main__":
    main()

