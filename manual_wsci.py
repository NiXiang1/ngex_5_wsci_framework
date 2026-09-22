"""Demonstrate manual SELECT: send only known-relevant files to the model."""

import os
from pathlib import Path
from ollama import chat


MODEL = os.getenv("OLLAMA_MODEL", "qwen3:0.6b")
KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"

question = """
I changed my university password this morning.
Now my Windows laptop won't connect to campus Wi-Fi,
but my phone still works.
""".strip()

selected_files = [
    KNOWLEDGE_DIR / "wifi_setup.txt",
    KNOWLEDGE_DIR / "password_changes.txt",
    KNOWLEDGE_DIR / "service_status.txt",
]

context = ""

for file in selected_files:
    context += f"--- {file.name} ---\n"
    context += file.read_text(encoding="utf-8").strip()
    context += "\n\n"

response = chat(
    model=MODEL,
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
            "content": f"SELECTED CONTEXT:\n{context}\nSTUDENT PROBLEM:\n{question}",
        },
    ],
    options={"temperature": 0},
)

print("Selected files:", ", ".join(file.name for file in selected_files))
print("Context characters:", len(context))
print(response.message.content)
