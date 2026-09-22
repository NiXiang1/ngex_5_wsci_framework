"""Demonstrate the deliberately inefficient "give the model everything" approach."""

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


context = ""

for file in sorted(KNOWLEDGE_DIR.glob("*.txt")):
    context += f"--- {file.name} ---\n"
    context += file.read_text(encoding="utf-8").strip()
    context += "\n\n"

response = chat(
    model=MODEL,
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
            "content": f"KNOWLEDGE BASE:\n{context}\nSTUDENT PROBLEM:\n{question}",
        },
    ],
    options={"temperature": 0},
)

print("Context characters:", len(context))
print(response.message.content)
