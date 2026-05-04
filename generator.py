import requests

from config import OLLAMA_BASE_URL, LLM_MODEL

_PROMPT_TEMPLATE = """\
You are a knowledgeable assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on available information."
If the context mentions a place or person that matches the question, name it specifically in your answer.

Context:
{context}

Question: {question}
Answer:"""


def _build_prompt(chunks: list[dict], question: str) -> str:
    if not chunks:
        context = "(no relevant information found)"
    else:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            title = chunk["metadata"].get("entity_title", "Unknown")
            parts.append(f"[{i}] ({title})\n{chunk['text']}")
        context = "\n\n".join(parts)
    return _PROMPT_TEMPLATE.format(context=context, question=question)


def generate(chunks: list[dict], question: str) -> str:
    prompt = _build_prompt(chunks, question)

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Ollama. Make sure it is running at http://localhost:11434."
    except requests.exceptions.Timeout:
        return "Error: Ollama request timed out. The model may still be loading."
    except Exception as e:
        return f"Error: {e}"
