"""
LLM client initialization. Supports two providers, switched via
LLM_PROVIDER in .env — "ollama" (fully local, $0 forever) or "gemini"
(free tier via Google AI Studio, no local hardware needed).

Every agent/module calls chat_completion() below rather than touching a
provider client directly, so switching providers (or adding a third
later) never requires touching agent code — just this file.
"""
from src.core.config import settings

_ollama_client = None
_gemini_client = None

if settings.llm_provider == "gemini":
    from google import genai

    if not settings.gemini_api_key:
        raise RuntimeError(
            "LLM_PROVIDER=gemini but GEMINI_API_KEY is not set. "
            "Get a free key (no card required) at https://aistudio.google.com "
            "and set it in .env."
        )
    _gemini_client = genai.Client(api_key=settings.gemini_api_key)
else:
    from ollama import Client

    _ollama_client = Client(host=settings.ollama_base_url)


def chat_completion(
    messages: list[dict],
    temperature: float = 0.2,
    model: str | None = None,
) -> str:
    """
    messages: list of {"role": "system"|"user"|"assistant", "content": str}
    Returns the assistant's reply text.
    """
    if settings.llm_provider == "gemini":
        return _gemini_chat_completion(messages, temperature, model)
    return _ollama_chat_completion(messages, temperature, model)


def _ollama_chat_completion(messages: list[dict], temperature: float, model: str | None) -> str:
    response = _ollama_client.chat(
        model=model or settings.ollama_model,
        messages=messages,
        options={"temperature": temperature},
    )
    return response["message"]["content"]


def _gemini_chat_completion(messages: list[dict], temperature: float, model: str | None) -> str:
    from google.genai import types

    # Gemini takes system instructions as a separate config field, not as
    # a message in the turn list — so system-role messages are pulled out
    # and joined rather than passed through as a "turn".
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    system_instruction = "\n".join(system_parts) if system_parts else None

    # Gemini also uses "model" instead of "assistant" as the role name
    # for prior AI turns.
    contents = [
        types.Content(
            role="model" if m["role"] == "assistant" else "user",
            parts=[types.Part(text=m["content"])],
        )
        for m in messages
        if m["role"] != "system"
    ]

    response = _gemini_client.models.generate_content(
        model=model or settings.gemini_model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
        ),
    )
    return response.text


def ensure_model_available() -> None:
    """
    Ollama-only: pulls the configured model if it isn't already present
    locally, so the first real request doesn't stall on a cold download.
    No-op for Gemini — there's nothing to pull, the model runs on Google's
    infrastructure.
    """
    if settings.llm_provider == "gemini":
        return

    local_models = {m["model"] for m in _ollama_client.list().get("models", [])}
    target = settings.ollama_model
    if target not in local_models and not any(m.startswith(target.split(":")[0]) for m in local_models):
        print(f"[llm_provider] Model '{target}' not found locally — pulling now (this may take a while)...")
        for progress in _ollama_client.pull(target, stream=True):
            status = progress.get("status", "")
            if status:
                print(f"[llm_provider] {status}")
        print(f"[llm_provider] Model '{target}' ready.")