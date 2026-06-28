"""LLM API helper for the Corvo internal endpoint.

Provides both async and synchronous interfaces for calling Claude models.
"""

from __future__ import annotations

import asyncio
import ssl
from typing import Optional

import aiohttp

URL_CORVO_COMPLETE_XP = (
    "https://corvoproxy.preprodc1.us-west-2.aws-dev.app.snowflake.com"
    "/v1/textcompletion_xp"
)

_HEADERS = {
    "sf-external-function-signature": (
        "(MODEL VARCHAR, MESSAGES ARRAY, OPTIONS OBJECT)"
    ),
    "sf-external-function-name": "TRY_COMPLETE$V2",
    "sf-ml-account-hash": "internal-eval",
    "sf-ml-enabled-cross-regions": "ANY_REGION",
}


async def _do_request(
    session: aiohttp.ClientSession,
    payload: dict,
) -> str | None:
    timeout = aiohttp.ClientTimeout(total=300)
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    async with session.post(
        url=URL_CORVO_COMPLETE_XP,
        json=payload,
        headers=_HEADERS,
        timeout=timeout,
        ssl=ssl_context,
    ) as response:
        response_text = await response.text()
        if response.status != 200:
            raise RuntimeError(
                f"Corvo request failed: {response.status} - {response_text[:500]}"
            )

        resp_json = await response.json()
        data = resp_json.get("data", [])
        if not data:
            raise RuntimeError(f"Empty response data: {response_text[:500]}")

        result = data[0][1]
        if isinstance(result, dict) and "choices" in result:
            choice = result["choices"][0]
            if isinstance(choice, dict):
                if "messages" in choice:
                    return choice["messages"]
                if "message" in choice and isinstance(choice["message"], dict):
                    return choice["message"].get("content", "")
                if "text" in choice:
                    return choice["text"]

        raise RuntimeError(f"Unexpected response format: {response_text[:500]}")


async def call_llm_async(
    prompt: str,
    model_name: str = "claude-opus-4-6",
    max_tokens: int = 8192,
    temperature: float = 0.0,
    system: str | None = None,
    session: aiohttp.ClientSession | None = None,
) -> str:
    """Send a prompt to the LLM and return the generated text.

    Raises RuntimeError on failure.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    options = {"max_tokens": max_tokens, "temperature": temperature}
    payload = {"data": [[0, model_name, messages, options]]}

    if session is None:
        async with aiohttp.ClientSession() as new_session:
            result = await _do_request(new_session, payload)
    else:
        result = await _do_request(session, payload)

    if result is None:
        raise RuntimeError("LLM returned None")
    return result


def call_llm(
    prompt: str,
    model_name: str = "claude-opus-4-6",
    max_tokens: int = 8192,
    temperature: float = 0.0,
    system: str | None = None,
) -> str:
    """Synchronous wrapper around call_llm_async."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(
                call_llm_async(prompt, model_name, max_tokens, temperature, system)
            )
        finally:
            new_loop.close()
    else:
        return asyncio.run(
            call_llm_async(prompt, model_name, max_tokens, temperature, system)
        )
