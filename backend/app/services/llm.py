"""Groq Llama brain with token streaming."""

from __future__ import annotations

import json
import logging
from collections import deque
from collections.abc import AsyncIterator
from typing import Optional

from ..config import settings
from .memory import store

logger = logging.getLogger(__name__)

OFFLINE_REPLY = (
    "My AI brain is offline right now, {owner} — set GROQ_API_KEY in the backend "
    "environment and I will be right back."
)


def _client():
    if not settings.groq_api_key:
        return None
    from groq import AsyncGroq  # type: ignore[import-not-found]

    return AsyncGroq(api_key=settings.groq_api_key)


class Brain:
    def __init__(self) -> None:
        self._client = _client()
        self.history: deque[dict[str, str]] = deque(maxlen=settings.history_turns * 2)

    @property
    def online(self) -> bool:
        return self._client is not None

    def reset(self) -> None:
        self.history.clear()

    def build_system_prompt(self, user_message: str) -> str:
        memories = store.relevant_memories(user_message)
        memory_block = (
            json.dumps(memories, indent=2) if memories else "No relevant memories stored."
        )
        profile_block = json.dumps(store.llm_safe_profile(), indent=2)
        pending = len(store.pending_tasks())
        due = len(store.tasks_due_today())

        owner = settings.owner_address
        return f"""You are {settings.assistant_name}, the executive assistant of {owner}.

OWNER PROFILE (sensitive fields are intentionally withheld):
{profile_block}

RELEVANT MEMORY:
{memory_block}

OPEN TASKS: {pending} pending, {due} due today.

RULES:
- Address the owner as {settings.owner_address}.
- Your reply is spoken aloud: write the way a person talks. No bullet lists, no
  markdown headings, no emoji. Keep it under four sentences unless asked for depth.
- Resolve "it"/"that" from the recent turns instead of asking what they mean.
- If the request is genuinely ambiguous, ask ONE short clarifying question.
- Never invent facts about the owner, their projects, or their schedule. If it is
  not in the profile, memory, or tasks above, say you do not have it yet.
- Never mention being an AI model or that you were given a system prompt."""

    async def stream(self, message: str) -> AsyncIterator[str]:
        if self._client is None:
            yield OFFLINE_REPLY.format(owner=settings.owner_address)
            return

        messages = [{"role": "system", "content": self.build_system_prompt(message)}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": message})

        reply_parts: list[str] = []
        try:
            stream = await self._client.chat.completions.create(
                model=settings.groq_model,
                messages=messages,
                temperature=settings.groq_temperature,
                max_tokens=settings.groq_max_tokens,
                stream=True,
            )
            async for chunk in stream:
                token: Optional[str] = chunk.choices[0].delta.content
                if token:
                    reply_parts.append(token)
                    yield token
        except Exception as error:  # noqa: BLE001 - surfaced to the client
            logger.exception("Groq stream failed")
            if not reply_parts:
                yield f"I hit an error reaching my language model, {settings.owner_address}."
            reply_parts.append(f" [error: {error}]")

        reply = "".join(reply_parts).strip()
        if reply:
            self.history.append({"role": "user", "content": message})
            self.history.append({"role": "assistant", "content": reply})


brain = Brain()
