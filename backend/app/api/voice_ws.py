"""WebSocket transport for the realtime voice loop.

Binary frames  -> PCM16 mono @ 16 kHz microphone audio
Text frames    -> JSON ClientEvent (config / text / interrupt)
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from ..core.events import AssistantState, ClientEvent, ServerEvent
from ..services.pipeline import VoiceSession

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/voice")
async def voice_socket(websocket: WebSocket) -> None:
    await websocket.accept()

    async def emit(event: ServerEvent) -> None:
        await websocket.send_text(event.model_dump_json(exclude_none=True))

    session = VoiceSession(emit)
    await emit(ServerEvent(type="state", state=AssistantState.IDLE))

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            if (payload := message.get("bytes")) is not None:
                await session.feed_audio(payload)
                continue

            raw = message.get("text")
            if not raw:
                continue
            try:
                event = ClientEvent.model_validate_json(raw)
            except ValidationError as error:
                await emit(ServerEvent(type="error", text=str(error)))
                continue

            if event.type == "config" and event.data:
                session.configure(event.data)
                logger.info("Session config: %s", event.data)
            elif event.type == "text" and event.text:
                # The client already rendered its own message; no transcript event.
                await session.handle_text(
                    event.text, speak=bool((event.data or {}).get("speak", True))
                )
            elif event.type == "interrupt":
                await session.interrupt()
                await session.set_state(AssistantState.IDLE)
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        logger.exception("Voice socket failed")
    finally:
        await session.interrupt()
