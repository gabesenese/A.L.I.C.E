from __future__ import annotations

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from app.api.dependencies import get_current_user, get_pipeline
from app.api.schemas import ChatRequest, ChatResponse, HealthResponse
from app.logging_config import get_logger, set_trace_id

logger = get_logger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: str = Depends(get_current_user),
    pipeline=Depends(get_pipeline),
) -> ChatResponse:
    result = await pipeline.run_turn(
        user_input=request.message,
        user_id=request.user_id or current_user,
    )
    metadata = dict(result.metadata or {})
    return ChatResponse(
        response=result.response_text,
        trace_id=str(metadata.get("trace_id", "")),
        requires_follow_up=bool(metadata.get("requires_follow_up", False)),
        tools_used=list(metadata.get("tools_used", [])),
    )


@router.websocket("/ws/chat")
async def chat_ws(websocket: WebSocket, pipeline=Depends(get_pipeline)) -> None:
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            trace_id = str(data.get("trace_id", ""))
            if trace_id:
                set_trace_id(trace_id)

            result = await pipeline.run_turn(
                user_input=str(data.get("message", "")),
                user_id=str(data.get("user_id", "anonymous")),
            )
            metadata = dict(result.metadata or {})
            await websocket.send_json(
                {
                    "type": "response",
                    "text": result.response_text,
                    "trace_id": metadata.get("trace_id", ""),
                }
            )
            if bool(metadata.get("requires_follow_up", False)):
                await websocket.send_json({"type": "awaiting_input"})
    except WebSocketDisconnect:
        logger.info("websocket_disconnected")
