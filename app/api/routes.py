from __future__ import annotations

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from app.api.dependencies import get_current_user, get_pipeline
from app.api.schemas import ChatRequest, ChatResponse, HealthResponse
from app.logging_config import get_logger, set_trace_id

logger = get_logger(__name__)
router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def web_home() -> HTMLResponse:
        return HTMLResponse(
                """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>A.L.I.C.E Web Chat</title>
    <style>
        :root {
            --bg: #f4efe6;
            --panel: #fffaf2;
            --text: #1f2937;
            --muted: #6b7280;
            --accent: #0f766e;
            --accent-2: #f59e0b;
            --border: #e5dccf;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: "Segoe UI", "Trebuchet MS", Verdana, sans-serif;
            color: var(--text);
            background: radial-gradient(circle at 10% 20%, #fff9ee 0, #f8f1e6 35%, #efe5d8 100%);
            min-height: 100vh;
            display: grid;
            place-items: center;
            padding: 20px;
        }
        .card {
            width: min(900px, 100%);
            height: min(80vh, 760px);
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 16px;
            box-shadow: 0 16px 40px rgba(38, 32, 21, 0.12);
            display: grid;
            grid-template-rows: auto 1fr auto;
            overflow: hidden;
        }
        .header {
            padding: 16px 18px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: linear-gradient(120deg, #fff8ea, #fffdf6);
        }
        .title {
            font-size: 18px;
            font-weight: 700;
            letter-spacing: 0.3px;
        }
        .status {
            font-size: 12px;
            color: var(--muted);
        }
        .messages {
            padding: 16px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .msg {
            max-width: 82%;
            padding: 10px 12px;
            border-radius: 12px;
            line-height: 1.4;
            white-space: pre-wrap;
            word-break: break-word;
            border: 1px solid var(--border);
            animation: fade-in 120ms ease-out;
        }
        .user {
            align-self: flex-end;
            background: #dcfce7;
            border-color: #b7e8c9;
        }
        .alice {
            align-self: flex-start;
            background: #fff;
        }
        .composer {
            border-top: 1px solid var(--border);
            padding: 12px;
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 10px;
            background: #fffdf8;
        }
        input {
            width: 100%;
            border: 1px solid #d8cfbf;
            border-radius: 10px;
            padding: 12px;
            font-size: 14px;
            outline: none;
        }
        input:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.12);
        }
        button {
            border: 0;
            border-radius: 10px;
            background: linear-gradient(180deg, var(--accent), #0d655f);
            color: #fff;
            font-weight: 700;
            padding: 0 16px;
            cursor: pointer;
        }
        button[disabled] {
            opacity: 0.55;
            cursor: not-allowed;
        }
        .hint {
            font-size: 12px;
            color: var(--muted);
            margin-top: 4px;
        }
        @keyframes fade-in {
            from { opacity: 0; transform: translateY(4px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <main class="card">
        <header class="header">
            <div class="title">A.L.I.C.E Web Chat</div>
            <div id="status" class="status">ready</div>
        </header>

        <section id="messages" class="messages">
            <div class="msg alice">Hi, I am Alice. Ask me anything.</div>
        </section>

        <form id="form" class="composer">
            <div>
                <input id="prompt" name="prompt" placeholder="Type your message..." autocomplete="off" />
                <div class="hint">Connected to POST /chat</div>
            </div>
            <button id="send" type="submit">Send</button>
        </form>
    </main>

    <script>
        const form = document.getElementById("form");
        const prompt = document.getElementById("prompt");
        const messages = document.getElementById("messages");
        const send = document.getElementById("send");
        const statusEl = document.getElementById("status");

        function addMessage(text, role) {
            const el = document.createElement("div");
            el.className = `msg ${role}`;
            el.textContent = text;
            messages.appendChild(el);
            messages.scrollTop = messages.scrollHeight;
        }

        async function askAlice(text) {
            const res = await fetch("/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: text, user_id: "web_user", context: {} }),
            });
            if (!res.ok) {
                const detail = await res.text();
                throw new Error(`HTTP ${res.status}: ${detail}`);
            }
            return res.json();
        }

        form.addEventListener("submit", async (event) => {
            event.preventDefault();
            const text = prompt.value.trim();
            if (!text) return;

            addMessage(text, "user");
            prompt.value = "";
            send.disabled = true;
            statusEl.textContent = "thinking...";

            try {
                const payload = await askAlice(text);
                addMessage(payload.response || "(no response)", "alice");
                statusEl.textContent = "ready";
            } catch (error) {
                addMessage(`Error: ${error.message}`, "alice");
                statusEl.textContent = "error";
            } finally {
                send.disabled = false;
                prompt.focus();
            }
        });

        prompt.focus();
    </script>
</body>
</html>
"""
        )


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
