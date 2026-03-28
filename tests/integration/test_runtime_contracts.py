from ai.contracts import (
    CallableMemoryAdapter,
    CallableResponseAdapter,
    CallableRoutingAdapter,
    CallableToolAdapter,
    MemoryRequest,
    MemoryResult,
    ResponseOutput,
    ResponseRequest,
    RouterDecision,
    RouterRequest,
    ToolInvocation,
    ToolResult,
)


def test_callable_routing_adapter_routes_request():
    adapter = CallableRoutingAdapter(
        route_fn=lambda req: RouterDecision(
            route="llm",
            intent="conversation:general",
            confidence=0.91,
            metadata={"turn": req.turn_number},
        )
    )

    result = adapter.route(RouterRequest(user_input="hello", turn_number=3))
    assert result.route == "llm"
    assert result.metadata["turn"] == 3


def test_callable_memory_adapter_recall_and_store():
    stored = []
    adapter = CallableMemoryAdapter(
        recall_fn=lambda req: MemoryResult(
            items=[{"text": req.query}],
            source="memory",
            confidence=0.8,
        ),
        store_fn=lambda item: stored.append(item),
    )

    recall = adapter.recall(MemoryRequest(query="project status", user_id="u1"))
    adapter.store({"text": "new item"})

    assert recall.items[0]["text"] == "project status"
    assert stored == [{"text": "new item"}]


def test_callable_tool_adapter_executes_tool_invocation():
    adapter = CallableToolAdapter(
        execute_fn=lambda inv: ToolResult(
            success=True,
            tool_name=inv.tool_name,
            action=inv.action,
            data={"ok": True},
            confidence=0.9,
        )
    )

    result = adapter.execute(ToolInvocation(tool_name="notes", action="search"))
    assert result.success is True
    assert result.tool_name == "notes"


def test_callable_response_adapter_generates_response():
    adapter = CallableResponseAdapter(
        generate_fn=lambda req: ResponseOutput(
            text=f"intent={req.decision.intent}",
            confidence=0.88,
        )
    )

    output = adapter.generate(
        ResponseRequest(
            user_input="hi",
            decision=RouterDecision(route="llm", intent="conversation:general", confidence=0.8),
            memory=MemoryResult(items=[]),
        )
    )
    assert output.text == "intent=conversation:general"
    assert output.confidence == 0.88
