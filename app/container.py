from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai.contracts import (
    CallableMemoryAdapter,
    CallableResponseAdapter,
    CallableRoutingAdapter,
    CallableToolAdapter,
    RuntimeBoundaries,
    MemoryRequest,
    MemoryResult,
    ResponseOutput,
    ResponseRequest,
    RouterDecision,
    RouterRequest,
    ToolInvocation,
    ToolResult,
)
from ai.core.llm_engine import LLMConfig, LocalLLMEngine
from ai.core.nlp_processor import NLPProcessor
from ai.memory.vector_store import VectorMemory
from ai.runtime.contract_pipeline import ContractPipeline

from app.config import Settings


@dataclass
class AppContainer:
    settings: Settings
    nlp: NLPProcessor
    llm: LocalLLMEngine
    pipeline: ContractPipeline


def _build_boundaries(nlp: NLPProcessor, llm: LocalLLMEngine, settings: Settings) -> RuntimeBoundaries:
    memory_store: list[dict[str, Any]] = []
    vector_memory: VectorMemory | None = None
    if settings.memory_backend == "chroma":
        try:
            vector_memory = VectorMemory()
        except Exception:
            vector_memory = None

    def route_fn(request: RouterRequest) -> RouterDecision:
        processed = nlp.process(request.user_input)
        route = "tool" if processed.intent.startswith(("weather:", "notes:", "email:", "calendar:")) else "conversation"
        return RouterDecision(
            route=route,
            intent=processed.intent,
            confidence=float(processed.intent_confidence),
            metadata={"keywords": list(processed.keywords)},
        )

    def recall_fn(request: MemoryRequest) -> MemoryResult:
        if vector_memory is not None:
            try:
                import asyncio

                items = asyncio.run(vector_memory.query(request.query, n_results=request.max_items))
                return MemoryResult(items=items, source="chroma", confidence=0.7)
            except Exception:
                pass
        matched = [item for item in memory_store if request.query.lower() in str(item.get("content", "")).lower()]
        return MemoryResult(items=matched[-request.max_items :], source="in_memory", confidence=0.5)

    def store_fn(item: dict[str, Any]) -> None:
        if vector_memory is not None:
            try:
                import asyncio

                trace_id = str(item.get("trace_id", ""))
                text = str(item.get("content", ""))
                asyncio.run(vector_memory.store(trace_id=trace_id, text=text, metadata=item))
                return
            except Exception:
                pass
        memory_store.append(dict(item))

    def tool_fn(invocation: ToolInvocation) -> ToolResult:
        return ToolResult(
            success=False,
            tool_name=invocation.tool_name,
            action=invocation.action,
            error="Tool execution not wired in API mode yet",
            confidence=0.0,
        )

    def response_fn(request: ResponseRequest) -> ResponseOutput:
        answer = llm.chat(request.user_input)
        return ResponseOutput(text=answer, confidence=0.8)

    return RuntimeBoundaries(
        routing=CallableRoutingAdapter(route_fn=route_fn),
        memory=CallableMemoryAdapter(recall_fn=recall_fn, store_fn=store_fn),
        tools=CallableToolAdapter(execute_fn=tool_fn),
        response=CallableResponseAdapter(generate_fn=response_fn),
        verifier=None,
    )


def build_container(settings: Settings) -> AppContainer:
    nlp = NLPProcessor()
    llm = LocalLLMEngine(
        LLMConfig(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=settings.temperature,
            max_history=settings.max_history,
        )
    )
    boundaries = _build_boundaries(nlp=nlp, llm=llm, settings=settings)
    pipeline = ContractPipeline(boundaries=boundaries)
    return AppContainer(settings=settings, nlp=nlp, llm=llm, pipeline=pipeline)
