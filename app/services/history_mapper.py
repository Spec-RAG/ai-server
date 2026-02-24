from __future__ import annotations

from langchain_core.messages import HumanMessage, AIMessage


def build_history_messages(history: list=[]) -> list:
    """Convert ChatRequest.history items to LangChain HumanMessage/AIMessage."""
    messages = []
    for h in history or []:
        role = h.role if hasattr(h, "role") else h.get("role")
        content = h.content if hasattr(h, "content") else h.get("content")
        if role == "human":
            messages.append(HumanMessage(content=content))
        elif role == "ai":
            messages.append(AIMessage(content=content))
    return messages
