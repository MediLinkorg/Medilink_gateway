"""
LangGraph wiring for the MediLink gateway.

This file is the traffic map. It does not classify the request, call models
directly, retrieve evidence, or format the final response. Each node does that
work. The graph only decides which node runs next based on `state["route"]`.

Implemented routes:
  direct          : router -> END
  triage_question : router -> triage -> END
  htan_only       : router -> htan -> medical_llm -> report -> END
  rag_only        : router -> rag -> medical_llm -> report -> END
  htan_rag        : router -> htan -> rag -> medical_llm -> report -> END
  vision_only     : router -> vision -> medical_llm -> report -> END
  vision_rag      : router -> vision -> rag -> medical_llm -> report -> END

Full structure:

  User text/image
       |
       v
  FastAPI app/main.py
       |
       v
  Initial MediLinkState
       |
       v
  intent_router
       |
       +-- direct -----------> END
       |
       +-- triage_question --> triage --------------------------> END
       |                      Sonnet writes exact follow-up
       |                      questions. The user answers in a
       |                      later request with conversation_history.
       |
       +-- rag_only ---------> rag -----------> medical_llm --> report --> END
       |
       +-- htan_only --------> htan ----------> medical_llm --> report --> END
       |
       +-- htan_rag ---------> htan ----------> rag ---------> medical_llm --> report --> END
       |
       +-- vision_only ------> vision --------> medical_llm --> report --> END
       |
       +-- vision_rag -------> vision --------> rag ---------> medical_llm --> report --> END

Route ownership:
  - intent_router decides the route with Haiku.
  - triage uses Sonnet to decide the exact missing questions.
  - htan handles supported segmentation images only.
  - vision handles medical documents, radiology, and general medical images.
  - rag retrieves evidence.
  - medical_llm produces the final clinical answer.
  - report formats the final answer and doctor_report.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.nodes import (
    htan_node,
    intent_router_node,
    medical_llm_node,
    rag_node,
    report_generator_node,
    triage_node,
    vision_node,
)
from app.state import MediLinkState


def route_after_intent(state: MediLinkState) -> str:
    """
    Choose the first executable node after the Haiku router.

    The router writes `state["route"]`. This function converts that route into
    a graph node name. Routes that already have a complete user-facing response
    return "end". The conditional edge table below maps "end" to LangGraph END.
    """
    route = state.get("route") or "direct"
    if route == "triage_question":
        return "triage"
    if route == "htan_only":
        return "htan"
    if route == "rag_only":
        return "rag"
    if route == "htan_rag":
        return "htan"
    if route == "vision_only":
        return "vision"
    if route == "vision_rag":
        return "vision"
    return "end"


def route_after_htan(state: MediLinkState) -> str:
    """
    Decide what happens after HTAN.

    - htan_only: image segmentation is enough context, so go to Sonnet.
    - htan_rag : combine image findings with retrieved evidence, so go to RAG.
    """
    return "rag" if state.get("route") == "htan_rag" else "medical_llm"


def route_after_vision(state: MediLinkState) -> str:
    """
    Decide what happens after Haiku vision extraction.

    - vision_only: extracted document/image context is enough, so go to Sonnet.
    - vision_rag : enrich extracted context with retrieved evidence first.
    """
    return "rag" if state.get("route") == "vision_rag" else "medical_llm"


def route_after_rag(state: MediLinkState) -> str:
    """
    Decide what happens after RAG.

    Normal RAG paths continue to Sonnet. This keeps a safety valve: if a future
    RAG node sets route to "direct" with a final_answer, the graph can stop.
    """
    return "end" if state.get("route") == "direct" else "medical_llm"


def build_graph():
    """Build and compile the LangGraph state machine."""
    graph = StateGraph(MediLinkState)

    # Register node names and the Python functions that execute them.
    graph.add_node("intent_router", intent_router_node)
    graph.add_node("triage", triage_node)
    graph.add_node("htan", htan_node)
    graph.add_node("vision", vision_node)
    graph.add_node("rag", rag_node)
    graph.add_node("medical_llm", medical_llm_node)
    graph.add_node("report", report_generator_node)

    # Every request starts with Haiku routing.
    graph.set_entry_point("intent_router")

    # Router branch: direct/triage/HTAN/RAG/vision.
    graph.add_conditional_edges(
        "intent_router",
        route_after_intent,
        {"triage": "triage", "htan": "htan", "vision": "vision", "rag": "rag", "end": END},
    )

    # Triage stops after Sonnet asks the exact follow-up questions.
    graph.add_edge("triage", END)

    # HTAN branch: either go straight to Sonnet or enrich with RAG.
    graph.add_conditional_edges(
        "htan",
        route_after_htan,
        {"rag": "rag", "medical_llm": "medical_llm"},
    )

    # Vision branch: either go straight to Sonnet or enrich with RAG.
    graph.add_conditional_edges(
        "vision",
        route_after_vision,
        {"rag": "rag", "medical_llm": "medical_llm"},
    )

    # RAG usually feeds Sonnet, but may short-circuit in future direct cases.
    graph.add_conditional_edges(
        "rag",
        route_after_rag,
        {"medical_llm": "medical_llm", "end": END},
    )

    # Final answer path is always Sonnet -> report -> END.
    graph.add_edge("medical_llm", "report")
    graph.add_edge("report", END)

    return graph.compile()


# Compiled once at import; reused across requests by app/main.py.
medilink_graph = build_graph()
