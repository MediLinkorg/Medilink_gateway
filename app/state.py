"""
Shared LangGraph state for the gateway.

Every node receives a MediLinkState dictionary and returns a partial dictionary
with only the fields it wants to add or update. LangGraph merges those partial
updates into the shared state as the request moves through the pipeline.

Trace rule:
  - main.py creates the input fields.
  - intent_router.py writes routing fields.
  - triage_node.py writes final triage questions.
  - htan_node.py writes HTAN image-analysis fields.
  - vision_node.py writes document/radiology/general-image fields.
  - rag_node.py writes evidence fields.
  - medical_llm.py writes draft answer and citation checks.
  - report_generator.py writes final_answer and doctor_report.
"""

from __future__ import annotations

from typing import Optional, TypedDict


class MediLinkState(TypedDict, total=False):
    # Input fields created by app/main.py.
    user_message: str
    image_bytes: Optional[bytes]
    image_media_type: Optional[str]  # Upload MIME type, e.g. image/jpeg or image/png.
    image_path: Optional[str]
    user_id: str
    session_id: str
    conversation_history: list  # [{"role": "user"|"assistant", "content": str}, ...]
    patient_mode: Optional[bool]

    # Router fields written by app/nodes/intent_router.py.
    intent: Optional[str]        # greeting | symptom_description | medication | general | ...
    route: Optional[str]         # direct | triage_question | rag_only | htan_only |
                                 # htan_rag | vision_only | vision_rag
    safety_level: Optional[str]  # none | urgent | emergency | crisis

    # Router booleans are trace/debug flags. graph.py uses `route` for control.
    needs_rag: Optional[bool]
    needs_htan: Optional[bool]
    needs_vision: Optional[bool]
    needs_triage: Optional[bool]

    # Haiku router can suggest missing questions, but this is not the final
    # patient-facing triage list. triage_node.py uses these as hints.
    router_triage_questions: Optional[list[str]]

    # Sonnet triage output written by app/nodes/triage_node.py.
    # This is what the API returns when route == "triage_question".
    triage_questions: Optional[list[str]]

    # Image classification/routing fields from intent_router.py.
    image_type: Optional[str]    # medical_document | xray | dermoscopy | histology | ...
    modality: Optional[str]      # HTAN-supported only: dermoscopy | histology | microscopy

    # Optional retrieval guidance from Haiku router. rag_node.py prefers this
    # over the raw user message when it is provided.
    rag_query: Optional[str]

    # Short explanation of why Haiku chose the route; useful for debugging and
    # doctor_report traceability.
    router_reason: Optional[str]

    # HTAN output written by app/nodes/htan_node.py.
    cv_result: Optional[dict]    # Raw structured JSON from the HTAN service.
    cv_text: Optional[str]       # Natural-language rendering of cv_result.

    # Vision output written by app/nodes/vision_node.py.
    vision_result: Optional[dict]  # Raw structured JSON from Haiku vision.
    vision_text: Optional[str]     # Extracted document text or image description.

    # RAG output written by app/nodes/rag_node.py.
    rag_query_used: Optional[str]    # Final retrieval query sent to the RAG service.
    retrieved_docs: Optional[list]  # Evidence chunks returned by the RAG service.
    rag_context: Optional[str]      # Formatted evidence string passed to Sonnet.

    # Generation output written by app/nodes/medical_llm.py.
    draft_answer: Optional[str]     # Sonnet's answer before report formatting.
    citations: Optional[dict]       # Citation verification report.

    # Final formatting output written by app/nodes/report_generator.py.
    doctor_report: Optional[dict]   # Structured clinician-facing trace/report.
    final_answer: Optional[str]     # Final user-facing response.

    # Error field set by any node that catches a recoverable failure.
    error: Optional[str]
