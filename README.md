# MediLink Gateway / Orchestrator

The gateway is the public entrypoint for the MediLink app. It receives text
and/or image uploads, routes the request with Haiku, optionally calls HTAN and
RAG services, generates one final Sonnet answer, and returns both the answer and
trace metadata.

## Routes

```text
direct          router -> END
triage_question router -> triage -> END
htan_only       router -> htan -> medical_llm -> report -> END
rag_only        router -> rag -> medical_llm -> report -> END
htan_rag        router -> htan -> rag -> medical_llm -> report -> END
vision_only     router -> vision -> medical_llm -> report -> END
vision_rag      router -> vision -> rag -> medical_llm -> report -> END
```

Route ownership:

- `intent_router.py` decides route, safety level, image type, RAG need, HTAN need,
  vision need, and whether triage follow-up is needed.
- `triage_node.py` uses Sonnet to write exact follow-up questions and then stops
  the graph until the user answers in a later request.
- `htan_node.py` handles segmentation-capable modalities only:
  dermoscopy, histology, microscopy by default.
- `vision_node.py` handles medical documents, radiology, and general medical
  images.
- `rag_node.py` retrieves evidence from the RAG service.
- `medical_llm.py` generates the final clinical answer with Sonnet.
- `report_generator.py` builds the final answer and structured doctor report.

## API

```text
GET  /health
POST /api/v1/agent
```

`POST /api/v1/agent` accepts multipart form fields:

- `message`: optional user text
- `image`: optional uploaded image/document
- `user_id`: trace identifier
- `session_id`: trace identifier
- `history`: JSON list of `{role, content}` turns
- `patient_mode`: boolean passed to RAG retrieval

Example:

```bash
curl -s -F "message=What are the early signs of melanoma?" \
  http://localhost:8000/api/v1/agent | jq

curl -s -F "message=What does this report mean?" -F "image=@report.png" \
  http://localhost:8000/api/v1/agent | jq
```

Response fields include:

- `answer`
- `intent`
- `route`
- `safety_level`
- `image_type`
- `modality`
- `router_reason`
- `router_triage_questions`
- `triage_questions`
- `rag_query_used`
- `doctor_report`
- `error`

## Run

For the whole local stack, use the root `docker-compose.yml` one level up.

Standalone gateway:

```bash
cp .env.example .env
docker compose up --build
```

The gateway needs the HTAN and RAG services reachable through:

```text
HTAN_SERVICE_URL=http://htan:8001
RAG_SERVICE_URL=http://rag:8002
```

## Layout

```text
app/
  main.py        FastAPI /api/v1/agent and /health
  config.py      Environment config, model names, service URLs, safety text
  state.py       MediLinkState TypedDict shared by graph nodes
  graph.py       LangGraph traffic controller
  clients.py     HTTP clients for HTAN and RAG services
  llm.py         Anthropic/Gemini provider boundary
  schemas.py     Public API response schemas
  nodes/
    intent_router.py
    triage_node.py
    htan_node.py
    vision_node.py
    rag_node.py
    quality_gate.py
    medical_llm.py
    report_generator.py
tests/
  test_config.py
  test_clients.py
  test_llm.py
  test_schemas.py
  test_intent_router.py
  test_quality_gate.py
  test_graph.py
```

## Notes

- No hardcoded emergency keyword list is used. Haiku owns safety routing.
- No generic medical disclaimers are injected into prompts. Safety responses are
  configurable through `EMERGENCY_RESPONSE` and `CRISIS_RESPONSE`.
- RAG is currently skin-cancer/skin-topic focused, but the gateway routes are
  general medical and can support broader RAG corpora over time.
- `doctor_report` is an informational trace/debug/clinician report, not an
  approval step.

## Tests

```bash
pip install -r requirements.txt
pytest -q
python -m compileall app tests
```
