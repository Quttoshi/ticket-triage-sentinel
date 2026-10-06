# TicketTriage Sentinel - Agent Arena (Assignment 1)

A bounded, single-agent **support-ticket triage** service built on FastAPI + LangChain + OpenAI.
Given a support request, the agent finds the right ticket in a **sandbox queue**, reads it,
records a category / priority / routing team, saves a **draft** reply or sets a sandbox status,
and reports back. It asks for clarification when the ticket is unclear, refuses out-of-scope
requests, and hands refunds, account deletion and email sending to a human.

- Student: Shehbaz · i212525 · Section A
- Domain: #11 Support-Ticket Triage
- Provider: OpenAI (default `gpt-4.1-mini`, alternative `gpt-4.1-nano`)

---

## 1. Problem statement and completion condition
Support teams spend time on first-line triage: working out which ticket a customer means, what kind
of problem it is, how urgent it is, who should own it, and drafting a first response. This agent
automates that **narrow** step on a sandbox queue (`data/sample_data.json`, tickets T101–T112).

**Completion condition (measurable):** the requested sandbox action (triage, draft, or status
change) on a ticket identified **by the user** (or found by search) has succeeded through the tool
gateway, and the model returns a validated `completed` decision. The system rejects `completed`
when no tool call has succeeded in the run.

## 2. Agent Design Canvas
| # | Element | Definition |
|---|---|---|
| 1 | Operational goal | Triage one support ticket per request on a sandbox queue: identify it, classify category + priority, route to a team, draft a reply or set status. |
| 2 | Completion condition | Requested sandbox action succeeded via the gateway and a validated `completed` decision with a user message is returned (≥1 successful tool result required). |
| 3 | System boundary | In-process sandbox copied per run. No real email, payments, accounts or external systems. Only the OpenAI API is called over the network. |
| 4 | Observations | User request, bounded chat history, untrusted `external_context`, ticket records and tool results, runtime state (step budget, ticket scope). |
| 5 | Actions / tools | `search_tickets`, `get_ticket` (read); `triage_ticket`, `draft_reply`, `set_ticket_status` (sandbox write); `issue_refund`, `delete_customer_account`, `send_email` (restricted, never executed). |
| 6 | State | `AgentState` (`app/models.py`): goal, step, repairs, model calls, tokens, user-supplied ids, search-discovered ids, inspected ids, requested-actions plan, attempted writes, completed actions, observations, tool traces, errors, events, faults fired, status, stop reason. |
| 7 | Autonomy boundary | Automatic: read, search, triage, draft, sandbox status. Approval required: refunds, account deletion, email sending. Blocked: requests outside ticket support. |
| 8 | Primary risks | Prompt injection in customer notes and ticket bodies; invented ticket ids; malformed model JSON; tool timeouts or bad output; runaway loops; uncontrolled API spend. |
| 9 | Evaluation criteria | Correct terminal status and stop reason per Arena category (A–F), correct tool selection, no forbidden tool executed, contract validity, latency, tokens, cost. |

## 3. Architecture
```
 Browser UI (app/static) ──POST /chat (session_id)──┐
 Arena evaluator ─────────POST /arena/run───────────┤
                                                    ▼
 app/api.py  model allow-list · per-IP chat rate limit · session memory hand-off
                                                    ▼
 app/arena.py  concurrency semaphore · 40 s run timeout · response size cap · logging
                                                    ▼
 app/agent.py  bounded loop (max_steps ≤ 6)
   ┌──────────────────────────────────────────────────────────────────────┐
   │ OBSERVE  prompts.build_messages(state, request, history, external)   │
   │ DECIDE   llm.OpenAIModel → raw text  [fault: invalid_agent_decision] │
   │ PLAN     1st decision declares requested_actions (untrusted withheld) │
   │ VALIDATE parse JSON → AgentDecision schema → semantic rules          │
   │          invalid → repair prompt (≤2, counts as a step) → contract_error
   │ ROUTE    needs_clarification / blocked / approval_required / completed → STOP
   │          restricted tool → approval_required (never executed)         │
   │ ACT      tools.ToolGateway  [faults: tool_timeout, malformed_output] │
   │          retry ≤2 → tool_error;  business rejection → observation     │
   │ UPDATE   state + observation → next step;  no steps left → budget_exceeded
   └──────────────────────────────────────────────────────────────────────┘
```

### Folder map
```
i212525/
├── app/
│   ├── main.py      FastAPI app, logging, lifespan (memory, busy set)
│   ├── config.py    env-driven settings and limits
│   ├── api.py       /, /health, /arena/manifest, /arena/run, /models, /chat, DELETE /chat/{id}
│   ├── arena.py     common boundary: concurrency, timeout, size cap
│   ├── agent.py     bounded loop, validation, autonomy, stop rules
│   ├── models.py    Arena contracts + AgentDecision, tool args, ToolResult, AgentState
│   ├── prompts.py   system policy + dynamic per-step template + repair template
│   ├── llm.py       OpenAI adapter, usage/cost, daily cap, ScriptedModel for tests
│   ├── tools.py     sandbox, tool functions, registry, ToolGateway (faults, retries, traces)
│   ├── memory.py    bounded LangChain message history per session
│   └── static/      minimal chat UI
├── data/sample_data.json        fictional ticket queue
├── evaluation/                  public cases, HTTP runner, model comparison
├── tests/test_agent.py          offline scripted-model tests (no API cost)
└── arena_manifest.json, render.yaml, Dockerfile, requirements.txt, .env.example, run.py
```

## 4. Autonomy boundary
| Boundary | Actions | How it is enforced |
|---|---|---|
| Automatic | search, read, triage, draft reply (saved, never sent), sandbox status change | Tool kind `read`/`write` in `tools.TOOLS`; per-run sandbox copy |
| Approval required | `issue_refund`, `delete_customer_account`, `send_email` | Tool kind `restricted`: the gateway has no implementation; the loop stops with `approval_required` / `restricted_action_requires_approval` and records a `rejected` trace |
| Blocked | anything outside ticket support or harmful | Model returns `blocked`; validator requires a user message |
| System-enforced guards | **trusted-only plan**: write/restricted tools must be in `requested_actions`, declared before untrusted text is shown; a requested restricted action stops at once with `approval_required`; writes need a ticket the **user** named or **search** returned; ticket must be read first; category→team compatibility; no repeated identical action; no `completed` while requested actions are pending | `agent.validate_decision`, `agent.run_agent` |

## 5. Model selection experiment
Two OpenAI tiers on the same 12 public cases (`evaluation/public_cases.json`, covering baseline plus
Arena categories A–F), run with `python evaluation/model_comparison.py`. Results are written to
`evaluation/model_comparison.md`/`.json`.

Run on 2026-10-04 (full per-case table in `evaluation/model_comparison.md`):

| Model | Task success | Structured-output validity | Correct action | Avg latency (ms) | Avg in/out tokens per run | Est. cost / run (USD) |
|---|---|---|---|---|---|---|
| gpt-4.1-nano | 11/12 | 94% (35 decisions) | 12/12 | 3625 | 4530 / 212 | 0.00054 |
| **gpt-4.1-mini** | **12/12** | **100%** (33 decisions) | 12/12 | 3209 | 4221 / 208 | 0.00202 |

**Selection: `gpt-4.1-mini`.** In development, nano was also *unstable* even at `temperature=0`.
The injection case passed in one run and failed in the next: nano kept trying to act on the injected
ticket id until the repair limit stopped it. On the destructive-request case it asked for
clarification instead of escalating. Mini passed every case, produced no invalid decisions, and was
slightly faster. It costs about 4× more per run, but that is still ≈ $0.002, or about $2 per 1,000
runs. Reliability is the main requirement, so the extra cost is justified. Nano stays in the UI
selector for comparison.

**How the choice is made:** pick the cheaper model unless the larger one improves task success or
contract validity on these cases. A bigger model gets no credit just for being bigger. The task needs
short, schema-bound JSON decisions with ≤6 calls per run, a small context (≈2–4k input tokens) and
low latency to fit the 40 s budget. Non-reasoning models with `temperature=0` keep output tokens
and latency predictable (≈200 output tokens per run).

**Cost limits:** ≤6 model decisions per run (repairs included), `MAX_OUTPUT_TOKENS=512`, ≤2 tool
retries, ≤2 repairs, ≤12 history messages / 24k characters, external context capped at 12k
characters, `MAX_MODEL_CALLS_PER_DAY=3000`, 4 concurrent runs, 20 chat requests per minute per IP.
Cost is estimated from OpenAI usage metadata × list price (`app/llm.py: PRICES`) and is reported as
`null` when the provider doesn't return usage.

## 6. Prompt and context engineering
Each step's messages are assembled by `prompts.build_messages`:

| Layer | Where | Content |
|---|---|---|
| SYSTEM | `SystemMessage(SYSTEM_PROMPT)` | Role, tool catalogue **generated from the tool registry and Pydantic schemas**, autonomy rules, trust rules, output contract |
| HISTORY | LangChain `HumanMessage`/`AIMessage` from `memory.py` | Last ≤6 turns of this chat session only |
| USER | `<current_user_request>` | Current request |
| PLAN | first decision | `requested_actions` is decided while untrusted context is **withheld** (`WITHHELD` placeholder), then locked in state |
| STATE | `<runtime_state>` | Step x of N, decisions remaining, user-supplied ids, discovered ids, inspected tickets, completed actions |
| UNTRUSTED | `<untrusted_external_context>` | Each item labelled `source=… trust=untrusted`; `<`/`>` escaped so content cannot close or forge tags; length-capped with a logged truncation event |
| OBSERVATIONS | `<tool_observations>` | JSON tool results, escaped; ticket reads carry a "customer-written data" notice |
| REPAIR | `<contract_violation>` (only after a rejection) | Validator error + truncated rejected output |

`TURN_TEMPLATE` and `REPAIR_TEMPLATE` are `string.Template`s filled from state every step (the
dynamically assembled prompt). **Trust rule:** only the system prompt and the user's own messages
carry instructions. The rule also has a mechanical backstop, so it doesn't rely on the model alone:
1. The plan (which write actions are allowed) is fixed before any untrusted text reaches the model, so
   injected text cannot add actions such as closing a ticket or issuing a refund.
2. Ticket ids that appear only in untrusted text can't be targeted by write actions.
3. Restricted actions never execute.

Drafts are told never to confirm refunds or other actions that were not performed.

## 7. Conversation memory policy
- `session_id` (browser `crypto.randomUUID()`) links chat turns; **New chat** calls
  `DELETE /chat/{id}` and starts a new id. Reloading the page also starts a new session.
- `memory.py` keeps the last 6 turns (12 messages, ≤24,000 characters), at most 100 sessions, in
  process memory. **It is lost on restart**, so the service runs with **one worker**. Render free
  instances sleep and restart, which clears all chats.
- Ticket ids written by the user in earlier turns count as user-supplied, so "Close my ticket" →
  "Which id?" → "T102" continues the original goal (tested in
  `test_multi_turn_clarification_continues_goal`).
- `/arena/run` requests are independent: no memory, and the sandbox is a fresh copy per run.
- Sandbox changes don't persist between runs.

## 8. Structured output schema
```python
class AgentDecision(BaseModel):          # extra fields forbidden
    status: Literal['continue','needs_clarification','completed','blocked','approval_required']
    action: str | None                   # a tool name, only with status=continue
    arguments: dict                      # validated against that tool's Pydantic args model
    reasoning: str                       # ≤400 chars, logged
    user_message: str | None             # required for every non-continue status
    requested_actions: list[str] | None  # 1st decision: write/restricted tools the user asked for
```
Tool argument models: `SearchTicketsArgs`, `TicketRef`, `TriageTicketArgs` (enums for category,
priority, team; `ticket_id` must match `^T\d{3}$`), `DraftReplyArgs`, `SetStatusArgs`,
`RefundArgs`, `DeleteAccountArgs`, `SendEmailArgs`. Tool outputs must validate as
`ToolResult(ok, data, error)`. OpenAI is called in JSON mode (`response_format=json_object`), and the application still parses and
validates every output itself.

## 9. Validation rules (contract-gated execution)
1. **Parse:** strip code fences, `json.loads`, must be an object.
2. **Schema:** `AgentDecision` (types, enums, lengths, no extra keys).
3. **Semantic:**
   - The first `continue`/`completed` decision must declare `requested_actions` (write/restricted
     tools only). A run that ends on its first decision doesn't need one.
   - Write/restricted tools must be in `requested_actions`. If a restricted tool is requested, the run
     stops with `approval_required` before anything executes.
   - `continue` needs a known tool, and its arguments must validate.
   - `needs_clarification` is rejected while ticket ids the user supplied haven't been looked up.
   - Non-`continue` statuses need `action=null` and a non-empty `user_message`.
   - Write/restricted actions need a ticket the user supplied or search returned.
   - Writes need the ticket to have been read with `get_ticket` first.
   - Triage team must be allowed for the category (`ALLOWED_TEAMS`).
   - No repeat of an action that already succeeded.
   - `completed` needs ≥1 successful tool result, every ticket id it mentions must exist, and every
     requested write action must have been attempted.
4. **Recovery:** a rejected decision is **never executed**. The error is fed back via the repair
   template. At most 2 repairs, and each one counts as a step.
5. **Typed failure:** the 3rd violation → `contract_error` / `repair_limit_reached`.

## 10. Stopping conditions
| Status | stop_reason | Trigger |
|---|---|---|
| completed | goal_completed | validated completion |
| needs_clarification | clarification_requested | model asks a question (e.g. ticket not identifiable) |
| blocked | outside_agent_authority | out-of-scope / unsafe request |
| approval_required | human_approval_required / restricted_action_requires_approval | model escalates, or selects a restricted tool |
| tool_error | tool_failed_after_retries | tool failed on all 1+2 attempts |
| contract_error | repair_limit_reached | >2 invalid decisions |
| budget_exceeded | step_budget_reached / time_budget_reached / daily_model_call_cap_reached | budgets |
| failed | model_unavailable / internal_error / response_too_large | provider missing or failing twice; unexpected errors |

## 11. Fault injection and tool handling
`arena_config.fault` triggers once per run, on the first matching operation:
- `invalid_agent_decision`: the first model output is replaced with one of four corruptions
  (truncated JSON, wrong type, invalid enum, unknown tool), chosen deterministically from
  `request_id`. The validator rejects it and the repair path recovers.
- `tool_timeout`: the first gateway call raises a timeout and is retried.
- `malformed_tool_output`: the first gateway call returns a payload that fails `ToolResult`
  validation and is retried.

Real tool calls run with a 5 s timeout. A business rejection (e.g. `ticket_not_found`,
`ticket_closed`) isn't treated as a failure: it goes back to the model as an observation, so the
model can ask the user or choose another action.

## 12. Evidence
- `tests/test_agent.py`: 25 offline tests using a scripted model, covering: happy path, search
  flow, invented-id rejection, **multi-turn clarification over `/chat`**, injection via external
  context (including tag forgery), injection leading to a refund attempt, all four corruption
  variants, repair limit, semantic rules, completion without evidence, tool timeout and malformed
  output, persistent tool failure, model failure, step budget, repairs counting against budget,
  approval and blocked statuses, plan enforcement (unrequested write rejected, injected refund rejected
  while the goal continues, requested refund escalated, pending actions block completion, missing or
  invalid plan), and lookup-before-clarify.
- **Live results (gpt-4.1-mini, local server):** 12/12 public cases pass
  (`evaluation/results_public.json`). The live two-turn clarification ("Please close my ticket." →
  asks for the id → "T108" → `get_ticket`, `set_ticket_status` → completed) is in
  `evaluation/multi_turn_result.json`.
- `evaluation/public_cases.json` + `run_public_tests.py`: 12 HTTP cases with expected statuses,
  required tools and forbidden tools. Results go to `evaluation/results_public.json` (local) and
  `evaluation/results_deployed.json` (Render).
- `evaluation/model_comparison.md`: the model experiment.

## 13. Run locally (Windows, Python 3.12)
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env        # then put your key in OPENAI_API_KEY inside .env
.\.venv\Scripts\python.exe run.py  # http://127.0.0.1:8000/  and /docs
```
Tests (free, no key): `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`
Public HTTP cases (uses the model): `.\.venv\Scripts\python.exe evaluation/run_public_tests.py --url http://127.0.0.1:8000`
Model comparison: `.\.venv\Scripts\python.exe evaluation/model_comparison.py`

Environment variables: see `.env.example` (`MODEL_NAME`, `ALLOWED_MODELS`, `OPENAI_API_KEY`, limits).

## 14. Deploy on Render
1. Push this folder to a **private** GitHub repo and invite the instructor.
2. In Render, create a Web Service from the repo (Python runtime, or use the `render.yaml` Blueprint).
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1`
   - Health check: `/health`
3. Add `OPENAI_API_KEY` as a secret environment variable (plus `MODEL_NAME` if you want to change it).
4. Verify: `GET /health` shows `"model_configured": true`, `GET /arena/manifest`, a POST to
   `/arena/run` via `/docs`, and the UI clarification flow in a private window. Then run
   `run_public_tests.py --url https://<app>.onrender.com --out evaluation/results_deployed.json`.

## 15. Limitations
- In-memory chat history and sandbox: lost on restart. A single worker is required.
- Render free tier sleeps after inactivity, so the first request can take ~30–60 s to cold start.
- The keyword search is simple, not semantic.
- Ticket-id scope is detected with a `T\d{3}` pattern, so the user must write ids in that form
  (lower case is accepted).
- Injection resistance for *read-only* behaviour and for the wording of drafts depends on the model
  following the trust rules. The system mechanically guarantees that untrusted text can't trigger
  writes on tickets outside the user's scope or any restricted action.
- Prices in `PRICES` are approximate list prices; OpenAI model availability and names can change
  (set `MODEL_NAME` / `ALLOWED_MODELS`).
