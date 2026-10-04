"""Domain-independent HTTP contracts. Add your own decision/state models below."""
from typing import Literal
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field, field_validator
class Contract(BaseModel):
    model_config = ConfigDict(extra='forbid')
class ExternalContext(Contract):
    source: str = Field(min_length=1, max_length=100)
    content: str = Field(max_length=10000)
    trust: Literal['untrusted'] = 'untrusted'
class Fault(Contract):
    type: Literal['none', 'tool_timeout', 'malformed_tool_output', 'invalid_agent_decision'] = 'none'
    trigger: Literal['first_matching_operation'] = 'first_matching_operation'
class ArenaConfig(Contract):
    max_steps: int = Field(default=6, ge=1, le=6)
    fault: Fault = Field(default_factory=Fault)
    @field_validator('fault', mode='before')
    @classmethod
    def accept_assignment_shorthand(cls, value):
        return {'type': value} if isinstance(value, str) else value
class ArenaRequest(Contract):
    arena_version: Literal['0.1'] = '0.1'
    request_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1, max_length=80)
    task: str = Field(min_length=1, max_length=10000)
    external_context: list[ExternalContext] = Field(default_factory=list, max_length=20)
    arena_config: ArenaConfig = Field(default_factory=ArenaConfig)
    @field_validator('task')
    @classmethod
    def nonblank(cls, value):
        if not value.strip(): raise ValueError('Task must not be blank')
        return value
class ToolTrace(Contract):
    step: int = Field(ge=1, le=6)
    tool: str
    attempt: int = Field(default=1, ge=1, le=3)
    outcome: Literal['success', 'timeout', 'malformed_output', 'rejected', 'exception']
    latency_ms: float = Field(default=0, ge=0)
class Metrics(Contract):
    latency_ms: float = Field(default=0, ge=0)
    model_calls: int = Field(default=0, ge=0, le=6)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)
class ArenaResponse(Contract):
    arena_version: Literal['0.1'] = '0.1'
    request_id: str
    status: Literal['completed', 'needs_clarification', 'blocked', 'approval_required', 'tool_error', 'contract_error', 'budget_exceeded', 'failed']
    final_response: str = Field(min_length=1, max_length=2000)
    steps: int = Field(default=0, ge=0, le=6)
    stop_reason: str
    tool_calls: list[ToolTrace] = Field(default_factory=list)
    errors: list[dict] = Field(default_factory=list)
    events: list[dict] = Field(default_factory=list)
    metrics: Metrics = Field(default_factory=Metrics)
class ChatRequest(ArenaRequest):
    session_id: str = Field(min_length=16, max_length=80)
    model: str = Field(default='', max_length=120)


TICKET_ID_PATTERN = r'^T\d{3}$'
Category = Literal['billing', 'technical', 'account', 'shipping', 'general']
Priority = Literal['low', 'medium', 'high', 'urgent']
Team = Literal['billing_team', 'tech_support', 'account_security', 'logistics', 'customer_care']
TicketStatus = Literal['open', 'pending', 'resolved', 'closed']
DecisionStatus = Literal['continue', 'needs_clarification', 'completed', 'blocked', 'approval_required']

ALLOWED_TEAMS: dict[str, set[str]] = {
    'billing': {'billing_team', 'customer_care'},
    'technical': {'tech_support'},
    'account': {'account_security', 'customer_care'},
    'shipping': {'logistics', 'customer_care'},
    'general': {'customer_care'},
}


class AgentDecision(Contract):
    status: DecisionStatus
    action: str | None = Field(default=None, max_length=40)
    arguments: dict = Field(default_factory=dict)
    reasoning: str = Field(default='', max_length=400)
    user_message: str | None = Field(default=None, max_length=1500)
    requested_actions: list[str] | None = Field(default=None, max_length=8)


class SearchTicketsArgs(Contract):
    query: str = Field(min_length=2, max_length=120)


class TicketRef(Contract):
    ticket_id: str = Field(pattern=TICKET_ID_PATTERN)


class TriageTicketArgs(TicketRef):
    category: Category
    priority: Priority
    team: Team


class DraftReplyArgs(TicketRef):
    body: str = Field(min_length=10, max_length=1200)


class SetStatusArgs(TicketRef):
    status: Literal['pending', 'resolved', 'closed']
    note: str = Field(default='', max_length=300)


class RefundArgs(TicketRef):
    amount_usd: float = Field(gt=0, le=10000)


class DeleteAccountArgs(Contract):
    customer_id: str = Field(pattern=r'^C\d{3}$')


class SendEmailArgs(TicketRef):
    body: str = Field(min_length=1, max_length=1200)


class ToolResult(Contract):
    ok: bool
    data: dict = Field(default_factory=dict)
    error: str | None = None


class AgentState(BaseModel):
    request_id: str
    goal: str
    max_steps: int
    step: int = 0
    repairs: int = 0
    model_calls: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    user_scope_ids: set[str] = Field(default_factory=set)
    discovered_ids: set[str] = Field(default_factory=set)
    inspected_ids: set[str] = Field(default_factory=set)
    failed_lookups: set[str] = Field(default_factory=set)
    plan: set[str] | None = None
    attempted_writes: set[str] = Field(default_factory=set)
    completed_actions: list[tuple[str, str]] = Field(default_factory=list)
    observations: list[dict] = Field(default_factory=list)
    tool_calls: list[ToolTrace] = Field(default_factory=list)
    errors: list[dict] = Field(default_factory=list)
    events: list[dict] = Field(default_factory=list)
    last_violation: str | None = None
    last_invalid_output: str | None = None
    faults_fired: set[str] = Field(default_factory=set)
    status: str = 'running'
    stop_reason: str | None = None
