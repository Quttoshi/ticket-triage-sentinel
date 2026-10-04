import asyncio
import copy
import json
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Literal
from pydantic import BaseModel, ValidationError
from app.config import ROOT, settings
from app.models import (ALLOWED_TEAMS, DeleteAccountArgs, DraftReplyArgs, RefundArgs,
    SearchTicketsArgs, SendEmailArgs, SetStatusArgs, TicketRef, ToolResult, ToolTrace,
    TriageTicketArgs)

SAMPLE = json.loads((ROOT / 'data/sample_data.json').read_text(encoding='utf-8'))
TOOL_FAULTS = {'tool_timeout', 'malformed_tool_output'}


class Sandbox:
    def __init__(self):
        self.tickets = {t['id']: t for t in copy.deepcopy(SAMPLE['tickets'])}
        self.drafts: dict[str, str] = {}

    def exists(self, ticket_id: str) -> bool:
        return ticket_id in self.tickets


def search_tickets(box: Sandbox, a: SearchTicketsArgs) -> ToolResult:
    words = [w for w in re.findall(r'\w+', a.query.lower()) if len(w) > 2]
    scored = []
    for t in box.tickets.values():
        text = ' '.join(str(t[k]) for k in ('id', 'customer_id', 'customer', 'subject', 'body')).lower()
        score = sum(w in text for w in words)
        if score: scored.append((score, t['id']))
    scored.sort(key=lambda x: (-x[0], x[1]))
    matches = [{k: box.tickets[i][k] for k in ('id', 'customer', 'subject', 'status')} for _, i in scored[:5]]
    return ToolResult(ok=True, data={'matches': matches, 'count': len(matches)})


def get_ticket(box: Sandbox, a: TicketRef) -> ToolResult:
    if not box.exists(a.ticket_id):
        return ToolResult(ok=False, error=f'ticket_not_found: {a.ticket_id}')
    ticket = dict(box.tickets[a.ticket_id])
    ticket['draft_reply'] = box.drafts.get(a.ticket_id)
    return ToolResult(ok=True, data={'ticket': ticket})


def triage_ticket(box: Sandbox, a: TriageTicketArgs) -> ToolResult:
    if not box.exists(a.ticket_id):
        return ToolResult(ok=False, error=f'ticket_not_found: {a.ticket_id}')
    t = box.tickets[a.ticket_id]
    if t['status'] == 'closed':
        return ToolResult(ok=False, error='ticket_closed: closed tickets cannot be re-triaged')
    if a.team not in ALLOWED_TEAMS[a.category]:
        return ToolResult(ok=False, error=f'invalid_route: {a.category} cannot go to {a.team}')
    t.update(category=a.category, priority=a.priority, team=a.team)
    return ToolResult(ok=True, data={'ticket_id': a.ticket_id, 'category': a.category,
                                     'priority': a.priority, 'team': a.team})


def draft_reply(box: Sandbox, a: DraftReplyArgs) -> ToolResult:
    if not box.exists(a.ticket_id):
        return ToolResult(ok=False, error=f'ticket_not_found: {a.ticket_id}')
    if box.tickets[a.ticket_id]['status'] == 'closed':
        return ToolResult(ok=False, error='ticket_closed: no replies on closed tickets')
    box.drafts[a.ticket_id] = a.body
    return ToolResult(ok=True, data={'ticket_id': a.ticket_id, 'draft_saved': True, 'sent': False})


def set_ticket_status(box: Sandbox, a: SetStatusArgs) -> ToolResult:
    if not box.exists(a.ticket_id):
        return ToolResult(ok=False, error=f'ticket_not_found: {a.ticket_id}')
    t = box.tickets[a.ticket_id]
    if t['status'] == 'closed':
        return ToolResult(ok=False, error='ticket_closed: already closed')
    old, t['status'] = t['status'], a.status
    return ToolResult(ok=True, data={'ticket_id': a.ticket_id, 'old_status': old, 'new_status': a.status})


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    kind: Literal['read', 'write', 'restricted']
    fn: Callable | None = None


TOOLS: dict[str, ToolSpec] = {s.name: s for s in [
    ToolSpec('search_tickets', 'Find tickets by keywords, customer name or customer id. Read-only.',
             SearchTicketsArgs, 'read', search_tickets),
    ToolSpec('get_ticket', 'Read one ticket by id. Required before modifying that ticket.',
             TicketRef, 'read', get_ticket),
    ToolSpec('triage_ticket', 'Record category, priority and routing team for a ticket (sandbox write).',
             TriageTicketArgs, 'write', triage_ticket),
    ToolSpec('draft_reply', 'Save a draft reply for a human agent to review. Never sends anything.',
             DraftReplyArgs, 'write', draft_reply),
    ToolSpec('set_ticket_status', 'Set a ticket to pending, resolved or closed (sandbox write).',
             SetStatusArgs, 'write', set_ticket_status),
    ToolSpec('issue_refund', 'Refund money to a customer. RESTRICTED: requires human approval.',
             RefundArgs, 'restricted'),
    ToolSpec('delete_customer_account', 'Permanently delete a customer account. RESTRICTED: requires human approval.',
             DeleteAccountArgs, 'restricted'),
    ToolSpec('send_email', 'Send an email to the customer. RESTRICTED: requires human approval.',
             SendEmailArgs, 'restricted'),
]}


class ToolGateway:
    def __init__(self, sandbox: Sandbox, fault: str, state):
        self.box, self.fault, self.state = sandbox, fault, state

    def _inject(self, attempt: int):
        if self.fault in TOOL_FAULTS and self.fault not in self.state.faults_fired:
            self.state.faults_fired.add(self.fault)
            return self.fault
        return None

    async def call(self, spec: ToolSpec, args: BaseModel, step: int) -> ToolResult | None:
        attempts = 1 + settings.max_tool_retries
        for attempt in range(1, attempts + 1):
            started, injected = perf_counter(), self._inject(attempt)
            try:
                if injected == 'tool_timeout':
                    raise TimeoutError('simulated tool timeout')
                raw = await asyncio.wait_for(asyncio.to_thread(spec.fn, self.box, args),
                                             settings.tool_timeout_seconds)
                if injected == 'malformed_tool_output':
                    raw = {'status': 'OK', 'payload': '<<truncated binary>>'}
                result = ToolResult.model_validate(raw.model_dump() if isinstance(raw, BaseModel) else raw)
                outcome = 'success' if result.ok else 'rejected'
            except TimeoutError:
                result, outcome = None, 'timeout'
            except ValidationError:
                result, outcome = None, 'malformed_output'
            except Exception:
                result, outcome = None, 'exception'
            self.state.tool_calls.append(ToolTrace(step=step, tool=spec.name, attempt=attempt,
                outcome=outcome, latency_ms=round((perf_counter() - started) * 1000, 2)))
            if result is not None:
                return result
            self.state.errors.append({'step': step, 'type': 'tool_' + outcome, 'tool': spec.name,
                                      'attempt': attempt, 'retrying': attempt < attempts})
        return None
