import json
import time
from collections import defaultdict, deque
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import FileResponse
from app.config import ROOT, settings
from app.models import ArenaRequest, ArenaResponse, ChatRequest
from app.arena import execute
router = APIRouter()
_chat_hits: dict[str, deque] = defaultdict(deque)
def _rate_limit(ip: str):
    now, hits = time.monotonic(), _chat_hits[ip]
    while hits and now - hits[0] > 60: hits.popleft()
    if len(hits) >= settings.chat_requests_per_minute:
        raise HTTPException(429, 'Too many chat requests; wait a minute')
    hits.append(now)
@router.get('/')
def index(): return FileResponse(ROOT / 'app/static/index.html')
@router.get('/health')
def health(): return {'status': 'ok', 'implementation': 'ticket-triage-sentinel',
                      'model_configured': bool(settings.api_key)}
@router.get('/arena/manifest')
def manifest(): return json.loads((ROOT / 'arena_manifest.json').read_text())
@router.get('/models')
def models():
    return {'models': settings.model_list, 'default': settings.model_name,
            'provider': settings.model_provider, 'configured': bool(settings.api_key)}
@router.post('/arena/run', response_model=ArenaResponse)
async def arena_run(payload: ArenaRequest):
    return await execute(payload, model=settings.model_name)
@router.post('/chat', response_model=ArenaResponse)
async def chat(payload: ChatRequest, request: Request):
    model = payload.model or settings.model_name
    if model not in settings.model_list:
        raise HTTPException(400, 'Model is not enabled')
    _rate_limit(request.client.host if request.client else 'unknown')
    busy = request.app.state.busy
    if payload.session_id in busy: raise HTTPException(409, 'This chat is already running')
    busy.add(payload.session_id)
    memory = request.app.state.memory
    try:
        result = await execute(payload, memory.get(payload.session_id), model)
        memory.add(payload.session_id, payload.task, result.final_response)
        return result
    finally:
        busy.discard(payload.session_id)
@router.delete('/chat/{session_id}')
def reset(session_id: str, request: Request):
    if session_id in request.app.state.busy: raise HTTPException(409, 'Chat is running')
    request.app.state.memory.clear(session_id)
    return {'status': 'cleared'}
