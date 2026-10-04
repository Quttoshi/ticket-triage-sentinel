import asyncio
import datetime as dt
import json
from dataclasses import dataclass
from app.config import settings

PRICES = {
    'gpt-4.1-nano': (0.10, 0.40),
    'gpt-4.1-mini': (0.40, 1.60),
}


class ModelUnavailable(Exception):
    pass


class SpendCapReached(Exception):
    pass


@dataclass
class ModelReply:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class _DailyCounter:
    def __init__(self): self.day, self.count = None, 0
    def take(self):
        today = dt.date.today()
        if today != self.day: self.day, self.count = today, 0
        if self.count >= settings.max_model_calls_per_day: raise SpendCapReached()
        self.count += 1


DAILY = _DailyCounter()


def estimate_cost(model_name: str, input_tokens, output_tokens):
    price = PRICES.get(model_name)
    if price is None or input_tokens is None or output_tokens is None: return None
    return round((input_tokens * price[0] + output_tokens * price[1]) / 1_000_000, 8)


def _text(content) -> str:
    if isinstance(content, str): return content
    if isinstance(content, list):
        return ''.join(p.get('text', '') if isinstance(p, dict) else str(p) for p in content)
    return str(content)


class OpenAIModel:
    def __init__(self, name: str):
        if not settings.api_key:
            raise ModelUnavailable('No OPENAI_API_KEY configured')
        from langchain_openai import ChatOpenAI
        self.name = name
        self.llm = ChatOpenAI(
            model=name, api_key=settings.api_key, temperature=0,
            max_tokens=settings.max_output_tokens, timeout=settings.model_timeout_seconds,
            max_retries=0).bind(response_format={'type': 'json_object'})

    async def __call__(self, messages, timeout: float) -> ModelReply:
        DAILY.take()
        try:
            msg = await asyncio.wait_for(self.llm.ainvoke(messages), timeout)
        except TimeoutError:
            raise
        except Exception as exc:
            raise ModelUnavailable(type(exc).__name__) from exc
        usage = getattr(msg, 'usage_metadata', None) or {}
        return ModelReply(_text(msg.content), usage.get('input_tokens'), usage.get('output_tokens'))


class ScriptedModel:
    name = 'scripted'
    def __init__(self, outputs):
        self.outputs, self.seen = list(outputs), []
    async def __call__(self, messages, timeout: float) -> ModelReply:
        self.seen.append(messages)
        if not self.outputs: raise ModelUnavailable('script exhausted')
        out = self.outputs.pop(0)
        if isinstance(out, Exception): raise out
        return ModelReply(out if isinstance(out, str) else json.dumps(out))


def get_model(name: str):
    if name not in settings.model_list:
        raise ModelUnavailable(f'Model {name!r} is not enabled')
    return OpenAIModel(name)
