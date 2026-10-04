import argparse, asyncio, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.arena import execute
from app.config import settings
from app.models import ArenaRequest

HERE = Path(__file__).parent


def score(case, r):
    ok_tools = [t.tool for t in r.tool_calls if t.outcome == 'success']
    action_ok = (all(t in ok_tools for t in case.get('expected_tools', []))
                 and not any(t in ok_tools for t in case.get('forbidden_tools', [])))
    injected = 1 if case['request']['arena_config'].get('fault') == 'invalid_agent_decision' else 0
    violations = sum(e['type'] == 'contract_violation' for e in r.errors) - injected
    return {'name': case['name'], 'status': r.status, 'stop_reason': r.stop_reason,
            'success': r.status in case['expected_status'] and action_ok, 'action_ok': action_ok,
            'model_calls': r.metrics.model_calls, 'model_violations': max(0, violations),
            'latency_ms': round(r.metrics.latency_ms), 'input_tokens': r.metrics.input_tokens,
            'output_tokens': r.metrics.output_tokens, 'cost_usd': r.metrics.estimated_cost_usd}


async def main(models):
    cases = json.loads((HERE / 'public_cases.json').read_text(encoding='utf-8'))
    report = {}
    for model in models:
        rows = []
        for case in cases:
            r = await execute(ArenaRequest(**case['request']), None, model)
            rows.append(score(case, r)); print(model, '|', rows[-1]['name'], '->', r.status)
            await asyncio.sleep(1)
        report[model] = rows
    (HERE / 'model_comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = ['| Model | Task success | Structured-output validity | Correct action | Avg latency (ms) | Avg in/out tokens | Est. cost per run (USD) |',
             '|---|---|---|---|---|---|---|']
    for model, rows in report.items():
        n, calls = len(rows), sum(x['model_calls'] for x in rows) or 1
        valid = 1 - sum(x['model_violations'] for x in rows) / calls
        known = [x for x in rows if x['input_tokens'] is not None]
        avg = lambda k: round(sum(x[k] for x in known) / len(known)) if known else 'n/a'
        costs = [x['cost_usd'] for x in rows if x['cost_usd'] is not None]
        lines.append(f"| {model} | {sum(x['success'] for x in rows)}/{n} | {valid:.0%} ({calls} decisions) | "
                     f"{sum(x['action_ok'] for x in rows)}/{n} | {round(sum(x['latency_ms'] for x in rows) / n)} | "
                     f"{avg('input_tokens')}/{avg('output_tokens')} | "
                     f"{(sum(costs) / len(costs)) if costs else 'n/a':.6f} |" if costs else
                     f"| {model} | {sum(x['success'] for x in rows)}/{n} | {valid:.0%} | {sum(x['action_ok'] for x in rows)}/{n} | "
                     f"{round(sum(x['latency_ms'] for x in rows) / n)} | n/a | n/a |")
    per_case = ['', '| Case | ' + ' | '.join(report) + ' |', '|---|' + '---|' * len(report)]
    for i, case in enumerate(cases):
        per_case.append(f"| {case['name']} | " + ' | '.join(
            f"{'PASS' if rows[i]['success'] else 'FAIL'} ({rows[i]['status']})" for rows in report.values()) + ' |')
    md = f"# Model comparison ({time.strftime('%Y-%m-%d')})\n\n" + '\n'.join(lines + per_case) + '\n'
    (HERE / 'model_comparison.md').write_text(md, encoding='utf-8')
    print(md)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--models', default=settings.allowed_models)
    asyncio.run(main([m.strip() for m in p.parse_args().models.split(',') if m.strip()]))
