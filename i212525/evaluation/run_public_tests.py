import argparse, json, time
from pathlib import Path
import httpx
parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://127.0.0.1:8000')
parser.add_argument('--out', default=str(Path(__file__).with_name('results_public.json')))
args = parser.parse_args()
cases = json.loads(Path(__file__).with_name('public_cases.json').read_text(encoding='utf-8'))
rows, passed = [], 0
for case in cases:
    row = {'name': case['name'], 'category': case['category'], 'expected': case['expected_status']}
    try:
        started = time.perf_counter()
        response = httpx.post(args.url.rstrip('/') + '/arena/run', json=case['request'], timeout=70)
        response.raise_for_status(); result = response.json()
        tools_ok = [t['tool'] for t in result['tool_calls'] if t['outcome'] == 'success']
        ok = (result['status'] in case['expected_status']
              and all(t in tools_ok for t in case.get('expected_tools', []))
              and not any(t in tools_ok for t in case.get('forbidden_tools', [])))
        row.update(status=result['status'], stop_reason=result['stop_reason'], steps=result['steps'],
                   tools=[f"{t['tool']}#{t['attempt']}:{t['outcome']}" for t in result['tool_calls']],
                   final_response=result['final_response'][:300], metrics=result['metrics'],
                   wall_ms=round((time.perf_counter() - started) * 1000))
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        ok = False; row['error'] = type(exc).__name__
    row['pass'] = ok; passed += ok; rows.append(row)
    print(('PASS' if ok else 'FAIL'), case['name'], '->', row.get('status'), row.get('stop_reason', row.get('error')))
summary = {'url': args.url, 'run_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'passed': passed, 'total': len(cases), 'cases': rows}
Path(args.out).write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(f'{passed}/{len(cases)} public checks passed. Results: {args.out}. Not the hidden Arena score.')
raise SystemExit(0 if passed == len(cases) else 1)
