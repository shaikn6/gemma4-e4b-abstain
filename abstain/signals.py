"""Turn swegemma agent runs into per-patch features for selective acceptance.

Inputs, as the harness (swegemma 0.2.11) writes them under one --results-dir:
  task_results.jsonl         one row per task (see swegemma/results.py)
  patches/<id>.patch         agent patch (instance_id with '/' -> '__')
  traces/trace_<id>.json     SessionTrace JSON (adk_eval_core; ATIF-compatible)

Label-leak rule. Several task_results fields are computed in Phase 2 (the
hidden-test container) and must never be used as features:
  resolved        the label itself
  test_exit_code  exit code of the hidden tests
  error           set to None exactly when resolved (verification.py:697)
  duration_seconds  wall time of setup + agent + hidden-test run (evaluate.py:233)
The trace is recorded only during Phase 1 (agent_runner.py), so trace-derived
features, including agent wall time from trace timestamps, are safe.

The SessionTrace schema lives in adk_eval_core, which is only shipped in the
competition wheelhouse. The parser below therefore walks the JSON generically
and recognises both ATIF steps (tool_calls / observation.results) and raw ADK
events (function_call / function_response). Run `inspect` on the first real
trace and check that the counts match task_results before trusting features.

Usage:
  python -I abstain/signals.py build --runs RUN_DIR [RUN_DIR ...] --out features.jsonl
  python -I abstain/signals.py inspect TRACE.json
Each RUN_DIR is one sample (one seed) over the same tasks; sample index = order.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

LEAKY_RESULT_FIELDS = ('resolved', 'test_exit_code', 'error', 'duration_seconds')

# ---------------------------------------------------------------------------
# Patch parsing
# ---------------------------------------------------------------------------

_HUNK = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@')


def _strip_prefix(path: str) -> str:
    path = path.split('\t', 1)[0].strip()
    if path.startswith(('a/', 'b/')):
        return path[2:]
    return path


def parse_patch(text: str) -> dict[str, dict[str, Any]]:
    """Parse a unified diff (git or plain) into per-file edit records.

    Each record: removed (old line numbers), added (list of (anchor_old_line,
    content)), n_added, n_removed, new_file, deleted_file, binary.
    Added lines are anchored in old-file coordinates so that patches made
    against the same baseline can be compared by location.
    """
    files: dict[str, dict[str, Any]] = {}
    cur: dict[str, Any] | None = None
    old_path = None
    lines = (text or '').splitlines()
    i = 0

    def rec(path: str) -> dict[str, Any]:
        return files.setdefault(
            path,
            {'removed': set(), 'added': [], 'n_added': 0, 'n_removed': 0,
             'new_file': False, 'deleted_file': False, 'binary': False},
        )

    while i < len(lines):
        line = lines[i]
        if line.startswith('diff --git '):
            parts = line.split(' ')
            cur = rec(_strip_prefix(parts[-1])) if len(parts) >= 4 else None
            old_path = None
        elif line.startswith('--- '):
            old_path = line[4:]
        elif line.startswith('+++ ') and old_path is not None:
            new_path = line[4:]
            if new_path.split('\t')[0].strip() == '/dev/null':
                cur = rec(_strip_prefix(old_path))
                cur['deleted_file'] = True
            else:
                cur = rec(_strip_prefix(new_path))
                if old_path.split('\t')[0].strip() == '/dev/null':
                    cur['new_file'] = True
            old_path = None
        elif line.startswith(('GIT binary patch', 'Binary files ')) and cur is not None:
            cur['binary'] = True
        elif line.startswith('new file mode') and cur is not None:
            cur['new_file'] = True
        elif line.startswith('deleted file mode') and cur is not None:
            cur['deleted_file'] = True
        else:
            m = _HUNK.match(line)
            if m and cur is not None:
                old_ln = int(m.group(1))
                old_left = int(m.group(2)) if m.group(2) is not None else 1
                new_left = int(m.group(4)) if m.group(4) is not None else 1
                if old_ln == 0 and old_left == 0:
                    cur['new_file'] = cur['new_file'] or new_left > 0
                    old_ln = 1  # anchor insertions into an empty file at line 1
                # Consume exactly the hunk body so content lines that look
                # like headers ('--- x') are not misread.
                while (old_left > 0 or new_left > 0) and i + 1 < len(lines):
                    i += 1
                    body = lines[i]
                    if body.startswith('\\'):
                        continue
                    tag, content = (body[:1], body[1:]) if body else (' ', '')
                    if tag == '-':
                        cur['removed'].add(old_ln)
                        cur['n_removed'] += 1
                        old_ln += 1
                        old_left -= 1
                    elif tag == '+':
                        cur['added'].append((old_ln, content))
                        cur['n_added'] += 1
                        new_left -= 1
                    else:
                        old_ln += 1
                        old_left -= 1
                        new_left -= 1
        i += 1
    return files


def patch_shape(parsed: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Size and spread features of one parsed patch."""
    n_add = sum(f['n_added'] for f in parsed.values())
    n_rem = sum(f['n_removed'] for f in parsed.values())
    n_hunks = 0
    for f in parsed.values():
        # Count contiguous runs of edited old-line locations as hunks.
        locs = sorted(set(f['removed']) | {a for a, _ in f['added']})
        n_hunks += sum(1 for k, x in enumerate(locs) if k == 0 or x - locs[k - 1] > 1)
    return {
        'patch_empty': float(len(parsed) == 0),
        'files_touched': float(len(parsed)),
        'lines_added': float(n_add),
        'lines_removed': float(n_rem),
        'lines_changed': float(n_add + n_rem),
        'edit_regions': float(n_hunks),
        'new_files': float(sum(f['new_file'] for f in parsed.values())),
        'touches_non_py': float(any(not p.endswith('.py') for p in parsed)),
    }


def location_keys(parsed: dict[str, dict[str, Any]]) -> set[tuple]:
    """Edited locations in old-file coordinates (content ignored)."""
    keys: set[tuple] = set()
    for path, f in parsed.items():
        keys.update((path, '-', ln) for ln in f['removed'])
        keys.update((path, '+', ln) for ln, _ in f['added'])
    return keys


def content_keys(parsed: dict[str, dict[str, Any]]) -> set[tuple]:
    """Edited lines by file, sign and whitespace-normalised content."""
    keys: set[tuple] = set()
    for path, f in parsed.items():
        keys.update((path, '+', ' '.join(c.split())) for _, c in f['added'])
    return keys | {k for k in location_keys(parsed) if k[1] == '-'}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def agreement_features(patches: list[str]) -> list[dict[str, float]]:
    """For each of k sampled patches, its mean agreement with the other k-1.

    Empty patches agree with nothing (an empty patch is never shipped).
    """
    parsed = [parse_patch(p) for p in patches]
    loc = [location_keys(p) for p in parsed]
    con = [content_keys(p) for p in parsed]
    fil = [set(p) for p in parsed]
    norm = [' '.join((p or '').split()) for p in patches]
    k = len(patches)
    out = []
    for i in range(k):
        others = [j for j in range(k) if j != i]
        if not others:
            out.append({'agree_lines': float('nan'), 'agree_content': float('nan'),
                        'agree_files': float('nan'), 'agree_exact': float('nan'),
                        'k_samples': 1.0, 'k_nonempty': float(bool(fil[i]))})
            continue

        def mean(fn):
            vals = [0.0 if not fil[i] or not fil[j] else fn(i, j) for j in others]
            return sum(vals) / len(vals)

        out.append({
            'agree_lines': mean(lambda a, b: jaccard(loc[a], loc[b])),
            'agree_content': mean(lambda a, b: jaccard(con[a], con[b])),
            'agree_files': mean(lambda a, b: jaccard(fil[a], fil[b])),
            'agree_exact': mean(lambda a, b: float(norm[a] == norm[b])),
            'k_samples': float(k),
            'k_nonempty': float(sum(bool(f) for f in fil)),
        })
    return out


def medoid_index(patches: list[str]) -> int:
    """Index of the non-empty sample with the highest mean line agreement."""
    feats = agreement_features(patches)
    best, best_val = 0, -1.0
    for i, f in enumerate(feats):
        val = f['agree_lines'] if patches[i].strip() else -1.0
        val = -0.5 if val != val else val  # nan when k == 1
        if val > best_val:
            best, best_val = i, val
    return best


def mean_pairwise_line_agreement(patches: list[str]) -> float:
    loc = [location_keys(parse_patch(p)) for p in patches if p.strip()]
    pairs = list(combinations(range(len(loc)), 2))
    if not pairs:
        return float('nan')
    return sum(jaccard(loc[a], loc[b]) for a, b in pairs) / len(pairs)


# ---------------------------------------------------------------------------
# Trace parsing
# ---------------------------------------------------------------------------

EDIT_TOOLS = {'edit_file', 'write_file'}
READ_TOOLS = {'read_file'}
GRAPH_TOOLS = {'get_code_neighbors', 'search_similar_code', 'get_code_subgraph'}
_TEST_CMD = re.compile(
    r'(\bpy\.?test\b|-m\s+pytest\b|-m\s+unittest\b|\bunittest\b|\btox\b|\bnox\b|\bruntests?\b)'
)
_SHELL_EDIT = re.compile(
    r'(\bsed\s+-i|\bperl\s+-pi|\bgit\s+(apply|checkout|restore|stash)\b|\bpatch\s+-p|'
    r'(?<![<>&0-9])>\s*/?workspace/\S+|(?<![<>&0-9])>\s*[\w./-]+\.py\b|\btee\s+\S+\.py)'
)
_PYTEST_COUNT = re.compile(r'(\d+)\s+(passed|failed|errors?|skipped)\b')
_NUDGE_MARKERS = (
    'Please continue your work using the available tools',
    'reached the token limit',
)


def _as_obj(x: Any) -> Any:
    if isinstance(x, str):
        s = x.strip()
        if s[:1] in '{[':
            try:
                return json.loads(s)
            except ValueError:
                return x
    return x


def _iter_nodes(obj: Any):
    """Depth-first walk in document order."""
    stack = [obj]
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, dict):
            stack.extend(reversed(list(node.values())))
        elif isinstance(node, list):
            stack.extend(reversed(node))


def _call_from(node: dict) -> dict | None:
    if 'function_name' in node and 'arguments' in node:  # ATIF ToolCall
        return {'id': node.get('tool_call_id'), 'name': node['function_name'],
                'args': _as_obj(node.get('arguments')) or {}}
    for key in ('function_call', 'functionCall'):  # ADK / genai Part
        fc = node.get(key)
        if isinstance(fc, dict) and fc.get('name'):
            return {'id': fc.get('id'), 'name': fc['name'], 'args': _as_obj(fc.get('args')) or {}}
    return None


def _result_from(node: dict) -> dict | None:
    extra = node.get('extra') or {}
    if 'content' in node and ('source_call_id' in node or
                              isinstance(extra, dict) and extra.get('tool_name')):
        # ATIF ObservationResult, or v1.7's single observation without a call ID.
        return {'id': node.get('source_call_id'),
                'name': extra.get('tool_name') if isinstance(extra, dict) else None,
                'content': _as_obj(node['content'])}
    for key in ('function_response', 'functionResponse'):
        fr = node.get(key)
        if isinstance(fr, dict) and ('response' in fr or 'name' in fr):
            return {'id': fr.get('id'), 'name': fr.get('name'), 'content': _as_obj(fr.get('response'))}
    return None


def _parse_ts(value: Any) -> float | None:
    if isinstance(value, (int, float)) and value > 1e9:
        return float(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()
        except ValueError:
            return None
    return None


def extract_events(trace: Any) -> dict[str, Any]:
    """Pull tool calls (paired with results), agent steps, tokens and times."""
    calls: list[dict] = []
    by_id: dict[Any, dict] = {}
    seen_call_ids: set = set()
    pending: list[dict] = []
    timestamps: list[float] = []
    reasoning_chars = 0
    nudges = 0
    steps = trace.get('steps') if isinstance(trace, dict) else None
    step_ids = {id(s) for s in steps} if isinstance(steps, list) else set()
    for node in _iter_nodes(trace):
        if isinstance(node, str):
            nudges += any(m in node for m in _NUDGE_MARKERS)
            continue
        if not isinstance(node, dict):
            continue
        if id(node) in step_ids:
            pending = []  # ID-less ATIF results belong to calls in this step only.
        for key in ('timestamp', 'time', 'ts'):
            ts = _parse_ts(node.get(key))
            if ts is not None:
                timestamps.append(ts)
        rc = node.get('reasoning_content')
        if isinstance(rc, str):
            reasoning_chars += len(rc)
        if node.get('thought') is True and isinstance(node.get('text'), str):
            reasoning_chars += len(node['text'])
        call = _call_from(node)
        if call is not None:
            if call['id'] is not None and call['id'] in seen_call_ids:
                continue  # same call mirrored in another part of the trace
            seen_call_ids.add(call['id'])
            call['result'] = None
            calls.append(call)
            if call['id'] is not None:
                by_id[call['id']] = call
            pending.append(call)
            continue
        res = _result_from(node)
        if res is not None:
            target = by_id.get(res['id']) if res['id'] is not None else None
            if res['id'] is None:
                target = next((c for c in pending if c['result'] is None and
                               (res['name'] is None or c['name'] == res['name'])), None)
            if target is not None and target['result'] is None:
                target['result'] = res['content']
                if target in pending:
                    pending.remove(target)

    # Refused attempts never executed and must not affect tool/test features.
    refused = [c for c in calls if isinstance(c['result'], dict) and
               c['result'].get('error_type') == 'BudgetExceeded']
    calls = [c for c in calls if not (isinstance(c['result'], dict) and
             c['result'].get('error_type') == 'BudgetExceeded')]
    agent_steps = None
    prompt_toks = completion_toks = 0
    if isinstance(steps, list):
        agent_steps = sum(1 for s in steps if isinstance(s, dict) and s.get('source') == 'agent')
        for s in steps:
            m = s.get('metrics') if isinstance(s, dict) else None
            if isinstance(m, dict):
                prompt_toks += int(m.get('prompt_tokens') or 0)
                completion_toks += int(m.get('completion_tokens') or 0)
    fm = trace.get('final_metrics') if isinstance(trace, dict) else None
    if isinstance(fm, dict) and (fm.get('total_prompt_tokens') or fm.get('total_completion_tokens')):
        prompt_toks = int(fm.get('total_prompt_tokens') or 0)
        completion_toks = int(fm.get('total_completion_tokens') or 0)
    if not prompt_toks and not completion_toks:
        for node in _iter_nodes(trace):
            if isinstance(node, dict) and 'prompt_token_count' in node:
                prompt_toks += int(node.get('prompt_token_count') or 0)
                completion_toks += int(node.get('candidates_token_count') or 0)
    return {
        'calls': calls,
        'budget_refused': len(refused),
        'agent_steps': agent_steps,
        'prompt_tokens': prompt_toks,
        'completion_tokens': completion_toks,
        'reasoning_chars': reasoning_chars,
        'nudges': nudges,
        'agent_seconds': (max(timestamps) - min(timestamps)) if len(timestamps) >= 2 else None,
    }


def _command_outcome(result: Any) -> tuple[int | None, str]:
    """(exit_code, output text) from a run_command tool response."""
    if isinstance(result, dict) and 'result' in result and 'status' not in result:
        result = _as_obj(result['result'])  # ADK may wrap non-dict returns
    if not isinstance(result, dict):
        return None, str(result or '')
    if result.get('status') == 'ok':
        return int(result.get('exit_code') or 0), str(result.get('stdout', '')) + str(result.get('stderr', ''))
    details = result.get('details') or {}
    if result.get('error_type') == 'TimeoutExceeded':
        return 124, str(result.get('error_message', ''))
    code = details.get('exit_code') if isinstance(details, dict) else None
    text = ''
    if isinstance(details, dict):
        text = str(details.get('stdout', '')) + str(details.get('stderr', ''))
    return (int(code) if code is not None else 1), text or str(result.get('error_message', ''))


def _pytest_counts(text: str) -> dict[str, int]:
    counts = {'passed': 0, 'failed': 0, 'errors': 0}
    for n, kind in _PYTEST_COUNT.findall(text):
        key = 'errors' if kind.startswith('error') else kind
        if key in counts:
            counts[key] = int(n)  # last summary line wins
    return counts


def trace_features(trace: Any) -> dict[str, float]:
    ev = extract_events(trace)
    calls = ev['calls']
    edit_idx: list[int] = []
    tests: list[dict] = []
    n_err = 0
    for i, c in enumerate(calls):
        name, args = c['name'], c['args'] if isinstance(c['args'], dict) else {}
        res = c['result'] if isinstance(c['result'], dict) else _as_obj(c['result'])
        if isinstance(res, dict) and res.get('status') == 'error':
            n_err += 1
        if name in EDIT_TOOLS:
            edit_idx.append(i)
        elif name == 'run_command':
            cmd = str(args.get('command', ''))
            if _TEST_CMD.search(cmd):
                code, text = _command_outcome(res)
                tests.append({'i': i, 'exit': code, **_pytest_counts(text)})
            elif _SHELL_EDIT.search(cmd):
                edit_idx.append(i)
    last_edit = edit_idx[-1] if edit_idx else -1
    after = [t for t in tests if t['i'] > last_edit]
    last = tests[-1] if tests else None
    nan = float('nan')
    return {
        'trace_tool_calls': float(sum(1 for c in calls if c['name'] not in ('submit_patch', 'get_status'))),
        'trace_agent_steps': float(ev['agent_steps']) if ev['agent_steps'] is not None else nan,
        'n_edits': float(len(edit_idx)),
        'n_reads': float(sum(c['name'] in READ_TOOLS for c in calls)),
        'n_graph_calls': float(sum(c['name'] in GRAPH_TOOLS for c in calls)),
        'n_tool_errors': float(n_err),
        'budget_refused': float(ev['budget_refused']),
        'n_submits': float(sum(c['name'] == 'submit_patch' for c in calls)),
        'n_nudges': float(ev['nudges']),
        'ran_tests': float(bool(tests)),
        'n_test_runs': float(len(tests)),
        'n_test_runs_failed': float(sum(1 for t in tests if t['exit'] not in (0, None))),
        'ran_tests_after_last_edit': float(bool(after)),
        'last_test_passed': (nan if last is None or last['exit'] is None else float(last['exit'] == 0)),
        'last_test_n_failed': float(last['failed'] + last['errors']) if last else nan,
        'last_test_n_passed': float(last['passed']) if last else nan,
        'prompt_tokens': float(ev['prompt_tokens']),
        'completion_tokens': float(ev['completion_tokens']),
        'reasoning_chars': float(ev['reasoning_chars']),
        'agent_seconds': float(ev['agent_seconds']) if ev['agent_seconds'] is not None else nan,
    }


# ---------------------------------------------------------------------------
# One task, one sample
# ---------------------------------------------------------------------------

def result_features(row: dict) -> dict[str, float]:
    """Phase-1 counters from a task_results row (leaky fields excluded)."""
    return {
        'agent_patch_chars': float(row.get('agent_patch_size') or 0),
        'tool_calls': float(row.get('tool_calls') or 0),
        'llm_calls': float(row.get('total_llm_calls') or 0),
    }


def run_features(row: dict, patch_text: str, trace: Any | None) -> dict[str, float]:
    """All single-sample features for one task run."""
    feats = result_features(row)
    feats.update(patch_shape(parse_patch(patch_text)))
    if trace is not None:
        feats.update(trace_features(trace))
    return feats


def safe_id(instance_id: str) -> str:
    return instance_id.replace('/', '__')


def load_run(run_dir: Path) -> dict[str, dict]:
    """instance_id -> {'row', 'patch', 'trace'} for one results dir."""
    out: dict[str, dict] = {}
    with open(run_dir / 'task_results.jsonl', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            iid = row['instance_id']
            patch_file = run_dir / 'patches' / f'{safe_id(iid)}.patch'
            trace = None
            for name in (f'trace_{safe_id(iid)}.json', f'trace_{iid.replace("/", "_")}.json'):
                tf = run_dir / 'traces' / name
                if tf.exists():
                    trace = json.loads(tf.read_text(encoding='utf-8'))
                    break
            out[iid] = {
                'row': row,
                'patch': patch_file.read_text(encoding='utf-8') if patch_file.exists() else '',
                'trace': trace,
            }
    return out


def build(run_dirs: list[Path]) -> list[dict]:
    """One output row per (task, sample) with features, label and repo."""
    runs = [load_run(d) for d in run_dirs]
    ids = sorted(set().union(*[r.keys() for r in runs]))
    rows = []
    for iid in ids:
        present = [(s, r[iid]) for s, r in enumerate(runs) if iid in r]
        agree = agreement_features([x['patch'] for _, x in present])
        for (s, x), ag in zip(present, agree):
            feats = run_features(x['row'], x['patch'], x['trace'])
            feats.update(ag)
            rows.append({
                'instance_id': iid,
                'repo': x['row'].get('repo', ''),
                'sample': s,
                'resolved': bool(x['row'].get('resolved')),
                'has_trace': x['trace'] is not None,
                'features': feats,
            })
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('--runs', nargs='+', type=Path, required=True)
    b.add_argument('--out', type=Path, required=True)
    i = sub.add_parser('inspect')
    i.add_argument('trace', type=Path)
    args = ap.parse_args(argv)
    if args.cmd == 'inspect':
        trace = json.loads(args.trace.read_text(encoding='utf-8'))
        ev = extract_events(trace)
        top = list(trace.keys()) if isinstance(trace, dict) else type(trace).__name__
        print('top-level keys:', top)
        print('tool calls found:', len(ev['calls']),
              '| with results:', sum(c['result'] is not None for c in ev['calls']))
        print('names:', sorted({c['name'] for c in ev['calls']}))
        print(json.dumps(trace_features(trace), indent=1))
        return 0
    rows = build(args.runs)
    with open(args.out, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print(f'wrote {len(rows)} rows for {len({r["instance_id"] for r in rows})} tasks to {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
