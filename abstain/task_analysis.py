"""Descriptive statistics of the 129 public tasks (tasks.jsonl only, no model runs).

Gold patches and test patches are read here only to describe the task set.
They are never given to the agent.

Usage (from ~/gemma-abstain):
  python -I abstain/task_analysis.py data/tasks.jsonl > task_tables.txt
"""

from __future__ import annotations

import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from signals import parse_patch, patch_shape  # noqa: E402

_TEST_DEF = re.compile(r'^\+\s*(async\s+)?def\s+test_\w*', re.M)


def q(vals, p):
    vals = sorted(vals)
    if not vals:
        return float('nan')
    k = (len(vals) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def describe(rows):
    g = [patch_shape(parse_patch(r['patch'])) for r in rows]
    t = [parse_patch(r['test_patch']) for r in rows]
    files = [x['files_touched'] for x in g]
    lines = [x['lines_changed'] for x in g]
    issue = [len(r['problem_statement']) for r in rows]
    tdefs = [len(_TEST_DEF.findall(r['test_patch'])) for r in rows]
    tfiles = [len(x) for x in t]
    return {
        'n': len(rows),
        'gold_files_median': st.median(files), 'gold_files_max': max(files),
        'frac_single_file': sum(f == 1 for f in files) / len(rows),
        'gold_lines_median': st.median(lines), 'gold_lines_p25': q(lines, .25),
        'gold_lines_p75': q(lines, .75), 'gold_lines_max': max(lines),
        'frac_gold_adds_new_file': sum(x['new_files'] > 0 for x in g) / len(rows),
        'frac_gold_touches_non_py': sum(x['touches_non_py'] > 0 for x in g) / len(rows),
        'test_files_median': st.median(tfiles),
        'test_defs_median': st.median(tdefs), 'test_defs_p25': q(tdefs, .25),
        'test_defs_p75': q(tdefs, .75), 'test_defs_max': max(tdefs),
        'frac_zero_test_defs': sum(d == 0 for d in tdefs) / len(rows),
        'issue_chars_median': st.median(issue), 'issue_chars_p25': q(issue, .25),
        'issue_chars_p75': q(issue, .75), 'issue_chars_max': max(issue),
        'frac_hints': sum(bool(r.get('hints_text', '').strip()) for r in rows) / len(rows),
        'frac_pr_template': sum('## Pull Request' in r['problem_statement'] for r in rows) / len(rows),
        'frac_issue_has_code_block': sum('```' in r['problem_statement'] for r in rows) / len(rows),
        'created_min': min(r['created_at'] for r in rows)[:10],
        'created_max': max(r['created_at'] for r in rows)[:10],
    }


def fmt(v):
    if isinstance(v, float):
        return f'{v:.2f}' if v != int(v) or abs(v) < 1 else f'{int(v)}'
    return str(v)


def main(path):
    rows = [json.loads(line) for line in open(path, encoding='utf-8') if line.strip()]
    by_repo = defaultdict(list)
    for r in rows:
        by_repo[r['repo']].append(r)
    groups = sorted(by_repo.items(), key=lambda kv: -len(kv[1])) + [('ALL', rows)]
    stats = [(name, describe(rs)) for name, rs in groups]
    keys = list(stats[0][1])
    width = max(len(k) for k in keys) + 2
    cols = [name.split('/')[-1] for name, _ in stats]
    print('metric'.ljust(width) + ''.join(c.rjust(12) for c in cols))
    for k in keys:
        print(k.ljust(width) + ''.join(fmt(s[k]).rjust(12) for _, s in stats))
    # Size distribution of gold patches (all repos), for the "hard tail".
    lines = sorted(patch_shape(parse_patch(r['patch']))['lines_changed'] for r in rows)
    edges = [0, 5, 10, 20, 50, 100, 200, 500, 10**9]
    print('\ngold lines changed histogram (all tasks)')
    for lo, hi in zip(edges, edges[1:]):
        n = sum(lo < x <= hi for x in lines)
        label = f'({lo}, {hi}]' if hi < 10**9 else f'> {lo}'
        print(f'  {label:>12s}  {n:4d}  {"#" * n}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'data/tasks.jsonl')
