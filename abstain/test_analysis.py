"""Synthetic-data tests for task_analysis.py and taxonomy.py.

Deterministic, CPU-only, offline. No model, dataset or network needed.
Runs under pytest or as a plain script:
  python -I abstain/test_analysis.py
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))          # abstain/: task_analysis, signals
sys.path.insert(0, str(_HERE.parent))   # repo root: taxonomy

import task_analysis as ta  # noqa: E402
import taxonomy as tx  # noqa: E402

GIT_PATCH = """diff --git a/pkg/mod.py b/pkg/mod.py
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -10,4 +10,5 @@ def f():
     a = 1
-    b = 2
+    b = 3
+    c = 4
     return a
"""

TEST_PATCH = """diff --git a/tests/test_mod.py b/tests/test_mod.py
--- a/tests/test_mod.py
+++ b/tests/test_mod.py
@@ -0,0 +1,2 @@
+def test_one():
+    assert f() == 1
"""


def _row(created, issue='issue body', hints='', patch=GIT_PATCH, test_patch=TEST_PATCH):
    return {'repo': 'a/a', 'patch': patch, 'test_patch': test_patch,
            'problem_statement': issue, 'hints_text': hints, 'created_at': created}


# ---------------------------------------------------------------------------
# task_analysis
# ---------------------------------------------------------------------------

def test_quantile():
    assert ta.q([1, 2, 3, 4], 0.5) == 2.5
    assert ta.q([10, 20, 30], 0.0) == 10 and ta.q([10, 20, 30], 1.0) == 30
    assert ta.q([5], 0.5) == 5
    assert math.isnan(ta.q([], 0.5))


def test_fmt():
    assert ta.fmt(1.0) == '1'          # integral, |v|>=1 -> int
    assert ta.fmt(0.0) == '0.00'       # |v|<1 -> 2dp
    assert ta.fmt(2.5) == '2.50'
    assert ta.fmt('ALL') == 'ALL'


def test_describe():
    rows = [_row('2024-01-02T00:00:00Z', hints='look here'),
            _row('2024-03-04T00:00:00Z', issue='```code```')]
    d = ta.describe(rows)
    assert d['n'] == 2
    assert d['gold_files_median'] == 1 and d['frac_single_file'] == 1.0
    assert d['gold_lines_median'] == 3          # 1 removed + 2 added
    assert d['test_files_median'] == 1 and d['test_defs_median'] == 1
    assert d['frac_zero_test_defs'] == 0.0
    assert d['frac_hints'] == 0.5               # one row has hints_text
    assert d['frac_issue_has_code_block'] == 0.5
    assert d['created_min'] == '2024-01-02' and d['created_max'] == '2024-03-04'


# ---------------------------------------------------------------------------
# taxonomy
# ---------------------------------------------------------------------------

def _ps(exists=True, size=50, n_files=1, n_hunks=1, tests=False, nontest=True):
    return dict(exists=exists, size=size, files=[], n_files=n_files, n_hunks=n_hunks,
                touches_tests=tests, touches_nontest=nontest)


def _sig(applied=1, edit_err=0, ran_tests=False):
    return dict(n_applied_edits=applied, n_edit_err=edit_err, ran_tests=ran_tests)


def _r(**kw):
    """A non-degenerate run row (non-empty patch) unless a branch overrides it."""
    return {'error': '', 'agent_patch_size': 10, **kw}


def test_classify_branches():
    assert tx.classify(_r(error='ContextWindowExceeded'), _sig(), _ps()) == 'budget_context'
    assert tx.classify(_r(error='hit the tool call budget'), _sig(), _ps()) == 'budget_context'
    assert tx.classify(_r(agent_patch_size=0), _sig(applied=0), _ps(size=0)) == 'empty_patch'
    assert tx.classify(_r(), _sig(), _ps(n_files=3)) == 'scattershot'
    assert tx.classify(_r(), _sig(), _ps(n_hunks=8)) == 'scattershot'
    assert tx.classify(_r(), _sig(applied=1, edit_err=3), _ps()) == 'broken_edits'
    assert tx.classify(_r(), _sig(), _ps(tests=True, nontest=False)) == 'wrong_file'
    assert tx.classify(_r(), _sig(ran_tests=True), _ps()) == 'tested_but_unresolved'
    assert tx.classify(_r(), _sig(ran_tests=False), _ps()) == 'never_tested'


def test_parse_obs():
    assert tx.parse_obs({'observation': {'content': json.dumps({'status': 'ok'})}}) == {'status': 'ok'}
    assert tx.parse_obs({'observation': {'content': 'not json'}}) == {}
    assert tx.parse_obs({'observation': {'content': {'status': 'error'}}}) == {'status': 'error'}
    assert tx.parse_obs({}) == {}


def _step(name, status='ok', command='', exit_code=None):
    obs = {'status': status}
    if exit_code is not None:
        obs['exit_code'] = exit_code
    return {'tool_calls': [{'function_name': name, 'arguments': {'command': command}}],
            'observation': {'content': json.dumps(obs)}}


def test_trace_signals():
    trace = {'steps': [
        _step('read_file'),
        _step('run_command', command='python -m pytest -q', exit_code=0),
        _step('edit_file'),
        _step('run_command', status='error', command='pytest -q', exit_code=1),
        _step('submit_patch'),
    ], 'final_metrics': {'total_steps': 5}}
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / 'trace.json'
        p.write_text(json.dumps(trace), encoding='utf-8')
        sig = tx.trace_signals(str(p))
    assert sig['n_tool_calls'] == 5 and sig['n_reads'] == 1
    assert sig['n_edit_ok'] == 1 and sig['n_applied_edits'] == 1
    assert sig['n_cmd'] == 2 and sig['n_cmd_err'] == 1 and sig['n_tool_err'] == 1
    assert sig['n_test_runs'] == 2 and sig['test_exit_codes'] == [0, 1]
    assert sig['n_submits'] == 1 and sig['total_steps'] == 5
    assert sig['ran_tests'] and sig['ran_tests_after_last_edit']  # last test (idx 3) > last edit (idx 2)


def test_patch_stats():
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        (run / 'patches').mkdir()
        (run / 'patches' / 'ok.patch').write_text(GIT_PATCH + TEST_PATCH, encoding='utf-8')
        hit = tx.patch_stats(str(run), 'ok')
        miss = tx.patch_stats(str(run), 'absent')
    assert hit['exists'] and hit['size'] > 0
    assert hit['n_files'] == 2 and hit['n_hunks'] == 2
    assert hit['touches_tests'] and hit['touches_nontest']
    assert miss == dict(exists=False, size=0, files=[], n_files=0, n_hunks=0,
                        touches_tests=False, touches_nontest=False)


if __name__ == '__main__':
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
            print(f'PASS {name}')
        except unittest.SkipTest as e:
            skipped += 1
            print(f'SKIP {name}: {e}')
        except Exception as e:
            failed += 1
            print(f'FAIL {name}: {type(e).__name__}: {e}')
    print(f'{len(tests) - failed - skipped}/{len(tests)} passed, {skipped} skipped')
    sys.exit(1 if failed else 0)
