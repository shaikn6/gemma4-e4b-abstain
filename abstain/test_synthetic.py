"""Synthetic-data tests for signals.py and risk_coverage.py.

Runs under pytest or as a plain script:
  python -I abstain/test_synthetic.py
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I drops the script dir

import numpy as np  # noqa: E402

import risk_coverage as rc  # noqa: E402
import signals as sg  # noqa: E402

GIT_PATCH = """diff --git a/pkg/mod.py b/pkg/mod.py
index 111..222 100644
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -10,4 +10,5 @@ def f():
     a = 1
-    b = 2
+    b = 3
+    c = 4
     return a
 # end
"""

PLAIN_NEW_FILE = """--- /dev/null
+++ b/pkg/new.py
@@ -0,0 +1,3 @@
+--- not a header
+x = 1
+y = 2
"""

OTHER_PATCH = """diff --git a/pkg/other.py b/pkg/other.py
--- a/pkg/other.py
+++ b/pkg/other.py
@@ -1,2 +1,2 @@
-old
+new
 keep
"""


def test_parse_git_patch():
    p = sg.parse_patch(GIT_PATCH)
    assert list(p) == ['pkg/mod.py']
    f = p['pkg/mod.py']
    assert f['removed'] == {11}
    assert f['n_added'] == 2 and f['n_removed'] == 1
    assert [a for a, _ in f['added']] == [12, 12]
    shape = sg.patch_shape(p)
    assert shape['files_touched'] == 1 and shape['lines_changed'] == 3
    assert shape['edit_regions'] == 1 and shape['new_files'] == 0


def test_parse_new_file_with_header_like_content():
    p = sg.parse_patch(PLAIN_NEW_FILE)
    assert list(p) == ['pkg/new.py'], p.keys()
    assert p['pkg/new.py']['new_file'] and p['pkg/new.py']['n_added'] == 3


def test_empty_patch():
    s = sg.patch_shape(sg.parse_patch(''))
    assert s['patch_empty'] == 1 and s['files_touched'] == 0


def test_agreement():
    same = sg.agreement_features([GIT_PATCH, GIT_PATCH, GIT_PATCH])
    assert all(a['agree_lines'] == 1 and a['agree_exact'] == 1 for a in same)
    mixed = sg.agreement_features([GIT_PATCH, GIT_PATCH, OTHER_PATCH])
    assert mixed[0]['agree_lines'] == 0.5 and mixed[2]['agree_lines'] == 0.0
    assert mixed[2]['agree_files'] == 0.0
    with_empty = sg.agreement_features([GIT_PATCH, ''])
    assert with_empty[0]['agree_lines'] == 0.0 and with_empty[1]['agree_lines'] == 0.0
    assert with_empty[0]['k_nonempty'] == 1
    assert math.isnan(sg.agreement_features([GIT_PATCH])[0]['agree_lines'])
    assert sg.medoid_index(['', OTHER_PATCH, GIT_PATCH, GIT_PATCH]) == 2
    assert sg.mean_pairwise_line_agreement([GIT_PATCH, GIT_PATCH, OTHER_PATCH]) == 1 / 3


def _ok(stdout, code=0):
    return json.dumps({'status': 'ok', 'stdout': stdout, 'stderr': '', 'exit_code': code})


def _err(stdout, code):
    return json.dumps({'status': 'error', 'error_type': 'CommandError', 'error_message': stdout,
                       'details': {'stdout': stdout, 'stderr': '', 'exit_code': code}})


def atif_trace():
    """ATIF-shaped trace: read, failing test, edit, passing test, submit."""
    seq = [
        ('read_file', {'filepath': 'pkg/mod.py'}, json.dumps({'status': 'ok', 'content': '...'})),
        ('run_command', {'command': 'python -m pytest tests/test_mod.py -q'},
         _err('1 failed, 4 passed in 0.2s', 1)),
        ('edit_file', {'filepath': 'pkg/mod.py', 'old_string': 'b = 2', 'new_string': 'b = 3'},
         json.dumps({'status': 'ok'})),
        ('run_command', {'command': 'cd /workspace && pytest tests/test_mod.py -q'},
         _ok('5 passed in 0.2s')),
        ('submit_patch', {}, json.dumps({'status': 'ok', 'patch_size': 10})),
    ]
    steps = [{'step_id': 1, 'source': 'user', 'message': 'task prompt',
              'timestamp': '2026-11-01T00:00:00Z'}]
    for i, (name, args, res) in enumerate(seq):
        steps.append({
            'step_id': i + 2, 'source': 'agent', 'message': '',
            'timestamp': f'2026-11-01T00:0{i + 1}:00Z',
            'reasoning_content': 'think',
            'tool_calls': [{'tool_call_id': f'c{i}', 'function_name': name, 'arguments': args}],
            'observation': {'results': [{'source_call_id': f'c{i}', 'content': res}]},
            'metrics': {'prompt_tokens': 100, 'completion_tokens': 10},
        })
    return {'schema_version': 'ATIF-v1.6', 'agent': {'name': 'x'}, 'steps': steps}


def adk_trace():
    """Raw ADK event list: edit after the only (passing) test run, plus a nudge."""
    def call(i, name, args):
        return {'author': 'coder', 'content': {'role': 'model', 'parts': [
            {'function_call': {'id': f'a{i}', 'name': name, 'args': args}}]}}

    def resp(i, name, body):
        return {'author': 'coder', 'content': {'role': 'user', 'parts': [
            {'function_response': {'id': f'a{i}', 'name': name, 'response': body}}]}}

    return {'events': [
        call(0, 'run_command', {'command': 'pytest -q tests'}),
        resp(0, 'run_command', {'status': 'ok', 'stdout': '3 passed', 'stderr': '', 'exit_code': 0}),
        call(1, 'run_command', {'command': "sed -i 's/a/b/' pkg/mod.py"}),
        resp(1, 'run_command', {'status': 'ok', 'stdout': '', 'stderr': '', 'exit_code': 0}),
        {'type': 'continuation_nudge', 'text': 'Please continue your work using the available tools, or call submit_patch'},
        call(2, 'search_similar_code', {'query': 'f'}),
        resp(2, 'search_similar_code', {'status': 'error', 'error_type': 'X', 'error_message': 'y'}),
    ]}


def test_trace_features_atif():
    f = sg.trace_features(atif_trace())
    assert f['trace_tool_calls'] == 4, f
    assert f['trace_agent_steps'] == 5
    assert f['ran_tests'] == 1 and f['n_test_runs'] == 2 and f['n_test_runs_failed'] == 1
    assert f['ran_tests_after_last_edit'] == 1 and f['last_test_passed'] == 1
    assert f['last_test_n_passed'] == 5 and f['last_test_n_failed'] == 0
    assert f['n_edits'] == 1 and f['n_reads'] == 1 and f['n_submits'] == 1
    assert f['prompt_tokens'] == 500 and f['completion_tokens'] == 50
    assert f['agent_seconds'] == 300.0 and f['reasoning_chars'] == 25


def _check_real_trace(instance_id, counted, refused):
    path = (Path(__file__).resolve().parent.parent / 'ref/starter_traces/results/traces'
            / f'trace_{instance_id}.json')
    trace = json.loads(path.read_text(encoding='utf-8'))
    ev = sg.extract_events(trace)
    f = sg.trace_features(trace)
    assert len(ev['calls']) == counted
    assert all(c['result'] is not None for c in ev['calls'])
    assert f['trace_tool_calls'] == counted
    assert f['budget_refused'] == refused
    assert f['prompt_tokens'] == trace['final_metrics']['total_prompt_tokens']
    assert f['completion_tokens'] == trace['final_metrics']['total_completion_tokens']
    assert not set(sg.LEAKY_RESULT_FIELDS) & f.keys()


def test_real_trace_fastapi_15661():
    _check_real_trace('fastapi_15661', 10, 1)


def test_real_trace_fastapi_15588():
    _check_real_trace('fastapi_15588', 5, 0)


def test_trace_features_adk():
    f = sg.trace_features(adk_trace())
    assert f['ran_tests'] == 1 and f['ran_tests_after_last_edit'] == 0, f
    assert f['n_edits'] == 1  # shell sed -i counts as an edit
    assert f['n_graph_calls'] == 1 and f['n_tool_errors'] == 1
    assert f['n_nudges'] == 1 and f['n_submits'] == 0
    assert math.isnan(f['trace_agent_steps']) and math.isnan(f['agent_seconds'])


def test_no_leaky_fields():
    row = {'instance_id': 'x', 'repo': 'r', 'resolved': True, 'agent_patch_size': 10,
           'test_exit_code': 0, 'duration_seconds': 5.0, 'error': None,
           'tool_calls': 7, 'total_llm_calls': 9}
    feats = sg.run_features(row, GIT_PATCH, atif_trace())
    for leaky in sg.LEAKY_RESULT_FIELDS:
        assert leaky not in feats
    assert 'duration_seconds' not in feats and 'test_exit_code' not in feats
    assert feats['tool_calls'] == 7 and feats['llm_calls'] == 9


def _write_run(d: Path, rows, patches, traces):
    (d / 'patches').mkdir(parents=True)
    (d / 'traces').mkdir()
    with open(d / 'task_results.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    for iid, p in patches.items():
        if p:
            (d / 'patches' / f'{iid}.patch').write_text(p)
    for iid, t in traces.items():
        (d / 'traces' / f'trace_{iid}.json').write_text(json.dumps(t))


def test_build_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        dirs = []
        for s, p2 in enumerate([OTHER_PATCH, OTHER_PATCH, '']):
            d = Path(tmp) / f'seed{s}'
            rows = [{'instance_id': 't1', 'repo': 'a/a', 'resolved': True, 'agent_patch_size': 1,
                     'tool_calls': 3, 'total_llm_calls': 4},
                    {'instance_id': 't2', 'repo': 'b/b', 'resolved': False, 'agent_patch_size': 0,
                     'tool_calls': 1, 'total_llm_calls': 2}]
            _write_run(d, rows, {'t1': GIT_PATCH, 't2': p2}, {'t1': atif_trace()})
            dirs.append(d)
        out = sg.build(dirs)
        assert len(out) == 6
        t1 = [r for r in out if r['instance_id'] == 't1']
        assert all(r['features']['agree_lines'] == 1 and r['has_trace'] for r in t1)
        t2 = {r['sample']: r for r in out if r['instance_id'] == 't2'}
        assert t2[0]['features']['agree_lines'] == 0.5 and t2[2]['features']['patch_empty'] == 1
        assert not t2[0]['has_trace'] and 'ran_tests' not in t2[0]['features']
        # medoid shipping must skip the empty sample
        table = rc.task_table(out, 'medoid')
        assert [r['sample'] for r in table if r['instance_id'] == 't2'] == [0]
        outfile = Path(tmp) / 'f.jsonl'
        assert sg.main(['build', '--runs', *map(str, dirs), '--out', str(outfile)]) == 0
        assert len(rc.load_rows(outfile)) == 6


# ---------------------------------------------------------------------------
# risk_coverage
# ---------------------------------------------------------------------------

def test_metrics_perfect_and_constant():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], float)
    s = np.arange(10, 0, -1, dtype=float)  # perfect ranking
    assert rc.auroc(s, y) == 1.0
    assert rc.coverage_at_precision(s, y, 0.9) == 0.3
    curve = rc.risk_coverage(s, y)
    assert curve['eaurc'] < 1e-12 and curve['risk'][0] == 0.0
    const = np.ones(10)
    assert rc.auroc(const, y) == 0.5
    assert rc.coverage_at_precision(const, y, 0.9) == 0.0  # ties move together
    assert math.isinf(rc.threshold_for_precision(const, y, 0.9))
    assert len(rc.risk_coverage(const, y)['coverage']) == 1
    assert math.isnan(rc.auroc(s, np.ones(10)))


def test_calibration():
    y = np.array([0, 0, 1, 1] * 25, float)
    good = rc.reliability(np.where(y == 1, 0.99, 0.01), y)
    bad = rc.reliability(np.full(100, 0.9), y)
    assert good['ece'] < 0.02 and abs(bad['ece'] - 0.4) < 1e-9
    assert rc.brier(y, y) == 0.0


def test_bootstrap_ci():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 129).astype(float)
    s = y + rng.normal(0, 1.0, 129)
    ci = rc.bootstrap_ci(rc.auroc, s, y, n_boot=500)
    assert ci['lo'] <= ci['point'] <= ci['hi'] and ci['n_valid'] == 500
    assert 0.6 < ci['point'] < 0.95 and ci['hi'] - ci['lo'] > 0.05


def test_loro_skips_small_repo():
    groups = np.array(['fastapi'] * 67 + ['rich'] * 48 + ['requests'] * 13 + ['httpx'])
    splits = rc.loro_splits(groups, min_test=5)
    assert len(splits) == 3
    assert all('httpx' not in set(groups[te]) for _, te in splits)
    assert all((groups[tr] == 'httpx').sum() == 1 for tr, _ in splits)


def synthetic_rows(n=129, seed=0, signal=2.0):
    """Repo mix of the real task set; label driven by agreement + tests."""
    rng = np.random.default_rng(seed)
    repos = ['fastapi/fastapi'] * 67 + ['Textualize/rich'] * 48 + ['psf/requests'] * 13 + ['encode/httpx']
    rows = []
    for i in range(n):
        z = rng.normal()
        p = 1 / (1 + math.exp(-(signal * z - 1.0)))
        resolved = bool(rng.random() < p)
        empty = rng.random() < 0.1
        feats = {
            'agree_lines': float(np.clip(0.5 + 0.2 * z + rng.normal(0, 0.1), 0, 1)),
            'ran_tests_after_last_edit': float(rng.random() < p),
            'lines_changed': float(rng.integers(1, 200)),
            'patch_empty': float(empty),
            'agent_seconds': float('nan') if rng.random() < 0.2 else float(rng.uniform(60, 3600)),
        }
        rows.append({'instance_id': f't{i}', 'repo': repos[i], 'sample': 0,
                     'resolved': resolved and not empty, 'features': feats})
    return rows


def test_evaluate_end_to_end():
    sets = {'patch_size_only': ['lines_changed'],
            'signal': ['agree_lines', 'ran_tests_after_last_edit', 'agent_seconds']}
    rep = rc.evaluate(synthetic_rows(), n_boot=200, feature_sets=sets)
    assert rep['n_tasks'] == 129 and len(rep['by_repo']) == 4
    good = rep['models']['signal']['cv5']
    noise = rep['models']['patch_size_only']['cv5']
    assert good['auroc']['point'] > 0.75, good['auroc']
    assert abs(noise['auroc']['point'] - 0.5) < 0.2, noise['auroc']
    assert good['auroc']['point'] > noise['auroc']['point']
    loro = rep['models']['signal']['loro']
    assert loro['n_scored'] == 128 and len(loro['heldout_rule']['folds']) == 3
    h = good['heldout_rule']
    assert h['n_covered'] == 129 and 0 <= h['coverage'] <= 1
    assert rep['single_feature_auroc']['agree_lines']['auroc'] > 0.6
    json.dumps(rep, default=float)  # report must serialise


def test_heldout_rule_hits_target_on_strong_signal():
    rows = synthetic_rows(n=129, seed=3, signal=6.0)
    rep = rc.evaluate(rows, n_boot=100, feature_sets={'s': ['agree_lines']})
    h = rep['models']['s']['cv5']['heldout_rule']
    assert h['n_accepted'] > 0 and h['precision'] >= 0.8, h
    assert rep['models']['s']['cv5']['oracle_coverage_at_target']['point'] > 0


def test_empty_patch_never_accepted():
    rows = synthetic_rows(seed=5)
    rep = rc.evaluate(rows, n_boot=50, feature_sets={'s': ['agree_lines']})
    table = rc.task_table(rows)
    y = np.array([r['resolved'] for r in table], float)
    empty = np.array([r['features']['patch_empty'] == 1 for r in table])
    X = rc.matrix(table, ['agree_lines'])
    s = rc.oof_scores(X, y, rc.cv_splits(y), empty)
    assert (s[empty] == 0).all() and not y[empty].any()
    assert rep['n_resolved'] == int(y.sum())


if __name__ == '__main__':
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f'PASS {name}')
        except Exception as e:  # report every failure, then exit non-zero
            failed += 1
            print(f'FAIL {name}: {type(e).__name__}: {e}')
    print(f'{len(tests) - failed}/{len(tests)} passed')
    sys.exit(1 if failed else 0)
