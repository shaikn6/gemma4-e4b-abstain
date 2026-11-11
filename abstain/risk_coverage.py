"""Selective patch acceptance: AUROC, calibration, risk-coverage, bootstrap CIs.

Input: features.jsonl from signals.py (one row per task x sample).
The unit of analysis is the task: per task we ship one patch (sample 0, or
the medoid of the k samples) and either accept it or abstain.

Coverage = accepted tasks / all tasks (empty patches count, always abstain).
Precision = resolved among accepted. Risk = 1 - precision.

Two kinds of threshold are reported and must not be confused in the paper:
  oracle  threshold picked on the same out-of-fold scores it is scored on
          (upper bound; describes the score, not a deployable rule)
  heldout threshold picked on the training fold only (inner CV), then applied
          to the held-out fold; this is the deployable number.

Usage:
  python -I abstain/risk_coverage.py --features features.jsonl --out report.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SHAPE = ['files_touched', 'lines_added', 'lines_removed', 'lines_changed',
         'edit_regions', 'new_files', 'touches_non_py']
TRACE = ['tool_calls', 'llm_calls', 'agent_seconds', 'n_edits', 'n_reads',
         'n_graph_calls', 'n_tool_errors', 'n_nudges', 'ran_tests', 'n_test_runs',
         'n_test_runs_failed', 'ran_tests_after_last_edit', 'last_test_passed',
         'last_test_n_failed', 'completion_tokens', 'reasoning_chars']
AGREE = ['agree_lines', 'agree_content', 'agree_files', 'agree_exact', 'k_nonempty']
FEATURE_SETS = {
    'patch_size_only': ['lines_changed'],
    'shape': SHAPE,
    'trace': TRACE,
    'agreement': AGREE,
    'shape+trace': SHAPE + TRACE,
    'all': SHAPE + TRACE + AGREE,
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_rows(path: Path) -> list[dict]:
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def task_table(rows: list[dict], ship: str = 'first') -> list[dict]:
    """One row per task: the shipped sample ('first' = sample 0, 'medoid' =
    non-empty sample with highest line agreement)."""
    by_task: dict[str, list[dict]] = {}
    for r in rows:
        by_task.setdefault(r['instance_id'], []).append(r)
    out = []
    for iid in sorted(by_task):
        samples = sorted(by_task[iid], key=lambda r: r['sample'])
        if ship == 'medoid':
            def key(r):
                f = r['features']
                if f.get('patch_empty', 0.0) == 1.0:
                    return (-1.0, -r['sample'])
                a = f.get('agree_lines', float('nan'))
                return (a if a == a else 0.0, -r['sample'])
            out.append(max(samples, key=key))
        else:
            out.append(samples[0])
    return out


def matrix(table: list[dict], names: list[str]) -> np.ndarray:
    return np.array([[float(r['features'].get(n, float('nan'))) for n in names]
                     for r in table], dtype=float)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def auroc(scores: np.ndarray, y: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    if len(set(y.tolist())) < 2:
        return float('nan')
    return float(roc_auc_score(y, scores))


def brier(probs: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(probs) - np.asarray(y)) ** 2))


def reliability(probs: np.ndarray, y: np.ndarray, n_bins: int = 5) -> dict:
    """Equal-mass bins (n is small). Returns ECE and the per-bin table."""
    probs, y = np.asarray(probs, float), np.asarray(y, float)
    order = np.argsort(probs, kind='mergesort')
    bins = [b for b in np.array_split(order, n_bins) if len(b)]
    table, ece = [], 0.0
    for b in bins:
        conf, acc = float(probs[b].mean()), float(y[b].mean())
        ece += len(b) / len(y) * abs(conf - acc)
        table.append({'n': int(len(b)), 'mean_prob': conf, 'frac_resolved': acc})
    return {'ece': float(ece), 'bins': table}


def risk_coverage(scores: np.ndarray, y: np.ndarray) -> dict:
    """Accept every task with score >= t, for each distinct t (ties move
    together). Returns coverage, precision, risk arrays and AURC."""
    scores, y = np.asarray(scores, float), np.asarray(y, float)
    n = len(y)
    order = np.argsort(-scores, kind='mergesort')
    s_sorted, y_sorted = scores[order], y[order]
    cum = np.cumsum(y_sorted)
    # last index of each tie group
    ends = np.r_[np.nonzero(np.diff(s_sorted))[0], n - 1]
    k = ends + 1
    prec = cum[ends] / k
    cov = k / n
    risk = 1 - prec
    # AURC over all n acceptance sizes, tie groups share the group's risk.
    per_k = np.repeat(risk, np.diff(np.r_[0, k]))
    aurc = float(per_k.mean())
    # Optimal AURC for this label set (perfect ranking), for excess AURC.
    pos = int(y.sum())
    opt = np.array([0.0 if i < pos else (i + 1 - pos) / (i + 1) for i in range(n)])
    return {'threshold': s_sorted[ends].tolist(), 'coverage': cov.tolist(),
            'precision': prec.tolist(), 'risk': risk.tolist(),
            'aurc': aurc, 'eaurc': aurc - float(opt.mean())}


def threshold_for_precision(scores: np.ndarray, y: np.ndarray, target: float) -> float:
    """Lowest threshold whose accepted set has precision >= target (max
    coverage). +inf if no threshold reaches it (abstain on everything)."""
    rc = risk_coverage(scores, y)
    ok = [t for t, p in zip(rc['threshold'], rc['precision']) if p >= target]
    return min(ok) if ok else math.inf


def coverage_at_precision(scores: np.ndarray, y: np.ndarray, target: float = 0.9) -> float:
    t = threshold_for_precision(scores, y, target)
    return float(np.mean(np.asarray(scores) >= t)) if math.isfinite(t) else 0.0


def accept_stats(scores: np.ndarray, y: np.ndarray, t: float) -> dict:
    acc = np.asarray(scores) >= t
    n_acc = int(acc.sum())
    return {'coverage': float(acc.mean()), 'n_accepted': n_acc,
            'precision': float(np.asarray(y)[acc].mean()) if n_acc else float('nan')}


def bootstrap_ci(fn: Callable[[np.ndarray, np.ndarray], float], scores, y,
                 n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile CI, resampling tasks with replacement. NaN draws (one-class
    resamples) are dropped and counted."""
    scores, y = np.asarray(scores, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = fn(scores[idx], y[idx])
        if v == v:
            vals.append(v)
    point = fn(scores, y)
    if not vals:
        return {'point': point, 'lo': float('nan'), 'hi': float('nan'), 'n_valid': 0}
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return {'point': point, 'lo': float(lo), 'hi': float(hi), 'n_valid': len(vals)}


# ---------------------------------------------------------------------------
# Models and splits
# ---------------------------------------------------------------------------

def make_model():
    return make_pipeline(SimpleImputer(strategy='median', add_indicator=True),
                         StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def _fit_predict(X_tr, y_tr, X_te) -> np.ndarray:
    if len(set(y_tr.tolist())) < 2:
        return np.full(len(X_te), float(np.mean(y_tr)) if len(y_tr) else 0.5)
    m = make_model()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        warnings.simplefilter('ignore', UserWarning)  # all-NaN columns dropped
        m.fit(X_tr, y_tr)
        return m.predict_proba(X_te)[:, 1]


def _stratified_splits(y, n_splits, seed):
    n_splits = max(2, min(n_splits, int(min(np.bincount(y.astype(int), minlength=2)))))
    return list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def cv_splits(y: np.ndarray, n_splits: int = 5, seed: int = 0):
    return _stratified_splits(y, n_splits, seed)


def loro_splits(groups: np.ndarray, min_test: int = 5):
    """Leave-one-repo-out. Repos with fewer than min_test tasks are never a
    test fold (they stay in training); their tasks get no held-out score."""
    groups = np.asarray(groups)
    out = []
    for tr, te in LeaveOneGroupOut().split(np.zeros(len(groups)), groups=groups):
        if len(te) >= min_test:
            out.append((tr, te))
    return out


def oof_scores(X, y, splits, empty_mask=None) -> np.ndarray:
    """Out-of-fold probability for every row covered by a test fold, NaN
    elsewhere. Empty patches are forced to 0 (never accepted)."""
    scores = np.full(len(y), np.nan)
    for tr, te in splits:
        scores[te] = _fit_predict(X[tr], y[tr], X[te])
    if empty_mask is not None:
        scores[np.asarray(empty_mask, bool)] = 0.0
    return scores


def heldout_selective(X, y, splits, target=0.9, empty_mask=None, seed=0) -> dict:
    """Deployable rule: per outer fold, pick the threshold from inner-CV
    scores on the training fold, then apply it to the test fold."""
    empty = np.zeros(len(y), bool) if empty_mask is None else np.asarray(empty_mask, bool)
    accepted = np.zeros(len(y), bool)
    covered = np.zeros(len(y), bool)
    folds = []
    for tr, te in splits:
        inner = [(tr[a], tr[b]) for a, b in _stratified_splits(y[tr], 5, seed)] \
            if len(set(y[tr].tolist())) == 2 else []
        inner_scores = oof_scores(X, y, inner, empty)[tr] if inner else np.zeros(len(tr))
        t = threshold_for_precision(inner_scores, y[tr], target)
        te_scores = _fit_predict(X[tr], y[tr], X[te])
        te_scores[empty[te]] = 0.0
        acc = te_scores >= t
        accepted[te], covered[te] = acc, True
        folds.append({'n_test': int(len(te)), 'threshold': t,
                      **accept_stats(te_scores, y[te], t)})
    n_acc = int(accepted[covered].sum())
    return {'folds': folds, 'n_covered': int(covered.sum()), 'n_accepted': n_acc,
            'coverage': float(accepted[covered].mean()) if covered.any() else float('nan'),
            'precision': float(y[accepted].mean()) if n_acc else float('nan')}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def single_feature_auroc(table: list[dict], y: np.ndarray) -> dict:
    """Raw AUROC of each feature (>0.5: larger value -> more often resolved).
    Descriptive only; direction is not fitted."""
    names = sorted({k for r in table for k in r['features']})
    out = {}
    for n in names:
        x = matrix(table, [n])[:, 0]
        ok = ~np.isnan(x)
        out[n] = {'auroc': auroc(x[ok], y[ok]), 'n': int(ok.sum())}
    return out


def evaluate(rows: list[dict], ship: str = 'first', target: float = 0.9,
             n_boot: int = 2000, seed: int = 0, feature_sets: dict | None = None) -> dict:
    table = task_table(rows, ship)
    y = np.array([float(r['resolved']) for r in table])
    groups = np.array([r['repo'] for r in table])
    empty = np.array([r['features'].get('patch_empty', 0.0) == 1.0 for r in table])
    report = {
        'n_tasks': len(table), 'n_resolved': int(y.sum()),
        'base_rate': float(y.mean()) if len(y) else float('nan'),
        'ship': ship, 'target_precision': target,
        'by_repo': {g: {'n': int((groups == g).sum()), 'resolved': int(y[groups == g].sum())}
                    for g in sorted(set(groups.tolist()))},
        'always_accept': {'coverage_at_target': 1.0 if y.mean() >= target else 0.0,
                          'precision': float(y.mean())},
        'single_feature_auroc': single_feature_auroc(table, y),
        'models': {},
    }
    cov_fn = lambda s, t: coverage_at_precision(s, t, target)  # noqa: E731
    for name, feats in (feature_sets or FEATURE_SETS).items():
        X = matrix(table, feats)
        entry = {}
        for split_name, splits in (('cv5', cv_splits(y, 5, seed)), ('loro', loro_splits(groups))):
            if not splits:
                entry[split_name] = None
                continue
            s = oof_scores(X, y, splits, empty)
            m = ~np.isnan(s)
            rc = risk_coverage(s[m], y[m])
            entry[split_name] = {
                'n_scored': int(m.sum()),
                'auroc': bootstrap_ci(auroc, s[m], y[m], n_boot, seed),
                'aurc': bootstrap_ci(lambda a, b: risk_coverage(a, b)['aurc'], s[m], y[m], n_boot, seed),
                'eaurc': rc['eaurc'],
                'oracle_coverage_at_target': bootstrap_ci(cov_fn, s[m], y[m], n_boot, seed),
                'brier': brier(s[m], y[m]),
                'calibration': reliability(s[m], y[m]),
                'heldout_rule': heldout_selective(X, y, splits, target, empty, seed),
                'curve': {k: rc[k] for k in ('coverage', 'precision')},
            }
        report['models'][name] = entry
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--features', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--ship', choices=['first', 'medoid'], default='first')
    ap.add_argument('--target', type=float, default=0.9)
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args(argv)
    rep = evaluate(load_rows(args.features), args.ship, args.target, args.n_boot, args.seed)
    args.out.write_text(json.dumps(rep, indent=1, default=float), encoding='utf-8')
    print(f"tasks={rep['n_tasks']} resolved={rep['n_resolved']} base_rate={rep['base_rate']:.3f}")
    for name, e in rep['models'].items():
        cv = e.get('cv5')
        if cv:
            a, c = cv['auroc'], cv['heldout_rule']
            print(f"{name:16s} cv5 AUROC {a['point']:.3f} [{a['lo']:.3f},{a['hi']:.3f}]  "
                  f"heldout cov {c['coverage']:.3f} prec {c['precision']:.3f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
