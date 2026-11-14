# gemma4-e4b-abstain

> **Teaching a small local Gemma code agent to say "not sure" and skip the patch instead of breaking your repo (paper + code + data).**

A small, locally-runnable code agent writes a patch for almost every bug it sees and fixes
almost none of them. Running Google's **Gemma 4 E4B** (4-bit quantized, served locally) as an
autonomous agent on a 127-task SWE-bench-*style* harness, it produces a non-empty patch on
115 of 129 tasks but resolves only ~11 per seed (8.7%). It is, precisely, *confidently wrong*.

This repo asks a selective-prediction question: **can the agent tell its good patches from its
bad ones using only signals it can compute itself** — the shape of the diff, its own run trace,
and how far its patches agree across repeated seeds — without ever seeing the hidden test?

**Headline result.** Cross-seed agreement ranks correct patches above incorrect ones at
**AUROC 0.840** (95% CI [0.745, 0.924]); patch size alone is near chance (0.614). But at ~11
positives this makes a *ranking* signal, not a high-precision autonomous accept gate: a
90%-precision gate is unreachable, while a moderate 30%-precision target accepts 14.2% of tasks
at 33.3% precision (≈3.8× the base rate). Consistency is **necessary, not sufficient** — on one
task all eight seeds emit the identical patch and it is wrong on all eight.

## What's here

| Path | What it is |
|---|---|
| `paper/draft_v1.md` | The paper (also `gemma_paper.pdf` / `.html`), with `fig_risk_coverage.svg` |
| `paper/failure_taxonomy.md` | Why E4B fails, by dominant failure mode |
| `abstain/signals.py` | Turns raw agent runs (patches + traces) into 37 non-leaky features |
| `abstain/risk_coverage.py` | AUROC, risk–coverage, E-AURC and bootstrap CIs — reproduces the paper |
| `abstain/task_analysis.py` | Descriptive stats of the task set (no model runs) |
| `abstain/test_synthetic.py` | Synthetic-data tests for the two scripts above |
| `taxonomy.py` | Assigns each unresolved task a failure mode from trace + patch |
| `dataset/features8_clean.jsonl` | 1,016 labeled runs (127 tasks × 8 E4B seeds), 37 features + label |
| `dataset/features_12b.jsonl` | 127 rows, Gemma 4 12B single seed, for the complementarity study |
| `dataset/DATASHEET.md` | Datasheet (Gebru et al. 2021 structure) |

## Reproduce the paper's numbers

The released feature files reproduce every ranking number in the paper (no GPU, no model, no
network). Agreement depends on the per-task seed set, so pass the 8-seed file:

```bash
python -m venv .venv && source .venv/bin/activate && pip install numpy scikit-learn
python -I abstain/risk_coverage.py --features dataset/features8_clean.jsonl --out rc.json --target 0.3
```

Expected (verified):

```
tasks=127 resolved=11 base_rate=0.087
patch_size_only  cv5 AUROC 0.614 [0.461,0.773]  heldout cov 0.000 prec nan
shape            cv5 AUROC 0.711 [0.583,0.829]  heldout cov 0.000 prec nan
trace            cv5 AUROC 0.738 [0.580,0.885]  heldout cov 0.055 prec 0.143
agreement        cv5 AUROC 0.840 [0.745,0.924]  heldout cov 0.142 prec 0.333
shape+trace      cv5 AUROC 0.744 [0.593,0.877]  heldout cov 0.063 prec 0.125
all              cv5 AUROC 0.799 [0.653,0.922]  heldout cov 0.079 prec 0.300
```

Tests: `python -I abstain/test_synthetic.py` (also runs under `pytest`).

Rebuilding features from scratch with `abstain/signals.py` needs the raw per-run `traces/` and
`patches/`, which are large and not committed here; the feature files above are the released
analysis inputs.

## Signal families

- **Shape** — files touched, lines added/removed/changed, edit regions, new files, non-Python edits.
- **Trace** — tool-call and step counts, reads/edits, tool errors, whether it ran tests, whether it ran a test *after its last edit* and the outcome, tokens, wall-time.
- **Agreement** — cross-seed consistency of the patches, computed leave-one-out (a sample's score excludes itself).

## Honest scope

One quantized model on one 127-task SWE-bench-*style* harness (**not** the SWE-bench dataset;
do not compare to its leaderboard). With ~11 positives the confidence intervals are wide, and
every number is reported with a CI. Two degenerate tasks whose hidden test passes with an empty
patch are excluded and flagged. The agreement gate can be gamed by a consistently-wrong agent.
The E4B and 12B models solve *different* bugs (union 22/127 > either alone), but because the
small model confidently accepts so little, routing does not save compute at this scale.

## License

Apache-2.0. Built for the Gemma 4 Developer Agent paper track. No personal data.
