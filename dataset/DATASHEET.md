# Datasheet — Gemma 4 E4B/12B selective-acceptance traces

A labeled dataset of autonomous SWE-agent runs for studying *selective patch acceptance* (when a code agent should abstain). Follows the Gebru et al. (2021) datasheet structure. Released under Apache-2.0.

## Motivation
Small local code agents produce a patch for almost every bug but fix few of them, and cannot see the hidden test. This dataset lets researchers study whether an agent's own behavior/trace predicts patch correctness, without the hidden test. Built for the Gemma 4 Developer Agent Paper Track; no personal data.

## Composition
- **Instances.** One row per (task, seed) agent run. **1,016 rows** = 127 tasks × 8 seeds (Gemma 4 **E4B**), plus the Gemma 4 **12B** single-seed run (127 rows) shipped separately as `features_12b.jsonl`.
- **Tasks.** 127 real GitHub bug-fix tasks (SWE-bench-*style*, **not** the SWE-bench dataset) over 4 Python repos: fastapi/fastapi (65 tasks), Textualize/rich (48), psf/requests (13), encode/httpx (1). Repo coverage is imbalanced — httpx has a single task; treat per-repo claims with care.
- **Label.** `resolved` (bool) = the hidden test passed. 66/1016 E4B rows resolved (~6.5%); 11/127 tasks resolve on the seed-0 run, 17/127 on ≥1 of 8 seeds.
- **Features (37, all non-leaky).** Diff *shape* (files_touched, lines_added/removed/changed, edit_regions, new_files, touches_non_py, patch_empty), run *trace* (tool_calls, llm_calls, n_edits/reads/graph_calls, n_tool_errors, ran_tests, ran_tests_after_last_edit, last_test_passed/n_passed/n_failed, n_submits, n_nudges, budget_refused, prompt/completion/reasoning tokens, agent_seconds, trace_agent_steps), and cross-seed *agreement* (agree_lines/content/files/exact, k_samples, k_nonempty) computed leave-one-out (a sample's agreement excludes itself).
- **Excluded.** Label-leaking fields (`test_exit_code`, `error`, `duration_seconds`, raw `resolved` is kept only as the label column). Two degenerate tasks (`fastapi_14077`, `fastapi_14851`) whose hidden test passes with an empty patch are removed.
- Raw per-run `traces/` (SessionTrace JSON) and `patches/` are included so others can derive new features.

## Collection
Runs produced by the Kaggle Gemma 4 Developer Agent harness on 4×L4 GPUs, vLLM-served, fixed sampling (temperature 1.0, top_p 0.95, top_k 64), budget 10 min / 40 tool calls per task; only the random seed varies across the 8 E4B samples. Labels are the harness's own hidden-test outcomes. No human annotation.

## Preprocessing
`signals.py` parses each patch and trace into the 37 features (leave-one-out agreement across the per-task seed set). `features8_clean.jsonl` is the analysis file (degenerate tasks removed). Scripts (`signals.py`, `risk_coverage.py`) are included and reproduce every number in the paper.

## Uses
Selective prediction / abstention for code agents; correctness prediction from behavior; small-vs-large complementarity; agreement-as-confidence studies. **Not** a general code-repair training set and **not** representative of SWE-bench; the task count is small (wide confidence intervals) and the repo mix is skewed.

## Distribution & maintenance
Files: `features8_clean.jsonl`, `features_12b.jsonl`, `traces/`, `patches/`, `signals.py`, `risk_coverage.py`, this datasheet. Apache-2.0. Versioned with the paper; the first follow-up is more labeled tasks to narrow the intervals.
