# Why Gemma 4 E4B Fails: A Failure-Mode Taxonomy

We ran Gemma 4 E4B as an autonomous SWE-agent on 127 SWE-bench tasks (fastapi,
rich, requests, httpx; two degenerate tasks `fastapi_14077` and `fastapi_14851`
excluded). On seed 0 the model resolved 11 tasks and failed 116. Aggregate
pass@1 tells us *how often* the agent fails; it does not tell us *why*. This
section categorizes every failed trajectory into one dominant failure mode,
assigned from the session trace (per-step `tool_calls`, their `observation`
status and exit codes) and from parsing the submitted patch. Each failed task
is given a single primary mode by a fixed priority rule
(`taxonomy.py`); the modes are mutually exclusive and exhaustive.

## Distribution (seed 0, 116 failed tasks)

| Failure mode | Count | % of failures | Representative instances |
|---|---:|---:|---|
| Submitted after an uninformative test (`tested_but_unresolved`) | 64 | 55.2% | fastapi_15588, fastapi_15030, fastapi_14964 |
| Never ran any test before submitting (`never_tested`) | 21 | 18.1% | fastapi_14986, requests_7502, rich_4070 |
| Broken / non-applying edits dominate (`broken_edits`) | 11 | 9.5% | fastapi_15589, requests_7328, fastapi_12942 |
| Empty / no effective patch (`empty_patch`) | 10 | 8.6% | fastapi_15280, rich_4075, fastapi_14258 |
| Ran out of context / tool-call budget (`budget_context`) | 8 | 6.9% | fastapi_14962, fastapi_14873, rich_3953 |
| Wrong-file / localization miss, inferred (`wrong_file`) | 1 | 0.9% | fastapi_15661 |
| Huge / scattershot patch (`scattershot`) | 1 | 0.9% | fastapi_14371 |

The shape is highly stable across all eight seeds: `tested_but_unresolved`
holds 44–59% of failures (median ~52%), `never_tested` 10–18%, and together the
two verification-related modes account for roughly 65–70% of every seed's
failures. `empty_patch` (10–24), `broken_edits` (7–17) and `budget_context`
(8–13) make up most of the remainder.

## What the dominant mode actually is

`tested_but_unresolved` is the single largest bucket, but its honest reading is
not "the agent saw a red test and submitted anyway." Of the 64 tasks, **zero**
showed a clean non-zero exit code in the agent's own test runs that it then
ignored. Instead, 21 ran a local test that *passed* (green but insufficient to
catch the real bug) and 43 ran a test command that *errored out* — the test
could not even be collected or executed (`run_command` returned
`status: error` with no exit code) — and the agent submitted regardless. In 62
of the 64 cases the agent did run a test after its last edit. This is a
**verification-gap** failure: the model reaches a plausible stopping point on a
weak or broken local signal rather than confirming the fix against the issue's
reproduction test.

Concrete example — `fastapi_15588`: the agent read and edited `fastapi/sse.py`
over four `edit_file` calls (one rejected mid-way), ran
`find tests -name "*sse*"`, then `pytest tests/test_sse.py`, which exited 0. It
submitted immediately. The harness's gold test returned exit 1: the patch
touched the right file but did not satisfy the hidden reproduction test, and the
one local test the agent chose to run was too weak to reveal it.

`never_tested` (18%) is the clean complement: the agent edited source and
submitted a patch without issuing any test command at all (e.g. `fastapi_14986`
edited `openapi/models.py` + `openapi/utils.py` and stopped).

## Resolved vs. failed: behavioral separation

| Signal (median unless noted) | Resolved (n=11) | Failed (n=116) |
|---|---:|---:|
| Tool calls | 6 | 9.5 |
| Total LLM calls | 8 | 12 |
| Patch size (bytes) | 553 | 739.5 |
| Files read | 2 | 4 |
| Ran a test at all | 82% | 66% |
| Ran a test *after* the last edit | 82% | 63% |

The three sharpest contrasts:

1. **Failed runs flail longer.** Median tool calls 9.5 vs 6 and LLM calls 12 vs
   8 — failure correlates with more, not fewer, steps, and with more files read
   (median 4 vs 2). The agent is searching, not converging.
2. **Failed runs verify less.** 82% of resolved runs ran a test after their
   final edit; only 63% of failed runs did. Skipped or post-edit-absent
   verification is a direct, trace-visible failure signal.
3. **Failed patches are larger** (median 739 vs 553 bytes), consistent with the
   flailing tail: bigger, less targeted edits.

## Implication for the abstention signal

Roughly **three failures in four are verification failures the trace can see**:
the agent either never ran a test (18%) or submitted on a test signal that was
green-but-insufficient or outright errored (55%). The decisive, cheap signals —
*did the agent run a test after its last edit, and did that test produce a clean
pass* — are exactly the features an abstention head reads from the trajectory,
without any ground-truth label. The behavioral gap (resolved runs are shorter
and verify more) says the model's own trajectory already carries most of the
information needed to predict its failure.

## Heuristic limits (honest caveats)

- **Localization is inferred, not labeled.** We have no gold modified-file list,
  so true wrong-file localization misses cannot be measured directly. The
  `wrong_file` mode counts only the unambiguous case (patch edits test files
  only, no source) and is therefore a floor — many `tested_but_unresolved`
  failures are almost certainly wrong-fix or wrong-location errors we cannot
  separate without gold files.
- **"Ran tests"** is detected by matching `pytest|unittest|tox|nosetests` in the
  issued command; a hand-rolled test invocation would be missed.
- **Priority ordering** assigns one dominant mode per task; a task with several
  problems is counted once, under the highest-priority matching mode.
- Numbers are seed 0 unless stated; cross-seed ranges are reported above.
  Regenerate with `~/.venvs/ntl/bin/python3 -I taxonomy.py`.
