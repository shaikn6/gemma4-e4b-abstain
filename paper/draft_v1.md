# Knowing When Not to Patch: Selective Patch Acceptance for a Small Local Code Agent

**Draft v1 — 2026-11-04. Numbers are from real runs on disk (`runs/ss/`). All arXiv citations in the References were opened and verified by the author (title + first author confirmed); the only two not personally opened are Farquhar 2024 (Nature) and Li 2022 (Science), both well-known venue papers.**

## Abstract

Small, locally-runnable language models are attractive as software-engineering agents because they are cheap and private, but they fail most repository-level bug-fix tasks while still emitting a confident-looking patch. Running Google's Gemma 4 E4B (a 4-bit-quantized, locally-servable model) as an autonomous agent on 129 SWE-bench-style bug-fix tasks, we find it produces a non-empty patch on 115 of 129 tasks yet resolves only 12 (9.3%). Two of those 129 tasks are satisfied by an *empty* patch — their hidden test passes with no change at all — so their gold fix is degenerate; we exclude them, leaving 127 tasks on which the agent genuinely resolves 11 (8.7%) per seed. We ask whether the agent can tell its good patches from its bad ones *without* access to the hidden evaluation tests, using only signals it can compute itself: the shape of the diff, behavioral features of its own run trace, and the agreement of its patches across repeated samples (different random seeds). Across eight seeds (1,016 agent runs) we show these signals rank correct patches above incorrect ones — cross-seed agreement is the strongest family (AUROC 0.840, 95% CI [0.745, 0.924]), while patch size alone is near chance (0.614). However, converting this ranking into a deployable, high-precision accept/reject gate is hard at this data scale: a 90%-precision target is unreachable even for an oracle threshold, while a moderate 30%-precision target yields a held-out rule that accepts ~14% of tasks at ~33% precision (≈3.8× the base rate). We conclude that behavioral selective acceptance is a useful *ranking* signal rather than an autonomous acceptance gate; a 12B model solves different bugs than E4B (union 22/127 vs 17 for either alone), so the two are complementary, but because the small model confidently accepts so little, routing does not save compute at this scale. We release the full labeled trace dataset to let others study the signal with more positives.

## 1. Introduction

A coding agent that silently submits a wrong patch is worse than one that says "I am not sure." As small local models (runnable on a single workstation GPU, no API calls, no data egress) are increasingly proposed for autonomous software maintenance, this failure mode becomes acute: the model is weak enough to be wrong most of the time, but fluent enough to always produce something that *looks* like a fix.

We quantify this with Gemma 4 E4B on a 129-task SWE-bench-style harness: **115/129 non-empty patches, 12/129 actually correct.** The agent is, in the precise sense, confidently wrong. Our question is the *selective prediction* question applied to a code agent: can the agent abstain on the patches it is likely to get wrong, so that what it *does* submit is reliable, using only information available at inference time (never the hidden test)?

Our contributions:
1. An empirical characterization of a small local SWE agent's confidently-wrong failure mode, with the seed-dependence of its successes measured across eight samples.
2. A lightweight selective-acceptance detector built from three signal families — diff *shape*, run-*trace* behavior, and cross-seed *agreement* — evaluated with risk–coverage analysis and bootstrap confidence intervals.
3. An honest negative-leaning result: the signals rank correctness (AUROC up to 0.840) but do not support a high-precision autonomous accept gate at ~11 positives; they do support moderate-precision deferral.
4. A released dataset of 1,032 labeled agent runs (8 seeds × 129 tasks; traces + patches), with the two degenerate tasks flagged, for the community.

## 2. Related Work

**Selective prediction and the reject option.** The reject option dates to Chow (1970)†. El-Yaniv & Wiener (2010, JMLR)† formalized the risk–coverage trade-off for a selective predictor (f, g); Geifman & El-Yaniv (2017, NeurIPS)† applied softmax-response thresholding to deep nets, and SelectiveNet (Geifman & El-Yaniv, 2019, ICML)† learned a reject head end-to-end. We adopt their coverage/selective-risk definitions and report E-AURC (Geifman, Uziel & El-Yaniv, arXiv:1805.08206)† because, at an 8.7% base rate, raw AURC is dominated by irreducible error. Recent surveys extend abstention to LLMs; we position against them but note our unit is a whole code patch, not a classification label.

**Confidence and calibration for code.** Verbalized confidence (Tian et al., 2023; Xiong et al., 2024) and logit/entropy methods (semantic entropy, Kuhn et al., 2023; Farquhar et al., 2024, Nature) estimate answer reliability, but Spiess et al. (ICSE 2025, arXiv:2402.02047) show token probabilities are poorly calibrated for code — at >90% model confidence only ~52% of completions pass — motivating our use of behavioral signals over logits. We also measured verbalized confidence to be near-useless here (consistent with ~0.57 AUROC reported for code failure prediction).

**Self-consistency / agreement.** Self-consistency (Wang et al., 2022)† and execution-agreement selection for code (CodeT, Chen et al., 2022†; AlphaCode clustering, Li et al., 2022, Science) use agreement across samples to *select* a candidate. We instead use cross-seed agreement as an *abstention* signal, and heed the "consensus can be confidently wrong" caveat: agreement is treated as a necessary condition to act (abstain on disagreement), not a sufficient condition to accept.

**Verifiers and the limits of resampling.** Trained verifiers/reward models (Cobbe et al., 2021†; Lightman et al., 2023†; LEVER, Ni et al., 2023; Coder-Reviewer, Zhang et al., 2023) rerank candidates but need large labeled data — infeasible with 12 positives, so we train no verifier network. Stroebl et al. (arXiv:2411.17501)† show an imperfect verifier imposes a false-positive floor that resampling cannot cross — direct support for abstaining rather than sampling harder.

**SWE agents and confidently-wrong patches.** SWE-bench (Jimenez et al., ICLR 2024)†, SWE-agent (Yang et al., 2024)† and Agentless (Xia et al., 2024)† define the task and scaffolds; SWE-Bench+ (arXiv:2410.06992) and related audits show a large fraction of "plausible" patches are actually wrong or test-gaming. This grounds our premise.

**Closest prior art (author-verified).** Several recent papers are adjacent and we differentiate explicitly:
- *EarlyEval* (Shi et al., arXiv:2609.02783)†: behavioral trajectory-prefix classifiers predict agent outcome — but for cheaper benchmark *evaluation*, on large models, not abstention.
- *A Draft-Model Gate* (Grotov & Malykh, arXiv:2609.05274)†: a small model gates a coding agent via token cross-likelihood and *vetoes*; we use discrete behavioral + cross-seed signals and frame *deferral/routing*.
- *Coding Agents Don't Know When to Act* / FixedBench (Gloaguen et al., arXiv:2605.07769)†: abstention on whether to act at all, not patch-correctness prediction.
- *Is Your Automated Software Engineer Trustworthy?* / BouncerBench (Mathews & Nagappan, arXiv:2506.17812)† and *Abstain and Validate* (Cambronero et al., arXiv:2510.03217)†: act-vs-abstain and dual-LLM gates, using LLM judges rather than cheap trace + cross-seed agreement; *Task Abstention for LLMs in Code Generation* (Zhou et al., arXiv:2605.17029)† gives a distribution-free guarantee for function-level generation with self-generated tests.

We found no prior work combining risk–coverage/E-AURC evaluation + a whole-patch acceptance unit + a **small quantized local** agent + behavioral trajectory signals + **cross-seed self-consistency** + an existing-(non-hidden)-tests gate + a released trace dataset. We claim this intersection, not priority over any single ingredient (we do not say "first").

## 3. Experimental Setup

**Model & harness.** Gemma 4 E4B (`gemma-4-e4b-it-qat-w4a16-ct`), served locally via vLLM (bf16, 4×L4) inside the Kaggle *Gemma 4 Developer Agent* harness. Sampling fixed across all runs (temperature 1.0, top-p 0.95, top-k 64); only the seed varies, so the eight samples are exchangeable and "seed 0" is a fair stand-in for one deployment. Budget: 10 min / 40 tool calls per task.

**Tasks.** 129 tasks over four real Python repositories — Textualize/rich, encode/httpx, fastapi/fastapi, psf/requests — with instance ids such as `fastapi_15661`, `rich_3471`, `requests_6592`. This is a SWE-bench-*style* harness (real repos, hidden tests), **not the SWE-bench dataset**; we therefore do not compare our resolve rate to SWE-bench leaderboard numbers.

**Runs.** Eight seeds × 129 tasks = 1,032 agent runs, ~1.5 GPU-h each (12.2 GPU-h total). Each run yields a patch, a hidden-test outcome (the label), and a full SessionTrace. Label-leaking fields (`resolved`, `test_exit_code`, `error`, `duration_seconds`) are excluded from all features; traces are recorded before the hidden test runs, so trace features are safe.

**Degenerate-task exclusion.** Two tasks (`fastapi_14077`, `fastapi_14851`) are "resolved" on some seeds by an *empty* patch: the hidden test passes with no edit, so a null submission scores as a fix. These do not demonstrate repair and would inflate the positive rate, so we drop them, leaving **127 tasks and 1,016 runs** for all analysis below. We report this rather than silently keeping the easy positives; removing them in fact strengthens the signal (the no-op "successes" were unpredictable from behavior).

## 4. Method

**Unit and metrics.** The unit is the task: we ship one patch per task (seed 0, or the medoid of the samples) and either accept or abstain. Coverage = accepted / 127 (empty patches always abstain); precision = resolved among accepted; selective risk = 1 − precision. We report AUROC (ranking quality, feature-set comparison), the risk–coverage curve, E-AURC, and operating points at target precisions, each with 2,000-sample bootstrap CIs. Thresholds are reported two ways: *oracle* (upper bound, picked on the scores it is scored on) and *held-out* (picked on a training fold, applied out-of-fold — the deployable number).

**Signal families.** (a) *Shape*: files touched, lines added/removed/changed, edit regions, new files, non-Python edits. (b) *Trace*: tool-call and step counts, reads/edits/graph-queries, tool errors, whether the agent ran tests, whether it ran tests after its last edit and their outcome, tokens, agent wall-time. (c) *Agreement*: cross-seed consistency of the patches (shape/behavior agreement across the samples). A logistic model over each family (and their unions) is fit with 5-fold stratified CV and, separately, leave-one-repo-out.

## 5. Results

**Resolution is seed-dependent.** Per seed the agent resolves ~11/127 (8.7%). Across eight seeds, the count of tasks resolved in exactly k seeds is {0:110, 1:5, 2:3, 3:1, 4:1, 5:1, 6:1, 7:3, 8:2}: **17/127 solve on at least one seed, only 2 on all eight.** The agent is unstable precisely at the edge of its ability — the variance that makes cross-seed agreement informative.

**Signals rank correctness (AUROC, 5-fold CV, ship = seed 0; 127 tasks, base rate 0.087):**

| feature family | AUROC | 95% CI |
|---|---|---|
| patch size only | 0.614 | [0.461, 0.773] |
| shape | 0.711 | [0.583, 0.829] |
| trace | 0.738 | [0.580, 0.885] |
| **agreement (cross-seed)** | **0.840** | **[0.745, 0.924]** |
| shape + trace | 0.744 | [0.593, 0.877] |
| all | 0.799 | [0.653, 0.922] |

Cross-seed agreement is by far the strongest family, with a confidence interval (lower bound 0.745) well clear of chance; patch size alone is weakest. Correct patches also use fewer steps (median 7 vs 10 tool calls) and smaller diffs in the tail (wrong-patch mean 4,396 vs 660 characters).

The full risk–coverage curves are in Figure 1 (`paper/fig_risk_coverage.svg`): agreement dominates across the coverage range, all families sit above the 0.087 base rate, and every curve collapses toward the base rate as coverage approaches 1.

**Deployable accept gate (held-out thresholds, 127 tasks):**
- Target precision 0.9: coverage 0 for every family, and oracle coverage@0.9 = 0 — a high-precision autonomous accept gate is unreachable at this scale.
- Target 0.5: unreliable (coverage near 0).
- Target 0.3: **agreement accepts 14.2% of tasks at 33.3% precision; all-features 7.9% at 30.0%** — ≈3.5–3.8× the 8.7% base rate, meeting the target.

**Sample budget (how many seeds the agreement signal needs).** Agreement is the one family that depends on drawing multiple samples, so we measure how its ranking scales with the sample budget k (a deployer pays k× inference for k samples): agreement AUROC rises **0.662 (k=2) → 0.813 (k=4) → 0.840 (k=8)**, while the combined all-features model is already strong at k=2 (0.754) because the trace/shape features need only one run. Practically, k=4 captures most of the agreement signal; beyond that returns diminish. This is the compute knob a deployer trades against coverage.

**Robustness.** Two checks. (i) *Ship policy*: shipping the medoid patch instead of seed 0 leaves within-CV ranking similar for most families (all 0.733, shape+trace 0.748, trace 0.740) but lowers the agreement family to 0.733 — expected, since the medoid policy itself consumes part of the consensus signal; agreement is strongest when one sample is shipped (0.840). (ii) *Cross-repo generalization*: under leave-one-repo-out (train on three of the four repos — Textualize/rich, encode/httpx, fastapi/fastapi, psf/requests — test on the fourth), AUROC falls to 0.71–0.73 (agreement 0.715 [0.574, 0.851]; all 0.726) — still above chance but below within-distribution CV, and the deployable gate degrades (coverage ≤0.06). The signal transfers across repositories in rank but a precise cross-repo gate does not, reinforcing the deferral framing.

So the detector is a usable gate at a moderate precision target and a strong *ranking*, but not a high-precision autonomous acceptor.

## 5.1 Adversarial analysis: consistency is not correctness

The agreement signal rests on an assumption an adversary (or simply a systematically-wrong model) can violate: that patches the agent repeats across seeds are more likely correct. We test it directly on patch *text*. Normalizing each patch to its added/removed content and clustering the eight per-task patches, 14 tasks have a modal patch shared by ≥4 of 8 seeds. **Only 5 of those 14 ever resolve — 9 are consistent-but-never-correct**: the agent confidently converges on a broken fix. The extreme case is `fastapi_14953`, where **all eight seeds emit the identical patch and it is wrong on all eight** — maximal self-consistency, zero correctness.

Agreement is still informative on average (solvable tasks have a mean modal-cluster size of 2.8 seeds versus 1.5 for never-resolved), but it is a **necessary, not sufficient**, condition: high cross-seed agreement does not certify a patch. This is the concrete failure mode a pure agreement gate would suffer, and it is why our deployable recommendation pairs agreement with the project's existing (non-hidden) tests and treats the signal as grounds to *abstain on disagreement* rather than to *accept on agreement*. It mirrors the "consensus error" and imperfect-verifier-floor results in the literature.

## 5.2 Why E4B fails: a failure-mode taxonomy

Categorizing the 116 seed-0 failures by dominant mode (trace + patch heuristics; localization is inferred, not gold-labeled): **tested-but-unresolved 55.2%, never-tested 18.1%, broken-edits 9.5%, empty-patch 8.6%, budget/context 6.9%, wrong-file 0.9%, scattershot 0.9%.** The split is stable across all eight seeds (the two verification modes together ~65–70% every seed). Crucially, no failed task ignored a clean red test: of the 64 tested-but-unresolved cases, 21 passed a too-weak local test and 43 ran a test command that errored without an exit code; 62/64 submitted after a test ran. This is a **verification gap, not laziness**. Resolved runs differ sharply: failed runs flail longer (median 9.5 vs 6 tool calls, 12 vs 8 LLM calls, 4 vs 2 files read), verify less (ran a test after the last edit 63% vs 82%), and submit larger patches (739 vs 553 bytes). ~3 of 4 failures are verification failures the trace records directly, so "ran a test after the last edit and it cleanly passed" is exactly the unlabeled feature the abstention head reads — the taxonomy explains *why* the behavioral signal has power.

## 6. Limitations

With only 11 positives the confidence intervals are wide (agreement AUROC [0.745, 0.924]); we report CIs everywhere and never a bare point. Results are for one quantized model on one 127-task SWE-bench-style harness and may not transfer. Abstaining lowers coverage (fixes fewer bugs). The gate can in principle be gamed: a wrong patch that leaves an innocent trace, or a consistently-wrong agent whose agreement is high, will fool it (our adversarial analysis quantifies this). We already found and excluded two degenerate tasks (§3) whose hidden test passes with an empty patch; we flag that other such artifacts may remain in SWE-bench-style harnesses and recommend an empty-patch audit before scoring.

## 6.1 Small + large complementarity (and an honest negative on compute saving)

We also ran the 12B model on the same 127 tasks (one seed; under the same budget it is ~2.4× slower, 97 s/task, and errored on 94 tasks that exceeded the time budget, so its 17 resolves are a lower bound). E4B (seed 0) resolves 11, the 12B resolves 17 — but they solve **different** bugs: 9 tasks only the 12B solves, 9 only E4B's 8-seed union solves, 8 both. Their **union is 22/127, above either alone**. Small and large agents are complementary.

A compute-saving cascade, however, does **not** materialize at this scale. Because E4B can confidently accept almost nothing (its high-precision gate has ≈0 coverage, §5), routing its non-accepted tasks to the 12B defers ~116/127 tasks; the cascade then costs ~132% of simply running the 12B on everything. We report this plainly: at E4B's capability the deferral signal buys **coverage/complementarity, not compute savings**. Compute savings would require either a stronger small model (higher confident-accept rate) or an asymmetric budget; we leave that to future work.

## 7. Conclusion

Behavioral and cross-seed-agreement signals carry real information about whether a small local code agent's patch is correct (agreement AUROC 0.840), but at realistic data scales they make a *ranking* signal, not an autonomous high-precision accept gate, and — because the small model confidently keeps so little — they do not yield a compute-saving cascade. What they do yield is a measured abstention curve and a small/large **complementarity** result (union 22 > max 17). We release the 1,032-run labeled dataset (two degenerate tasks flagged) so the community can study the signal with more positives and sharper gates. We are explicit about the main threat to these conclusions: with 11–18 positives on a 129-task harness, every estimate carries wide intervals, and the first priority for follow-up is simply more labeled tasks.

## Resource release

`features8.jsonl` (1,032 task×seed rows; `features8_clean.jsonl` = the 1,016 used for analysis), non-leaky features + labels, the per-seed traces/patches, and the analysis scripts (`signals.py`, `risk_coverage.py`) are released under Apache-2.0.

## References

*Selective prediction.* Chow (1970), On optimum recognition error and reject tradeoff, IEEE Trans. Info. Theory (DOI 10.1109/TIT.1970.1054406). El-Yaniv & Wiener (2010), On the foundations of noise-free selective classification, JMLR 11. Geifman & El-Yaniv (2017), Selective Classification for Deep Neural Networks, NeurIPS, arXiv:1705.08500. Geifman & El-Yaniv (2019), SelectiveNet: A Deep Neural Network with an Integrated Reject Option, ICML, arXiv:1901.09192. Geifman, Uziel & El-Yaniv (2019), Bias-Reduced Uncertainty Estimation for Deep Neural Classifiers, ICLR, arXiv:1805.08206 (E-AURC).

*Confidence & calibration.* Tian et al. (2023), Just Ask for Calibration, EMNLP, arXiv:2305.14975. Xiong et al. (2024), Can LLMs Express Their Uncertainty?, ICLR, arXiv:2306.13063. Kuhn et al. (2023), Semantic Uncertainty, ICLR, arXiv:2302.09664. Farquhar et al. (2024), Detecting hallucinations using semantic entropy, Nature 630:625–630 (DOI 10.1038/s41586-024-07421-0). Spiess et al. (2024), Calibration and Correctness of Language Models for Code, arXiv:2402.02047 (ICSE 2025).

*Self-consistency.* Wang et al. (2022), Self-Consistency Improves Chain of Thought Reasoning, ICLR 2023, arXiv:2203.11171. Chen et al. (2022), CodeT: Code Generation with Generated Tests, arXiv:2207.10397. Li et al. (2022), Competition-Level Code Generation with AlphaCode, Science 378:1092–1097.

*Verifiers & resampling limits.* Cobbe et al. (2021), Training Verifiers to Solve Math Word Problems, arXiv:2110.14168. Lightman et al. (2023), Let's Verify Step by Step, arXiv:2305.20050. Ni et al. (2023), LEVER: Learning to Verify Language-to-Code Generation with Execution, arXiv:2302.08468. Zhang et al. (2022), Coder Reviewer Reranking for Code Generation, arXiv:2211.16490. Stroebl et al. (2024), The Limits of Inference Scaling Through Resampling, arXiv:2411.17501.

*SWE agents & benchmarks.* Jimenez et al. (2024), SWE-bench: Can Language Models Resolve Real-World GitHub Issues?, ICLR, arXiv:2310.06770. Yang et al. (2024), SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering, NeurIPS, arXiv:2405.15793. Xia et al. (2024), Agentless: Demystifying LLM-based Software Engineering Agents, arXiv:2407.01489. Aleithan et al. (2024), SWE-Bench+: Enhanced Coding Benchmark for LLMs, arXiv:2410.06992.

*Closest prior art.* Shi et al., EarlyEval: Cheaper Agent Evaluation via Early Outcome Prediction, arXiv:2609.02783. Grotov & Malykh, How to Speculate about Uncertainty in Agentic Coding? A Draft-Model Gate Method, arXiv:2609.05274. Gloaguen et al., Coding Agents Don't Know When to Act, arXiv:2605.07769. Mathews & Nagappan, Is Your Automated Software Engineer Trustworthy? (BouncerBench), arXiv:2506.17812. Cambronero et al., Abstain and Validate: A Dual-LLM Policy for Reducing Noise in Agentic Program Repair, arXiv:2510.03217. Zhou et al., Task Abstention for Large Language Models in Code Generation, arXiv:2605.17029.
