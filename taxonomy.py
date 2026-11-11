#!/usr/bin/env python3
"""Failure-mode taxonomy for the Gemma 4 E4B SWE-agent paper.

Reads the on-disk SWE-agent run (default seed 0) and assigns each non-resolved
task a single dominant failure mode from trace + patch signals. Real measured
numbers only, no network, no GPU.

Run: ~/.venvs/ntl/bin/python3 -I taxonomy.py
"""
import json
import os
import re
import glob
import statistics as st
from collections import Counter, defaultdict

RUN = os.path.expanduser("~/gemma-abstain/runs/ss/results_s0")
EXCLUDE = {"fastapi_14077", "fastapi_14851"}  # degenerate tasks

TEST_RE = re.compile(r"\b(pytest|py\.test|unittest|tox|nosetests)\b")


def load_results(run):
    rows = {}
    with open(os.path.join(run, "task_results.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            rows[r["instance_id"]] = r
    return rows


def parse_obs(step):
    obs = step.get("observation") or {}
    c = obs.get("content")
    if isinstance(c, str):
        try:
            return json.loads(c)
        except Exception:
            return {}
    return c or {}


def trace_signals(path):
    """Extract behavioral signals from one SessionTrace."""
    d = json.load(open(path))
    sig = dict(
        n_reads=0, n_edit_ok=0, n_edit_err=0, n_write_ok=0, n_write_err=0,
        n_cmd=0, n_cmd_err=0, n_submits=0, n_tool_err=0, n_search=0,
        n_test_runs=0, test_exit_codes=[], budget_warn=0,
        last_edit_idx=-1, last_test_idx=-1, n_tool_calls=0,
    )
    steps = d.get("steps", [])
    for i, s in enumerate(steps):
        for tc in (s.get("tool_calls") or []):
            fn = tc.get("function_name")
            cj = parse_obs(s)
            status = cj.get("status")
            sig["n_tool_calls"] += 1
            if status == "error":
                sig["n_tool_err"] += 1
            if fn == "read_file":
                sig["n_reads"] += 1
            elif fn == "search_similar_code":
                sig["n_search"] += 1
            elif fn == "edit_file":
                if status == "ok":
                    sig["n_edit_ok"] += 1
                    sig["last_edit_idx"] = i
                elif status == "error":
                    sig["n_edit_err"] += 1
            elif fn == "write_file":
                if status == "ok":
                    sig["n_write_ok"] += 1
                    sig["last_edit_idx"] = i
                elif status == "error":
                    sig["n_write_err"] += 1
            elif fn == "run_command":
                sig["n_cmd"] += 1
                if cj.get("budget_warning"):
                    sig["budget_warn"] += 1
                if status == "error":
                    sig["n_cmd_err"] += 1
                cmd = (tc.get("arguments") or {}).get("command", "") or ""
                if TEST_RE.search(cmd):
                    sig["n_test_runs"] += 1
                    sig["last_test_idx"] = i
                    ec = cj.get("exit_code")
                    try:
                        sig["test_exit_codes"].append(int(ec))
                    except (TypeError, ValueError):
                        pass
            elif fn == "submit_patch":
                sig["n_submits"] += 1
    sig["total_steps"] = (d.get("final_metrics") or {}).get("total_steps")
    sig["ran_tests"] = sig["n_test_runs"] > 0
    sig["ran_tests_after_last_edit"] = (
        sig["last_edit_idx"] >= 0 and sig["last_test_idx"] > sig["last_edit_idx"]
    )
    sig["n_applied_edits"] = sig["n_edit_ok"] + sig["n_write_ok"]
    return sig


def patch_stats(run, iid):
    p = os.path.join(run, "patches", f"{iid}.patch")
    if not os.path.exists(p):
        return dict(exists=False, size=0, files=[], n_files=0, n_hunks=0,
                    touches_tests=False, touches_nontest=False)
    txt = open(p, errors="ignore").read()
    files = re.findall(r"^diff --git a/(\S+) b/\S+", txt, re.M)
    if not files:
        files = re.findall(r"^\+\+\+ b/(\S+)", txt, re.M)
    n_hunks = len(re.findall(r"^@@ ", txt, re.M))

    def is_test(fp):
        return ("test" in fp.lower()) or fp.lower().startswith("tests/")

    touches_tests = any(is_test(f) for f in files)
    touches_nontest = any(not is_test(f) for f in files)
    return dict(exists=True, size=len(txt.strip()), files=files,
                n_files=len(files), n_hunks=n_hunks,
                touches_tests=touches_tests, touches_nontest=touches_nontest)


def classify(r, sig, ps):
    """Assign one dominant failure mode. Priority order top-down."""
    err = r.get("error") or ""
    # (b) ran out of budget / context (crashed before finishing)
    if "ContextWindowExceeded" in err or "maximum context length" in err:
        return "budget_context"
    if "tool call budget" in err:
        return "budget_context"
    # (a) empty / no patch
    if (not ps["exists"]) or ps["size"] == 0 or r.get("agent_patch_size", 0) == 0 \
            or sig["n_applied_edits"] == 0:
        return "empty_patch"
    # (f) huge / scattershot patch (flailing tail)
    if ps["n_files"] >= 3 or ps["n_hunks"] >= 8:
        return "scattershot"
    # (g) non-applying / broken edits dominate
    if sig["n_edit_err"] > sig["n_applied_edits"] and sig["n_edit_err"] >= 3:
        return "broken_edits"
    # (c) wrong-file / localization miss (inferred): patch edits tests only
    if ps["touches_tests"] and not ps["touches_nontest"]:
        return "wrong_file"
    # (d) ran tests, submitted, gold/hidden test still red. Honest label:
    # the agent never saw a clean red exit it ignored -- its local test was
    # green-but-insufficient or the test command itself errored out.
    if sig["ran_tests"]:
        return "tested_but_unresolved"
    # (e) never ran tests before submitting
    return "never_tested"


def main():
    rows = load_results(RUN)
    rows = {k: v for k, v in rows.items() if k not in EXCLUDE}
    resolved, failed = [], []
    per_task = {}
    for iid, r in rows.items():
        tp = os.path.join(RUN, "traces", f"trace_{iid}.json")
        sig = trace_signals(tp) if os.path.exists(tp) else {}
        ps = patch_stats(RUN, iid)
        rec = dict(iid=iid, repo=r["repo"], resolved=r["resolved"],
                   tool_calls=r["tool_calls"], total_llm_calls=r["total_llm_calls"],
                   patch_size=r["agent_patch_size"], test_exit_code=r["test_exit_code"],
                   error=r.get("error"), sig=sig, patch=ps)
        per_task[iid] = rec
        (resolved if r["resolved"] else failed).append(rec)

    # classify failures
    modes = Counter()
    mode_examples = defaultdict(list)
    for rec in failed:
        m = classify(rows[rec["iid"]], rec["sig"], rec["patch"])
        rec["mode"] = m
        modes[m] += 1
        mode_examples[m].append(rec)

    nF = len(failed)
    print(f"RUN={RUN}")
    print(f"tasks(after exclude)={len(rows)}  resolved={len(resolved)}  failed={nF}")
    print("\n=== FAILURE-MODE DISTRIBUTION (failed tasks) ===")
    order = ["empty_patch", "budget_context", "scattershot", "broken_edits",
             "wrong_file", "tested_but_unresolved", "never_tested"]
    for m in order:
        c = modes.get(m, 0)
        if c:
            ex = ", ".join(e["iid"] for e in mode_examples[m][:4])
            print(f"  {m:30s} {c:3d}  {100*c/nF:5.1f}%   e.g. {ex}")
    assert sum(modes.values()) == nF

    def med(xs):
        xs = [x for x in xs if x is not None]
        return st.median(xs) if xs else float("nan")

    print("\n=== RESOLVED vs FAILED behavioral contrasts (medians) ===")
    def grp(recs, key):
        return [r[key] for r in recs]
    print(f"  tool_calls          resolved={med(grp(resolved,'tool_calls'))}  failed={med(grp(failed,'tool_calls'))}")
    print(f"  total_llm_calls     resolved={med(grp(resolved,'total_llm_calls'))}  failed={med(grp(failed,'total_llm_calls'))}")
    print(f"  patch_size          resolved={med(grp(resolved,'patch_size'))}  failed={med(grp(failed,'patch_size'))}")
    for label, recs in [("resolved", resolved), ("failed", failed)]:
        n = len(recs)
        ran = sum(1 for r in recs if r["sig"].get("ran_tests"))
        rafter = sum(1 for r in recs if r["sig"].get("ran_tests_after_last_edit"))
        nf = med([r["sig"].get("n_applied_edits") for r in recs])
        reads = med([r["sig"].get("n_reads") for r in recs])
        nfiles = med([r["patch"]["n_files"] for r in recs])
        toolerr = med([r["sig"].get("n_tool_err") for r in recs])
        print(f"  [{label}] n={n}  ran_tests={ran}({100*ran/n:.0f}%)  "
              f"ran_tests_after_last_edit={rafter}({100*rafter/n:.0f}%)  "
              f"med_applied_edits={nf}  med_reads={reads}  med_patch_files={nfiles}  "
              f"med_tool_errors={toolerr}")

    # dump json for the markdown writer
    out = dict(run=RUN, n_tasks=len(rows), n_resolved=len(resolved), n_failed=nF,
               modes=dict(modes),
               examples={m: [e["iid"] for e in mode_examples[m][:5]] for m in modes},
               resolved_med=dict(
                   tool_calls=med(grp(resolved, 'tool_calls')),
                   llm=med(grp(resolved, 'total_llm_calls')),
                   patch=med(grp(resolved, 'patch_size')),
                   ran_tests=sum(1 for r in resolved if r['sig'].get('ran_tests')),
               ),
               failed_med=dict(
                   tool_calls=med(grp(failed, 'tool_calls')),
                   llm=med(grp(failed, 'total_llm_calls')),
                   patch=med(grp(failed, 'patch_size')),
                   ran_tests=sum(1 for r in failed if r['sig'].get('ran_tests')),
               ))
    with open(os.path.join(os.path.dirname(__file__) if "__file__" in dir() else ".",
                           "taxonomy_out.json"), "w") as f:
        json.dump(out, f, indent=2)
    return per_task, modes, mode_examples, resolved, failed


if __name__ == "__main__":
    main()
