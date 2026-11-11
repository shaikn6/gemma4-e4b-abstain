# gemma4-e4b-abstain-traces
Labeled SWE-agent runs for selective patch acceptance. See DATASHEET.md.
- features8_clean.jsonl : 1016 rows (127 tasks x 8 E4B seeds), 37 non-leaky features + resolved label + repo.
- features_12b.jsonl    : 127 rows (Gemma 4 12B, single seed).
- signals.py / risk_coverage.py : reproduce the paper's numbers.
- Raw traces/ and patches/ per run: ../../runs/ss/results_s0..s7 (E4B) and results_b0 (12B).
Apache-2.0.
