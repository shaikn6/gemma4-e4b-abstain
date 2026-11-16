# gemma4-e4b-abstain-traces
Labeled SWE-agent runs for selective patch acceptance. See DATASHEET.md.
- features8_clean.jsonl : 1016 rows (127 tasks x 8 E4B seeds), 37 non-leaky features + resolved label + repo.
- features_12b.jsonl    : 127 rows (Gemma 4 12B, single seed).
- signals.py / risk_coverage.py : reproduce the paper's numbers (same files as ../abstain/).
- Raw per-run traces/ and patches/ are not committed (too large); the feature files above are the released analysis inputs.
Apache-2.0.
