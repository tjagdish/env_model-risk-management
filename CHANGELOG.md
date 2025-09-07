## v0.1.1

- Package restructure: move code to `vf_mrm_mini/__init__.py` and bundle data under `vf_mrm_mini/data`.
- Packaging: add `[tool.hatch.build]` include rules; `uv build` now ships data.
- Scoring: stricter citation parsing (only tokens inside `<citations>`), allowed‑only check from `sources.json`, gentle `<answer>`-only length reward; configurable judge toggle and weight.
- Data: add additional SR 11‑7 and OCC MRM items; keep bracketed tokens `[SR11-7]`, `[OCC-Handbook]`.
- Scripts: add `scripts/validate_dataset.py` for JSONL/schema checks.
- Docs: update README with packaging notes and usage; add submission checklist.

## v0.1.0

- Initial minimal environment with SR 11‑7 items, XML parser, citation + length rewards, and judge rubric.

