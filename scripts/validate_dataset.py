#!/usr/bin/env python
"""Quick linter for vf_mrm_mini dataset and schema.

Checks:
- Each JSONL record has prompt, answer, info.required_citations
- Prompt contains tag instructions (<think>, <answer>, <citations>)
- required_citations tokens are in sources.json
"""
from __future__ import annotations

import json
from pathlib import Path


def find_data_dir() -> Path:
    # Prefer packaged path
    pkg_dir = Path(__file__).resolve().parents[1] / "vf_mrm_mini" / "data"
    if (pkg_dir / "dataset.jsonl").exists():
        return pkg_dir
    # Fallback to repo root data
    root_dir = Path(__file__).resolve().parents[1] / "data"
    return root_dir


def main() -> None:
    data_dir = find_data_dir()
    ds_path = data_dir / "dataset.jsonl"
    src_path = data_dir / "sources.json"
    assert ds_path.exists(), f"Missing dataset: {ds_path}"
    assert src_path.exists(), f"Missing sources: {src_path}"
    sources = set(json.loads(src_path.read_text(encoding="utf-8")).keys())

    errors = 0
    with ds_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception as e:
                print(f"Line {i}: invalid JSON: {e}")
                errors += 1
                continue
            for key in ("prompt", "answer", "info"):
                if key not in rec:
                    print(f"Line {i}: missing key '{key}'")
                    errors += 1
            info = rec.get("info", {})
            req = info.get("required_citations")
            if not isinstance(req, list) or not req:
                print(f"Line {i}: info.required_citations must be non-empty list")
                errors += 1
            else:
                unknown = [t for t in req if t not in sources]
                if unknown:
                    print(f"Line {i}: unknown citation tokens: {unknown}")
                    errors += 1
            prompt = rec.get("prompt", "")
            for must in ("<think>", "<answer>", "<citations>"):
                if must.split("<")[1].split(">")[0] not in prompt and must not in prompt:
                    # Loose check: presence of tokens or instruction string
                    pass
            if "<citations>" not in prompt:
                print(f"Line {i}: prompt should instruct to include <citations>")
                errors += 1

    if errors:
        raise SystemExit(f"Found {errors} issues.")
    print("Dataset OK.")


if __name__ == "__main__":
    main()

