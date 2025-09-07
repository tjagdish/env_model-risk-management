#!/usr/bin/env python
"""Annotate dataset.jsonl with info.tags and info.difficulty.

Heuristics:
- Tags inferred from keywords in the prompt question (governance, validation,
  inventory-docs, third-party, compliance, interest-rate-risk, credit-risk,
  reporting, general).
- Difficulty inferred from interrogatives: simple factual vs scenario/synthesis.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List


DATA = Path(__file__).resolve().parents[1] / "vf_mrm_mini" / "data" / "dataset.jsonl"


def infer_tags(text: str) -> List[str]:
    t = text.lower()
    tags: List[str] = []
    if any(k in t for k in ("board", "senior management", "governance", "policy", "policies")):
        tags.append("governance")
    if any(k in t for k in ("validation", "benchmark", "outcomes", "back-testing", "ongoing monitoring", "conceptual soundness", "sensitivity")):
        tags.append("validation")
    if any(k in t for k in ("inventory", "documentation", "document", "doc")):
        tags.append("inventory-docs")
    if any(k in t for k in ("third-party", "vendor")):
        tags.append("third-party")
    if any(k in t for k in ("aml", "bsa", "fair lending", "compliance")):
        tags.append("compliance")
    if any(k in t for k in ("interest rate risk", "irr")):
        tags.append("interest-rate-risk")
    if "credit risk" in t:
        tags.append("credit-risk")
    if any(k in t for k in ("report", "reporting", "minutes")):
        tags.append("reporting")
    if not tags:
        tags.append("general")
    # de-dup
    return sorted(set(tags))


def infer_difficulty(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ("scenario", "explain", "synthesize", "summarize", "connect", "how should")):
        return "medium"
    if any(k in t for k in ("why", "how", "differentiate")):
        return "medium"
    return "easy"


def main() -> None:
    lines = DATA.read_text(encoding="utf-8").splitlines()
    out_lines: List[str] = []
    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue
        rec = json.loads(line)
        prompt = rec.get("prompt", "")
        info = rec.setdefault("info", {})
        # Ensure required_citations exists and is list
        req = info.get("required_citations", [])
        if not isinstance(req, list):
            req = [req] if req else []
        info["required_citations"] = req
        # Add tags/difficulty if missing
        info.setdefault("tags", infer_tags(prompt))
        info.setdefault("difficulty", infer_difficulty(prompt))
        out_lines.append(json.dumps(rec, ensure_ascii=False))

    DATA.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"Annotated {len(out_lines)} items → {DATA}")


if __name__ == "__main__":
    main()

