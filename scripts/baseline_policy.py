#!/usr/bin/env python
"""Minimal format-only baseline policy for sanity checks.

Emits valid XML with a generic answer and cites [SR11-7] by default.
Use to calibrate rubric weights—expect high format/citation, low judge.
"""
from __future__ import annotations

import argparse


TEMPLATE = """
<think>Compose a short answer.</think>
<answer>{answer}</answer>
<citations>{citations}</citations>
""".strip()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("question", help="Question text (ignored by baseline)")
    p.add_argument("--answer", default="This is a generic baseline answer.")
    p.add_argument("--citations", default="[SR11-7]")
    args = p.parse_args()
    print(TEMPLATE.format(answer=args.answer, citations=args.citations))


if __name__ == "__main__":
    main()

