"""Minimal model risk management environment for verifiers.

This module exposes a `load_environment` function that constructs a
`verifiers` environment for answering short questions about model risk
management.  The environment uses a single‑turn protocol and rewards
agents for adhering to a strict XML format, citing required sources,
maintaining conciseness, and providing semantically correct answers.

The dataset is stored in `data/dataset.jsonl` and contains question–answer
pairs drawn from two public documents: SR 11‑7 (Supervisory Guidance on
Model Risk Management) and the OCC’s *Model Risk Management* handbook.

Usage:

    import vf_mrm_mini
    env = vf_mrm_mini.load_environment()

You can then evaluate or train a policy using `verifiers` tooling.

"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import re

import verifiers as vf
from datasets import Dataset

__all__ = ["load_environment"]

# Precompiled patterns for parsing
_CIT_PAT = re.compile(r"<citations>(.*?)</citations>", re.IGNORECASE | re.DOTALL)
_ANS_PAT = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)

def _citation_reward(completion: List[Dict[str, Any]], info: Dict[str, Any], strict: bool = True, **_: Any) -> float:
    """Reward based only on tokens inside the <citations> block.

    Computes fraction of required tokens present; in strict mode, lightly
    penalizes extra tokens to discourage spraying.
    """
    text = completion[-1]["content"] if completion else ""
    m = _CIT_PAT.search(text)
    block = m.group(1) if m else ""
    required = info.get("required_citations", []) or []
    # If no requirements and no citations present, return 1.0 (nothing to do)
    if not required and not block:
        return 1.0
    hits = sum(1 for tok in required if tok.lower() in block.lower())
    score = hits / max(1, len(required))
    if strict:
        found = set(re.findall(r"\[[^\]]+\]", block))
        extra = len(found - set(required))
        score = max(0.0, score - 0.1 * extra)
    return score


def _length_reward(completion: List[Dict[str, Any]], **_: Any) -> float:
    """Gentle length reward based on <answer> content only.

    Target 80–180 words. Penalize shorter than 60 and longer than 220.
    """
    text = completion[-1]["content"] if completion else ""
    m = _ANS_PAT.search(text)
    ans = (m.group(1) if m else text) or ""
    words = len(re.findall(r"\w+", ans))
    if words < 60:
        return max(0.0, (words - 20) / 40.0)
    if words > 220:
        return max(0.0, (280.0 - words) / 60.0)
    return 1.0


def _load_allowed_citation_tokens() -> List[str]:
    """Load the set of allowed citation tokens from sources.json.

    Returns:
        A list of strings like ["[SR11-7]", "[OCC-Handbook]"]
    """
    sources_path = Path(__file__).resolve().parent / "data" / "sources.json"
    try:
        with open(sources_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return sorted(list(data.keys()))
    except Exception:
        # Fallback to an empty list if sources cannot be read.
        return []


def _allowed_citations_only_reward(completion: List[Dict[str, Any]], **_: Any) -> float:
    """Reward if only approved tokens are cited in <citations>.

    Parses the <citations> field from the final message and checks that all
    bracketed tokens (e.g., "[SR11-7]") are drawn from the allowed set loaded
    from data/sources.json. If no citations are present, the score is 0.0; if
    the tag is present and all tokens are allowed, the score is 1.0; otherwise
    0.0.
    """
    text = completion[-1]["content"] if completion else ""
    m = re.search(r"<citations>(.*?)</citations>", text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return 0.0
    cited = re.findall(r"\[[^\]]+\]", m.group(1))
    if not cited:
        return 0.0
    allowed = set(_load_allowed_citation_tokens())
    return 1.0 if set(cited).issubset(allowed) else 0.0


def _load_samples(split: str | None = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load question–answer pairs from the packaged JSONL file.

    The dataset is stored in `data/dataset.jsonl` relative to this file.  Each
    line of the file is a JSON object with keys `prompt`, `answer`, and
    `info`.  This helper reads the file and optionally partitions it into
    train/dev/test splits by simple slicing.  The splits use an 80/10/10
    proportion based on the original ordering of examples.  You can also
    limit the number of examples returned.

    Args:
        split: One of {"train", "dev", "test", None}.  If None, all examples
            are returned.
        limit: An optional integer limiting the number of examples returned.

    Returns:
        A list of sample dictionaries.
    """
    data_path = Path(__file__).resolve().parent / "data" / "dataset.jsonl"
    with open(data_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    # Partition into train/dev/test using 80/10/10 split if requested.
    if split in {"train", "dev", "test"}:
        n = len(samples)
        n_train = int(n * 0.8)
        n_dev = int(n * 0.1)
        if split == "train":
            samples = samples[:n_train]
        elif split == "dev":
            samples = samples[n_train:n_train + n_dev]
        else:  # test
            samples = samples[n_train + n_dev:]
    if limit is not None:
        samples = samples[:limit]
    return samples


def load_environment(
    split: str | None = None,
    limit: Optional[int] = None,
    judge_model_name: Optional[str] = None,
    use_judge: bool = True,
    judge_weight: float = 1.0,
    **kwargs: Any
) -> vf.Environment:
    """Construct and return a `verifiers.SingleTurnEnv` for model risk management.

    Parameters
    ----------
    split : str or None, optional
        If provided, selects a partition of the dataset.  One of
        {"train", "dev", "test", None}.  Defaults to None, meaning all
        examples.
    limit : int or None, optional
        If provided, limits the number of examples returned.  Useful for
        debugging.
    judge_model_name : str or None, optional
        The name of the judge model to use in the `JudgeRubric`.  If
        unspecified, the default model configured in `verifiers` will be
        used.
    **kwargs : Any
        Additional keyword arguments forwarded to the `SingleTurnEnv` constructor.

    Returns
    -------
    vf.Environment
        A fully configured environment ready for evaluation or training.
    """
    # 1) Load the dataset
    samples = _load_samples(split=split, limit=limit)
    dataset = Dataset.from_list(samples)

    # 2) Parser enforces the <think>, <answer>, <citations> tags
    parser = vf.XMLParser(fields=["think", "answer", "citations"], answer_field="answer")

    # 3) Base rubric: format, citations (required + allowed-only), and length
    base_rubric = vf.Rubric(
        funcs=[
            parser.get_format_reward_func(),
            _citation_reward,
            _allowed_citations_only_reward,
            _length_reward,
        ],
        weights=[0.2, 0.5, 0.2, 0.1],
    )

    # 4) Judge rubric: uses an LLM to check semantic correctness
    judge_prompt = (
        "You are grading a short bank supervision answer against the provided\n"
        "canonical answer and info. Score between 0.0 and 1.0.\n"
        "Ground truth sources are limited to SR 11-7 and the OCC Model Risk\n"
        "Management Comptroller’s Handbook. Give full credit only if the answer\n"
        "is factually correct, captures the key points of the canonical answer,\n"
        "and avoids unsupported claims. Treat any statements not grounded in\n"
        "the cited sources as errors. If the required XML tags are missing or\n"
        "citations are irrelevant/absent, reduce the score. Return only a\n"
        "floating point score and a brief justification."
    )
    judge = vf.JudgeRubric(judge_prompt=judge_prompt, model_name=judge_model_name)

    # 5) Assemble environment
    rubrics: List[vf.Rubric] = [base_rubric]
    if use_judge:
        # Apply the judge as an additional rubric with configurable weight.
        if judge_weight != 1.0:
            judge = vf.Rubric(funcs=[judge], weights=[judge_weight])  # lightweight wrapper
        rubrics.append(judge)

    allowed_tokens = _load_allowed_citation_tokens()
    allowed_tokens_str = ", ".join(allowed_tokens) if allowed_tokens else "[SR11-7], [OCC-Handbook]"

    env = vf.SingleTurnEnv(
        dataset=dataset,
        parser=parser,
        rubric=vf.RubricGroup(rubrics),
        system_prompt=(
            "Answer as a bank supervision analyst. Always use <think>..</think>,"
            " then <answer>..</answer>, then <citations>..</citations>. Only cite\n"
            f"approved tokens: {allowed_tokens_str}."
        ),
        **kwargs,
    )
    return env
