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


def _extract_float_score(text: str) -> float:
    """Extract a numeric score from judge text; map yes/no if present.

    Returns a float in [0, 1]. If the first number found is in [0, 10],
    it is scaled to [0, 1] when > 1. Missing/invalid -> 0.0.
    """
    t = (text or "").strip().lower()
    if "yes" in t and "no" not in t and not re.search(r"\d", t):
        return 1.0
    if "no" in t and "yes" not in t and not re.search(r"\d", t):
        return 0.0
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", t)
    if not m:
        return 0.0
    try:
        val = float(m.group(0))
    except Exception:
        return 0.0
    # Normalize likely 0-10 scores to 0-1
    if val > 1.0:
        if val <= 10.0:
            val = val / 10.0
        else:
            # Out-of-range; clamp
            val = 1.0
    return max(0.0, min(1.0, val))


def _make_llm_judge_reward(judge_obj: vf.Rubric) -> Any:
    """Create an async reward function that calls a JudgeRubric.judge and returns float.

    The underlying JudgeRubric.judge returns a string; we parse to [0,1].
    """

    async def _judge_reward(prompt: List[Dict[str, str]] | str,
                            completion: List[Dict[str, str]] | str,
                            answer: str,
                            state: Dict[str, Any],
                            **kwargs: Any) -> float:
        # Access the coroutine method `judge` on the provided judge object.
        try:
            resp = await judge_obj.judge(
                prompt=prompt,
                completion=completion,
                answer=answer,
                state=state,
            )
        except Exception as e:
            return 0.0
        return _extract_float_score(str(resp))

    _judge_reward.__name__ = "llm_judge_reward"
    return _judge_reward


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
    eval_recipe: Optional[str] = None,
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

    # Build the system prompt string once. We will embed it into each example's
    # message list to satisfy verifiers' MultiTurnEnv rollout interface, which
    # expects `prompt` to be a list of role/content messages.
    allowed_tokens = _load_allowed_citation_tokens()
    allowed_tokens_str = ", ".join(allowed_tokens) if allowed_tokens else "[SR11-7], [OCC-Handbook]"
    system_prompt_str = (
        "Answer as a bank supervision analyst. Always use <think>..</think>,"
        " then <answer>..</answer>, then <citations>..</citations>. Only cite\n"
        f"approved tokens: {allowed_tokens_str}."
    )

    prepared_samples = []
    for s in samples:
        user_prompt = s.get("prompt", "")
        messages = [
            {"role": "system", "content": system_prompt_str},
            {"role": "user", "content": user_prompt},
        ]
        new_s = dict(s)
        new_s["prompt"] = messages
        prepared_samples.append(new_s)

    dataset = Dataset.from_list(prepared_samples)

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
        parser=parser,
    )

    # Variant: light compliance rubric (for hybrid judge-heavy eval)
    light_compliance = vf.Rubric(
        funcs=[parser.get_format_reward_func()],
        weights=[0.1],
        parser=parser,
    )

    # 4) Judge rubric: uses an LLM to check semantic correctness
    judge_prompt = (
        "You are grading a short bank supervision answer against the provided\n"
        "canonical answer and info. Score the response’s factual correctness,\n"
        "coverage of key points, and avoidance of unsupported claims. Grounding\n"
        "sources: SR 11-7 and the OCC Model Risk Management Comptroller’s Handbook.\n\n"
        "Question:\n{question}\n\n"
        "Canonical answer:\n{answer}\n\n"
        "Model answer:\n{response}\n\n"
        "Output format:\n"
        "<score in [0.0, 1.0]>\n"
        "<one-sentence justification>\n\n"
        "Deduct for statements not supported by SR 11-7/OCC MRM, missing required\n"
        "structure (tags), or irrelevant citations."
    )
    judge = vf.JudgeRubric(parser=parser, judge_model=(judge_model_name or "gpt-4.1-nano"), judge_prompt=judge_prompt)
    judge_reward_func = _make_llm_judge_reward(judge)

    # 5) Assemble environment
    # Choose scoring recipe
    rubrics: List[vf.Rubric]
    recipe = (eval_recipe or "deterministic").lower()
    if recipe == "judge_only":
        # Judge-only holistic scoring
        rubrics = [vf.Rubric(funcs=[judge_reward_func], weights=[judge_weight], parser=parser)]
    elif recipe == "hybrid":
        # Slight compliance presence + dominant judge
        j = vf.Rubric(funcs=[judge_reward_func], weights=[judge_weight], parser=parser)
        rubrics = [light_compliance, j]
    else:
        # Deterministic by default; optionally include judge if requested
        rubrics = [base_rubric]
        if use_judge:
            j = vf.Rubric(funcs=[judge_reward_func], weights=[judge_weight], parser=parser)
            rubrics.append(j)

    env = vf.SingleTurnEnv(
        dataset=dataset,
        parser=parser,
        rubric=vf.RubricGroup(rubrics),
        system_prompt=None,  # embedded directly in each example's messages
        **kwargs,
    )
    return env
