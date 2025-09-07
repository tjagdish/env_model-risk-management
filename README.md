# vf_mrm_mini

This repository contains a minimal reinforcement‐learning and evaluation environment
for **model risk management**.  It is designed to train language models to answer
short, bank‑supervision questions about model risk while citing authoritative
guidance.  The environment follows the conventions of the `verifiers` framework
and can be pushed to the Prime Intellect Environment Hub.

## Task

Models interacting with this environment are given a short question about
model risk management and must respond in a strict XML‑like format:

```
<think>…free reasoning…</think>
<answer>…final answer, a few sentences…</answer>
<citations>[SR11-7]; [OCC-Handbook]</citations>
```

Only the `<answer>` portion is scored for correctness by a judge model.  The
`<citations>` tag must contain bracketed tokens corresponding to the
authoritative sources defined in `vf_mrm_mini/data/sources.json` (see below).

## Dataset

The dataset lives in `vf_mrm_mini/data/dataset.jsonl`.  Each line is a JSON object with
three keys:

* **prompt** – a string instructing the model to answer using the required tags
  followed by the actual question.  Prompts take the general form
  `"Answer as a bank supervision analyst. Use <think>..</think>, <answer>..</answer>, and <citations>..</citations> with bracketed source tokens.\nQ: …"`.
* **answer** – the canonical ground‑truth answer against which model outputs
  are judged.  Answers are concise and cite the key facts from the source.
* **info** – a dictionary containing metadata.  At minimum it includes
  `required_citations`, a list of tokens that **must** appear in the
  `<citations>` tag.

The current dataset covers model risk concepts drawn from two public sources:

1. **[SR11-7]** – *Supervisory Guidance on Model Risk Management* (Board of
   Governors of the Federal Reserve System and Office of the Comptroller of
   the Currency, April&nbsp;4,&nbsp;2011).  This document describes the purpose and
   scope of model risk management, defines what constitutes a model, and
   outlines principles for development, validation, and governance.  It notes
   that banks incur both direct costs to build models and indirect costs from
   reliance on incorrect or misused models【76071430114180†L69-L76】, and
   emphasises that model risk management must encompass robust development,
   implementation, validation, and governance【76071430114180†L94-L106】.  A
   model is defined as having an information input component, a processing
   component, and a reporting component【76071430114180†L151-L162】, and the
   guidance stresses that models are simplified representations whose quality
   can be measured by precision, accuracy, robustness, and other metrics【76071430114180†L185-L206】.

2. **[OCC-Handbook]** – *Model Risk Management, Comptroller’s Handbook*
   (Office of the Comptroller of the Currency, Version&nbsp;1.0, August&nbsp;2021).
   This booklet provides examiners and banks with detailed examination
   procedures and best practices.  It describes sound model governance as
   encompassing board and management oversight, policies and procedures,
   internal controls, model inventory, and documentation【336363501890617†L1029-L1048】.
   It also emphasises the need for “effective challenge” by objective and
   informed personnel and outlines the roles of the board and senior
   management in establishing a bank‑wide model risk framework【336363501890617†L1134-L1153】.

The questions in `dataset.jsonl` cover factual, scenario‑based, and synthesis
topics drawn from these sources.  Each example includes the token for
whichever document it relies on (`[SR11-7]` or `[OCC-Handbook]`).

## Sources

The file `vf_mrm_mini/data/sources.json` maps citation tokens to their canonical
descriptions and URLs.  For example:

```json
{
  "[SR11-7]": {
    "title": "Supervisory Guidance on Model Risk Management",
    "url": "https://www.federalreserve.gov/supervisionreg/srletters/sr1107a1.pdf",
    "publisher": "Board of Governors of the Federal Reserve System and Office of the Comptroller of the Currency",
    "date": "2011-04-04"
  },
  "[OCC-Handbook]": {
    "title": "Model Risk Management, Comptroller’s Handbook",
    "url": "https://www.occ.treas.gov/publications-and-resources/publications/comptrollers-handbook/files/model-risk-management/pub-ch-model-risk.pdf",
    "publisher": "Office of the Comptroller of the Currency",
    "date": "2021-08-01"
  }
}
```

These tokens must be used verbatim inside the `<citations>` tag when
answering questions.  If a question’s `required_citations` list contains
`[SR11-7]`, the model is expected to cite that token.

## Installation

Before using this environment, ensure you have Python&nbsp;3.11+ installed.  The
repository defines a `pyproject.toml` that declares dependencies on
[`verifiers`](https://pypi.org/project/verifiers/) and
[`datasets`](https://pypi.org/project/datasets/).

You can install the environment locally for development and testing:

```bash
uv pip install -e .
```

To evaluate the environment on your machine, run:

```bash
uv run vf-eval vf-mrm-mini -n 10 -r 3
```

This command samples 10 examples and performs three rollouts per example.

## Usage

Models interact with the environment via the `verifiers` API.  The
`load_environment` function defined in `vf_mrm_mini.py` returns a
`SingleTurnEnv` instance.  The environment uses a `vf.XMLParser` to enforce
format compliance, a custom citation reward to check that required tokens are
present and that only approved tokens are cited, a gentle length reward, and
an optional `JudgeRubric` to assess semantic
correctness.  See the docstring in `vf_mrm_mini.py` for details.

Deterministic mode (no judge):

```python
import vf_mrm_mini as mrm
env = mrm.load_environment(use_judge=False)  # purely rule-based scoring
```
This mode is useful for reproducible local RL without any external model.

Scoring summary (default weights):
- Format: 0.2 — XML tag compliance.
- Citations (required): 0.5 — only counts tokens inside `<citations>`; extra tokens lightly penalized.
- Citations (allowed-only): 0.2 — all tokens in `<citations>` must be whitelisted (from `vf_mrm_mini/data/sources.json`).
- Length: 0.1 — gentle shaping on `<answer>` content only.

Good output example

```
<think>Consider SR 11-7’s definition and two causes.</think>
<answer>Model risk is the potential for adverse consequences from decisions based on incorrect or misused model outputs and reports. It arises when a model contains fundamental errors that yield inaccurate estimates for its design and intended use, and when otherwise sound models are used incorrectly or inappropriately, such as outside the environment for which they were designed.</answer>
<citations>[SR11-7]</citations>
```

Bad output example (will score poorly)

```
<think>Answer directly.</think>
<answer>Model risk happens when models are wrong. Validate yearly.</answer>
<citations>[Random-Source]</citations>
```
Reasons: cites an unapproved token; asserts a blanket annual requirement not supported by SR 11‑7; answers lack specificity.

## Versioning

The environment is versioned via the `version` field in `pyproject.toml`.  Any
changes to the dataset, rubric weights, or implementation should result in a
version bump before pushing to the Prime Intellect Hub.  After updating
version numbers, run:

```bash
prime env push
```

to publish a new version.  Previous versions will remain available for
reproducibility.

## Hub Submission Checklist

- Prime CLI: `uv tool install prime` then `prime login` (set username).
- Version bump in `pyproject.toml` if content changed.
- Local quick check: `uv run vf-eval vf-mrm-mini -n 10 -r 1`.
- From this directory: `prime env push` (add `--visibility PRIVATE` if needed).

## Packaging

Data is included in the wheel for out-of-the-box installs. Verify locally:

```bash
uv build  # produce a wheel including vf_mrm_mini/data/*
```

Optional dataset sanity check:

```bash
python scripts/validate_dataset.py
```

## Disclaimer

This environment benchmarks model risk management knowledge grounded in SR 11‑7 and the OCC Model Risk Management Comptroller’s Handbook. It is for educational and evaluation purposes only and does not constitute supervisory guidance or advice.

## Quick Start (Local)

- Prereqs: Python 3.11+, `uv` installed.
- Install editable: `uv pip install -e .`
- Lint data: `python scripts/validate_dataset.py`
- Optional: re‑annotate tags/difficulty if you edit prompts: `python scripts/annotate_dataset.py`

Evaluate (rule + judge if configured):

```bash
uv run vf-eval vf-mrm-mini -n 12 -r 2
```

- Baseline sanity (format-only):

```bash
python scripts/baseline_policy.py "What is model risk?"
```

Generation settings (for your model/policy):
- Stop sequences: `</citations>`
- Max output tokens: ~256
- Temperature: 0.0–0.3; Top‑p: 1.0 (judge runs: temp=0)
- Always emit the three tags in order; only cite `[SR11-7]` and/or `[OCC-Handbook]` in `<citations>`

Judge configuration (for evaluation):
- Freeze judge model and params (temp 0, top_p 1) for reproducibility.
- Set reasonable concurrency to avoid rate limits. If a judge is not configured, only deterministic rubric scores will be reported.
