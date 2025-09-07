# vf_mrm_mini

Minimal RL and evaluation environment for model risk management (MRM). It trains and evaluates models on short questions grounded in SR 11‑7 and the OCC Model Risk Management Comptroller’s Handbook. The environment follows the `verifiers` framework and runs locally for deterministic RL and judge‑based evaluation.

## Task

Agents receive a short MRM question and must reply in this strict XML format:

```
<think>…free reasoning…</think>
<answer>…final answer, a few sentences…</answer>
<citations>[SR11-7]; [OCC-Handbook]</citations>
```

- Only `<answer>` is judged for semantic correctness.
- `<citations>` must contain bracket tokens defined in `vf_mrm_mini/data/sources.json`.

## Dataset

Path: `vf_mrm_mini/data/dataset.jsonl`. Each line is a JSON object:

- prompt: instruction + question (includes tag requirements)
- answer: concise canonical answer
- info: metadata (includes `required_citations`, plus `tags` and `difficulty`)

Covered sources:

1. [SR11-7] — Supervisory Guidance on Model Risk Management (April 4, 2011). Describes purpose/scope, defines “model,” and sets principles for development, validation, and governance. Notes both direct development costs and indirect costs from reliance on incorrect or misused models. Defines model components (input, processing, reporting) and emphasizes that models are simplified representations with quality measured by precision, accuracy, robustness, stability, etc.
2. [OCC-Handbook] — Model Risk Management, Comptroller’s Handbook (Version 1.0, August 2021). Provides examination procedures and practices. Describes sound governance (board/senior management oversight, policies/procedures, internal controls, inventory, documentation), the need for effective challenge, and roles of board and senior management.

The dataset includes factual, scenario‑based, and synthesis questions with the relevant required citation token.

## Sources

`vf_mrm_mini/data/sources.json` maps tokens to canonical details, e.g.:

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

## Installation

Requires Python 3.11+. Declared deps: `verifiers`, `datasets`.

```bash
uv pip install -e .
```

## Usage

```python
import vf_mrm_mini as mrm
env = mrm.load_environment(use_judge=False)  # deterministic, rule-based RL
```

CLI sanity eval (rule + judge if configured):

```bash
uv run vf-eval vf-mrm-mini -n 12 -r 2
```

## Scoring

- Format (0.2): XML tags present in order.
- Citations required (0.5): counts only tokens inside `<citations>`; extra tokens lightly penalized.
- Citations allowed‑only (0.2): `<citations>` must use whitelisted tokens.
- Length (0.1): gentle shaping on `<answer>` only.

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

Reasons: unapproved token; blanket annual rule not supported by SR 11‑7; vague answer.

## Development Notes

- Data and sources are bundled under `vf_mrm_mini/data`.
- Validate dataset: `python scripts/validate_dataset.py`
- Re‑annotate tags/difficulty after edits: `python scripts/annotate_dataset.py`

## Quick Start (Local)

- Install: `uv pip install -e .`
- Lint data: `python scripts/validate_dataset.py`
- Evaluate: `uv run vf-eval vf-mrm-mini -n 12 -r 2`
- Baseline: `python scripts/baseline_policy.py "What is model risk?"`

Generation settings (for your model/policy):
- Stop sequences: `</citations>`
- Max output tokens: ~256
- Temperature: 0.0–0.3; top_p: 1.0 (judge runs: temp=0)
- Always emit `<think>`, `<answer>`, `<citations>`; only cite `[SR11-7]`/`[OCC-Handbook]`

Judge (for evaluation): freeze model + params (temp 0, top_p 1) and cap concurrency; if no judge is configured, only deterministic scores are reported.

## Disclaimer

This environment benchmarks model risk management knowledge grounded in SR 11‑7 and the OCC Model Risk Management Comptroller’s Handbook. It is for educational and evaluation purposes only and does not constitute supervisory guidance or advice.
