from __future__ import annotations

# Thin shim so `verifiers.load_environment('banking-mrm')` can import this
# package and find `load_environment`. The implementation remains in
# `vf_mrm_mini` to avoid renaming the internal module.

from vf_mrm_mini import load_environment  # re-export

__all__ = ["load_environment"]

