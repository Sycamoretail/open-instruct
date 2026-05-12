"""Compatibility shim for scholar evaluator utilities.

The canonical implementation lives in `open_instruct.scholar_evaluator`.
Reward code should import that package module directly instead of depending on
this repository-root file.
"""

from open_instruct.scholar_evaluator import *  # noqa: F401,F403
