"""Minimal OpenCV stub for text-only training environments.

This repository's RL/vLLM path does not use OpenCV directly, but some spawned
processes may attempt to import ``cv2`` opportunistically. On machines missing
system GUI/X11 libs (e.g. ``libxcb.so.1``), importing the wheel can fail and
crash vLLM initialization. Placing this stub earlier on ``sys.path`` keeps the
text-only stack working without requiring system package changes.
"""

__all__: list[str] = []
__version__ = "0.0-stub"
