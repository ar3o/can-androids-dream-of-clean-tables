"""Hand masks of every tracked frame: the union of SAM 3's 'hand' masklets -> hand_video.npz.

The hands are the agent, never an object: their masks tell the tracker which object pixels are hidden or held, and give
the world model a hand position per frame.
"""

from __future__ import annotations

import numpy as np

from .files import AGENT_PHRASES, H_LO, W_LO, video_dir

HAND_MIN_PROB = 0.4


def hand_union(masklets: list[dict], n_t: int) -> np.ndarray:
    """(n_t, 270, 480) bool: any confident agent masklet."""
    hand = np.zeros((n_t, H_LO, W_LO), bool)
    for m in masklets:
        if m["phrase"] in AGENT_PHRASES:
            hand |= m["masks"] & (m["prob"][:, None, None] >= HAND_MIN_PROB)
    return hand


def save(video: str, frames: np.ndarray, hand: np.ndarray) -> None:
    np.savez_compressed(
        video_dir(video) / "hand_video.npz",
        frames=np.array(frames, np.int32),
        masks=np.packbits(hand),
        shape=np.array(hand.shape),
    )
