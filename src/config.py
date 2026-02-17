from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class Config:
    # Sampling / windowing
    fs: int = 100
    win_sec: float = 1.0
    hop_sec: float = 0.5

    # Training
    epochs: int = 40
    batch_size: int = 32
    random_state: int = 42
    test_size: float = 0.25  # fraction for TEST

    # Sessions used for training/eval (12..27 inclusive)
    sessions: Tuple[int, ...] = tuple(range(12, 28))

    # Output
    out_dir: str = "out"


def default_session_to_activity() -> Dict[int, str]:
    return {
        12: "walk", 13: "walk",
        14: "jogging", 15: "jogging",
        16: "situp", 17: "situp",
        18: "jumping_jacks", 19: "jumping_jacks",
        20: "squat", 21: "squat",

        22: "pumping", 23: "pumping",
        24: "jump_squat", 25: "jump_squat",
        26: "lunge", 27: "lunge",
    }


def session_to_subject(session: int) -> str:
    # Explicit mapping (not even/odd)
    subject_map = {
        12: "Amilin", 13: "Thasa",
        14: "Amilin", 15: "Thasa",
        16: "Amilin", 17: "Thasa",
        18: "Amilin", 19: "Thasa",
        20: "Amilin", 21: "Thasa",

        22: "Thasa", 23: "Amilin",
        24: "Thasa", 25: "Amilin",
        26: "Thasa", 27: "Amilin",
    }
    return subject_map.get(session, "unknown")