from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Sequence


# Project root (HAR_Projekt/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data folders (keep your structure)
DEFAULT_DATA_ROOT = PROJECT_ROOT / "ax6_cnn_project" / "data" / "raw"
TRIMMED_DATA_ROOT = PROJECT_ROOT / "ax6_cnn_project" / "data" / "trimmed"
ACTIVE_SECTIONS_DATA_ROOT = PROJECT_ROOT / "ax6_cnn_project" / "data" / "active_sections_tensors"


@dataclass
class Config:
    """Central configuration for dataset + model runs."""
    # Paths
    data_root: str = str(DEFAULT_DATA_ROOT)
    out_dir: str = "out"

    # Sensor IDs (left/right)
    right_id: str = "6032357"
    left_id: str = "6033722"

    # Sampling rate after resampling
    fs: int = 100

    # Sessions used in the project (12..21 by default)
    sessions: Tuple[int, ...] = tuple(range(12, 22))

    # Windowing
    win_sec: float = 1.0
    hop_sec: float = 0.5

    # Training
    test_size: float = 0.1
    random_state: int = 42
    batch_size: int = 32
    epochs: int = 40

    # Clap/trim parameters (kept for compatibility even if not used in active-sections pipeline)
    clap_search_sec: float = 8.0
    trim_after_clap_sec: float = 2.0
    drop_start_sec: float = 1.0
    drop_end_sec: float = 1.0

    # Idle filtering (optional)
    remove_idle: bool = True
    idle_energy_threshold: float = 0.08


def default_session_to_activity() -> Dict[int, str]:
    """Session → activity mapping used in the dataset."""
    return {
        12: "walk", 13: "walk",
        14: "jogging", 15: "jogging",
        16: "situp", 17: "situp",
        18: "jumping_jacks", 19: "jumping_jacks",
        20: "squat", 21: "squat",
    }


def session_to_subject(session: int) -> str:
    """Subject label used for grouping/stratification."""
    return "amy" if (session % 2 == 0) else "thasa"