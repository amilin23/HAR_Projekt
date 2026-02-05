from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]  # HAR_Projekt/
DEFAULT_DATA_ROOT = str(BASE_DIR / "ax6_cnn_project" / "data" / "raw")
TRIMMED_DATA_ROOT = str(BASE_DIR / "ax6_cnn_project" / "data" / "trimmed")
ACTIVE_SECTIONS_DATA_ROOT = str(BASE_DIR / "ax6_cnn_project" / "data" / "active_sections_tensors")

@dataclass
class Config:
    # Paths
    data_root: str = DEFAULT_DATA_ROOT
    out_dir: str = "out"

    # Sensor IDs
    right_id: str = "6032357"
    left_id: str = "6033722"

    # Sampling
    fs: int = 100

    # Sessions
    sessions: List[int] = None

    # Windowing
    win_sec: float = 2.0
    hop_sec: float = 1.0
    win_sec: float = 1.0

    # Clap handling
    clap_search_sec: float = 8.0
    trim_after_clap_sec: float = 2.0
    drop_start_sec: float = 1.0
    drop_end_sec: float = 1.0

    # Idle removal
    remove_idle: bool = True
    idle_energy_threshold: float = 0.08

    # Train
    test_size: float = 0.35
    random_state: int = 42
    batch_size: int = 64
    epochs: int = 40

def default_session_to_activity() -> Dict[int, str]:
    # Your session mapping
    return {
        12: "walk", 13: "walk",
        14: "jogging", 15: "jogging",
        16: "situp", 17: "situp",
        18: "jumping_jacks", 19: "jumping_jacks",
        20: "squat", 21: "squat",
    }

def session_to_subject(session: int) -> str:
    # Based on your note: even = Amy, odd = Thasa
    return "amy" if (session % 2 == 0) else "thasa"
