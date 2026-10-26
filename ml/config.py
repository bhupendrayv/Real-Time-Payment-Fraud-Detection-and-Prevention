"""Central configuration for the FraudShield ML pipeline.

All magic numbers, paths, and hyperparameters live here.
No other file should hard-code these values.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
ARTIFACTS_DIR: Path = PROJECT_ROOT / "ml" / "artifacts"
DOCS_DIR: Path = PROJECT_ROOT / "docs"

RAW_DATA_PATH: Path = DATA_DIR / "creditcard.csv"

# ── Reproducibility ───────────────────────────────────────────────────
SEED: int = 42

# ── Time-based split ratios ───────────────────────────────────────────
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15  # = 1 - TRAIN_RATIO - VAL_RATIO

# ── Feature columns (ULB Credit Card dataset) ─────────────────────────
PCA_FEATURES: list[str] = [f"V{i}" for i in range(1, 29)]  # V1–V28
OTHER_FEATURES: list[str] = ["Time", "Amount"]
ALL_FEATURES: list[str] = OTHER_FEATURES + PCA_FEATURES
TARGET: str = "Class"

# ── Drift Monitoring Configuration ────────────────────────────────────
REFERENCE_STATS_PATH: Path = ARTIFACTS_DIR / "reference_stats.json"
DRIFT_FEATURES: list[str] = PCA_FEATURES + ["Amount"]  # Time excluded

# PSI Thresholds
PSI_OK_THRESHOLD: float = 0.10
PSI_WARNING_THRESHOLD: float = 0.25

# Rolling Window & Evaluation Defaults
DRIFT_WINDOW_SIZE: int = 500
DRIFT_MIN_ROWS: int = 200
DRIFT_CACHE_TTL_SECONDS: int = 30
