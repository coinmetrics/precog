# Project Constants

# Wandb Constants
WANDB_PROJECT = "yumaai"

# Predictions Constants
EVALUATION_WINDOW_HOURS: int = 6
PREDICTION_FUTURE_HOURS: int = 1
PREDICTION_INTERVAL_MINUTES: int = 5

# Supported Assets
SUPPORTED_ASSETS = ["tao_bittensor", "btc", "eth"]
DEFAULT_ASSET = "btc"

# Task Weights - each asset has point and interval prediction tasks
# Weights should sum to 1.0 across all tasks
# Format: {asset: {"point": weight, "interval": weight}}
TASK_WEIGHTS = {
    "btc": {"point": 0.166, "interval": 0.166},
    "eth": {"point": 0.166, "interval": 0.166},
    "tao_bittensor": {"point": 0.166, "interval": 0.166},
}

# Cache limits to prevent memory leaks
MAX_CACHE_SIZE_MB = 100  # Maximum cache size in MB
MAX_CACHE_ROWS = 500000  # Maximum number of rows in cache
