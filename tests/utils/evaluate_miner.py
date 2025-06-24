# =========================
# Imports
# =========================
import os
import logging
import pandas as pd
import numpy as np
from coinmetrics.api_client import CoinMetricsClient
from typing import Optional

# =========================
# Constants and Config
# =========================

# CSV Downloaded from precog.coinmetrics.io
CSV_PATH = "/workspaces/data-tools/2025-06-23T21-06_export.csv"

COLUMN_RENAME_MAP = {
    "CM Reference Rate at Eval Time": "RR_EvalTime",
    "Miner Hotkey": "miner_hotkey",
    "Prediction Time": "prediction_time",
    "Evaluation Time": "evaluation_time",
    "Point Forecast": "point_forecast",
    "Interval Lower Bound": "interval_lower",
    "Interval Upper Bound": "interval_upper",
    "Avg Reward": "avg_reward",
    "Miner UID": "uid",
}
BASE_REWARD_RATE = 0.9
EMA_ALPHA = 0.0095808525
EMA_WINDOW = 144

# =========================
# Logging Setup
# =========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# =========================
# Utility Functions
# =========================
def get_api_key() -> Optional[str]:
    """Get API key from environment variable. For Reference Rates > 7 days ago need paid key."""
    return os.environ.get("COINMETRICS_API_KEY")


def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """
    Load and clean the predictions data from CSV.
    Args:
        csv_path: Path to the CSV file.
    Returns:
        Cleaned DataFrame.
    """
    df = pd.read_csv(csv_path)
    df.rename(columns=COLUMN_RENAME_MAP, inplace=True)
    df = df.loc[~df.RR_EvalTime.isnull()]
    df["prediction_time"] = pd.to_datetime(df["prediction_time"])
    df["evaluation_time"] = pd.to_datetime(df["evaluation_time"])
    for col in ["interval_lower", "interval_upper", "point_forecast", "RR_EvalTime"]:
        df[col] = df[col].astype(str).str.replace("[$,]", "", regex=True).astype(float)
    return df


def rank_to_share(
    miner_df: pd.DataFrame,
    col_prefix: str,
    score_or_error: str,
    base: float = BASE_REWARD_RATE,
) -> pd.DataFrame:
    """
    Calculate share of reward based on rank.
    Args:
        miner_df: DataFrame with miner predictions.
        col_prefix: Prefix for the column (e.g., 'point' or 'interval').
        score_or_error: Suffix for the column (e.g., 'error' or 'score').
        base: Base for exponential decay.
    Returns:
        DataFrame with new share column.
    """
    miner_df[f"{col_prefix}_share"] = miner_df.groupby(
        ["prediction_time", f"{col_prefix}_{score_or_error}"]
    )[f"{col_prefix}_rank"].transform(lambda x: np.mean((base) ** x))
    return miner_df


# =========================
# Evaluation Functions
# =========================
def point_eval(miner_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate absolute error between point prediction and RR_EvalTime and assign ranks/shares.
    Args:
        miner_df: DataFrame with miner predictions.
    Returns:
        DataFrame with error, rank, and share columns.
    """
    miner_df["point_error"] = (
        np.abs(miner_df["point_forecast"] - miner_df["RR_EvalTime"])
        / miner_df.RR_EvalTime
    )
    
    # Calculate Point Error Rank, starting at 0
    miner_df["point_rank"] = (
        miner_df.groupby("evaluation_time")["point_error"].rank(
            method="first", ascending=True
        )
        - 1
    )
    miner_df = rank_to_share(miner_df, col_prefix="point", score_or_error="error")
    return miner_df


def min_max_rr_interval(
    miner_df: pd.DataFrame,
    rr_df: pd.DataFrame,
    pred_window: pd.Timedelta = pd.Timedelta(hours=1),
) -> pd.DataFrame:
    """
    Compute min/max Reference Rate for each prediction window.
    Args:
        miner_df: DataFrame with miner predictions.
        rr_df: DataFrame with reference rates.
        pred_window: Prediction window as timedelta.
    Returns:
        DataFrame with min/max RR and window info.
    """

    def hourly_stats_miner(ptime_group: pd.DataFrame) -> pd.Series:
        hourly_rr = rr_df[
            (rr_df["time"] > ptime_group["prediction_time"].iloc[0])
            & (rr_df["time"] <= ptime_group["prediction_time"].iloc[0] + pred_window)
        ]
        return pd.Series(
            {
                "min_RR": hourly_rr["ReferenceRateUSD"].min(),
                "max_RR": hourly_rr["ReferenceRateUSD"].max(),
                "window_start": ptime_group["prediction_time"].iloc[0],
                "window_end": ptime_group["prediction_time"].iloc[0] + pred_window,
                "num_evals": hourly_rr["time"].count(),
            }
        )

    minmax_rr = (
        miner_df.groupby(["prediction_time"]).apply(hourly_stats_miner).reset_index()
    )
    minmax_rr["total_times"] = (
        minmax_rr["window_end"] - minmax_rr["window_start"]
    ).dt.total_seconds()
    return minmax_rr


def interval_width_eval(
    miner_df: pd.DataFrame,
    rr_df: pd.DataFrame,
    pred_window: pd.Timedelta = pd.Timedelta(hours=1),
) -> pd.DataFrame:
    """
    Evaluate interval width and inclusion for each miner's prediction.
    Args:
        miner_df: DataFrame with miner predictions.
        rr_df: DataFrame with reference rates.
        pred_window: Prediction window as timedelta.
    Returns:
        DataFrame with interval evaluation columns.
    """

    def inclusion_by_miner(pred_miner_group: pd.DataFrame) -> pd.Series:
        hourly_rr = rr_df[
            (rr_df["time"] > pred_miner_group["prediction_time"].iloc[0])
            & (
                rr_df["time"]
                <= pred_miner_group["prediction_time"].iloc[0] + pred_window
            )
        ]
        inclusion_num = (
            (
                hourly_rr["ReferenceRateUSD"]
                >= pred_miner_group["interval_lower"].iloc[0]
            )
            & (
                hourly_rr["ReferenceRateUSD"]
                <= pred_miner_group["interval_upper"].iloc[0]
            )
        ).sum()
        return pd.Series({"inclusion_num": inclusion_num})

    # Calculate inclusion values
    inclusion_df = (
        miner_df.groupby(["prediction_time", "miner_hotkey"])
        .apply(inclusion_by_miner)
        .reset_index()
    )

    # Calculate width factor
    miner_df["eff_top"] = np.minimum(miner_df["interval_upper"], miner_df["max_RR"])
    miner_df["eff_bottom"] = np.maximum(miner_df["interval_lower"], miner_df["min_RR"])
    miner_df["width_factor"] = (miner_df["eff_top"] - miner_df["eff_bottom"]) / (
        miner_df["interval_upper"] - miner_df["interval_lower"]
    )
    
    # Calculate inclusion factor
    miner_df = miner_df.merge(
        inclusion_df, on=["prediction_time", "miner_hotkey"], how="left"
    )
    miner_df["inclusion_factor"] = miner_df["inclusion_num"] / miner_df["total_times"]
    
    # Combine for final score and rank, starting at 0
    miner_df["interval_score"] = miner_df["width_factor"] * miner_df["inclusion_factor"]
    miner_df["interval_rank"] = (
        miner_df.groupby("evaluation_time")["interval_score"].rank(
            method="first", ascending=False
        )
        - 1
    )
    
    # Calculate share based on rank
    miner_df = rank_to_share(miner_df, col_prefix="interval", score_or_error="score")
    
    return miner_df


# =========================
# EMA Calculation
# =========================
def add_ema_score(
    shares_df: pd.DataFrame, n_window: int = EMA_WINDOW, alpha: float = EMA_ALPHA
) -> pd.DataFrame:
    """
    Add exponential weighted mean (EMA) score to DataFrame, filling missing values as 0.
    Args:
        shares_df: DataFrame with share columns.
        n_window: Window size for rolling calculation.
        alpha: Smoothing factor for EMA.
    Returns:
        DataFrame with new EMA column.
    """
    shares_df = shares_df.sort_values(["miner_hotkey", "prediction_time"]).reset_index(
        drop=True
    )
    shares_df["total_share"] = (
        shares_df["point_share"] + shares_df["interval_share"]
    ) / 2

    def ewm_fixed_window(series, window, alpha):
        coeff = (1 - alpha) ** np.arange(window)[::-1]
        idx = range(series.index[0], series.index[0] + window)
        series_align = series.reindex(idx, fill_value=0)
        return (coeff * series_align).sum() / coeff.sum()

    shares_df["share_ema"] = (
        shares_df.groupby("miner_hotkey")["total_share"]
        .apply(
            lambda x: x.rolling(window=n_window, min_periods=1).apply(
                lambda s: ewm_fixed_window(s, window=n_window, alpha=alpha)
            )
        )
        .reset_index(level=0, drop=True)
    )
    return shares_df


# =========================
# API Data Fetch
# =========================
def get_rr(client: CoinMetricsClient, start, end) -> pd.DataFrame:
    """
    Fetch Reference Rate data from CoinMetrics API.
    Args:
        client: CoinMetricsClient instance.
        start: Start time.
        end: End time.
    Returns:
        DataFrame with reference rates.
    """
    rr = client.get_asset_metrics(
        assets="btc",
        metrics="ReferenceRateUSD",
        start_time=start,
        end_time=end,
        start_inclusive=False,
        frequency="1s",
        page_size=10000,
    ).to_dataframe()
    return rr


# =========================
# Main Execution
# =========================
if __name__ == "__main__":
    api_key = get_api_key()
    if not api_key:
        logging.warning(
            "API key not found. Please set COINMETRICS_API_KEY environment variable for prices >7 days old."
        )
        exit(1)
    client = CoinMetricsClient(api_key)
    try:
        predictions_df = load_and_clean_data(CSV_PATH)
    except Exception as e:
        logging.error(f"Failed to load or clean data: {e}")
        exit(1)
    predictions_df = point_eval(predictions_df)
    rr_df = get_rr(
        client,
        predictions_df.prediction_time.min(),
        predictions_df.evaluation_time.max(),
    )
    minmax_rr = min_max_rr_interval(predictions_df, rr_df)
    pred_rr_range = (
        pd.merge(predictions_df, minmax_rr, on="prediction_time", how="inner")
        .sort_values(by=["miner_hotkey", "prediction_time"])
        .reset_index(drop=True)
    )
    score_df = interval_width_eval(pred_rr_range, rr_df)
    score_df = add_ema_score(score_df)
    score_df.to_csv("precog_miner_scores.csv", index=False)
