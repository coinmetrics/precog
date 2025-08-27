import time
from typing import Tuple

import bittensor as bt
import pandas as pd

from precog.protocol import Challenge
from precog.utils.cm_data import CMData
from precog.utils.timestamp import get_before, to_datetime, to_str


def calculate_prediction_interval(
    point_estimate: float, historical_prices: pd.Series, asset: str = "btc"
) -> Tuple[float, float]:
    """Calculate prediction interval using historical volatility from provided data

    Args:
        point_estimate (float): The center of the prediction interval
        historical_prices (pd.Series): Historical price data for volatility calculation
        asset (str): The asset name for logging

    Returns:
        (float): The lower bound
        (float): The upper bound
    """
    try:

        if historical_prices.empty or len(historical_prices) < 100:
            bt.logging.warning(f"Insufficient data for {asset}, using fallback interval")
            # Fallback: ±10% of point estimate
            margin = point_estimate * 0.10
            return point_estimate - margin, point_estimate + margin

        # Calculate returns (percentage changes)
        hourly_returns = historical_prices.pct_change().dropna()

        # Remove extreme outliers (beyond 3 std devs) to get realistic volatility
        returns_std = hourly_returns.std()
        returns_mean = hourly_returns.mean()
        outlier_mask = abs(hourly_returns - returns_mean) <= 3 * returns_std
        clean_returns = hourly_returns[outlier_mask]

        if len(clean_returns) < 12:
            bt.logging.warning(f"Too few clean data points for {asset}, using fallback")
            margin = point_estimate * 0.10  # Increased from 5%
            return point_estimate - margin, point_estimate + margin

        # Use standard deviation of hourly returns for 1-hour prediction
        hourly_vol = float(clean_returns.std())

        # Use a wider confidence interval for better coverage
        # 2.58 standard deviations = 99% confidence interval
        # This provides much better coverage for all assets
        margin = point_estimate * hourly_vol * 2.58

        # Increase bounds to be more generous for all assets
        max_margin = point_estimate * 0.30  # Cap at ±30% (increased from 15%)
        min_margin = point_estimate * 0.02  # Minimum ±2% (increased from 1%)

        margin = max(min_margin, min(margin, max_margin))

        lower_bound = point_estimate - margin
        upper_bound = point_estimate + margin

        bt.logging.debug(f"{asset}: hourly_vol={hourly_vol:.4f}, margin=${margin:.2f}")

        return lower_bound, upper_bound

    except Exception as e:
        bt.logging.error(f"Error calculating interval for {asset}: {e}")
        # Emergency fallback: ±15% interval for better coverage
        margin = point_estimate * 0.15  # Increased to 15% for better coverage
        return point_estimate - margin, point_estimate + margin


async def forward_async(synapse: Challenge, cm: CMData) -> Challenge:
    total_start_time = time.perf_counter()

    # Get list of assets to predict and ensure lowercase
    raw_assets = synapse.assets if hasattr(synapse, "assets") else ["btc"]
    assets = [asset.lower() for asset in raw_assets]

    bt.logging.info(
        f"👈 Received prediction request from: {synapse.dendrite.hotkey} for {assets} at timestamp: {synapse.timestamp}"
    )

    # Get timestamps for data fetch
    provided_timestamp = to_datetime(synapse.timestamp)
    start_timestamp = get_before(synapse.timestamp, hours=1, minutes=0, seconds=0)  # 1 hour - sufficient for volatility

    # Fetch ALL data we need in a single API call
    all_data = cm.get_CM_ReferenceRate(
        assets=assets,
        start=to_str(start_timestamp),
        end=to_str(provided_timestamp),
        frequency="1s",
    )

    predictions = {}
    intervals = {}

    if not all_data.empty:
        for asset in assets:
            # Filter data for this asset
            asset_data = all_data[all_data["asset"] == asset]

            if not asset_data.empty:
                # Get latest price as point estimate
                point_estimate = float(asset_data["ReferenceRateUSD"].iloc[-1])
                bt.logging.info(f"Point estimate for {asset}: ${point_estimate:.2f}")

                # Use historical prices for volatility calculation
                historical_prices = asset_data["ReferenceRateUSD"]

                # Calculate interval using the historical data
                interval = calculate_prediction_interval(point_estimate, historical_prices, asset)

                predictions[asset] = point_estimate
                intervals[asset] = list(interval)

                # Log the complete prediction for this asset
                bt.logging.info(
                    f"{asset}: Prediction=${point_estimate:.2f} | Interval=[${interval[0]:.2f}, ${interval[1]:.2f}]"
                )
            else:
                bt.logging.warning(f"No data for {asset} in response")

    synapse.predictions = predictions
    synapse.intervals = intervals

    total_time = time.perf_counter() - total_start_time
    bt.logging.debug(f"⏱️ Total forward call took: {total_time:.3f} seconds")

    if synapse.predictions:
        bt.logging.success(f"Predictions complete for {list(predictions.keys())}")
    else:
        bt.logging.info("No predictions for this request.")
    return synapse


async def forward(synapse: Challenge, cm: CMData) -> Challenge:
    """Async forward function for handling predictions"""
    return await forward_async(synapse, cm)
