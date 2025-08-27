import asyncio
import time
from typing import Tuple

import bittensor as bt
import pandas as pd

from precog.protocol import Challenge
from precog.utils.cm_data import CMData
from precog.utils.timestamp import get_before, to_datetime, to_str


def get_point_estimate(cm: CMData, timestamp: str, asset: str = "btc") -> float:
    """Make a naive forecast by predicting the most recent price

    Args:
        cm (CMData): The CoinMetrics API client
        timestamp (str): The current timestamp provided by the validator request
        asset (str): The asset to predict (default: "btc")

    Returns:
        (float): The current asset price tied to the provided timestamp
    """

    # Ensure timestamp is correctly typed and set to UTC
    provided_timestamp = to_datetime(timestamp)

    # Query CM API for a pandas dataframe with only one record
    price_data: pd.DataFrame = cm.get_CM_ReferenceRate(
        assets=asset,
        start=None,
        end=to_str(provided_timestamp),
        frequency="1s",
        limit_per_asset=1,
        paging_from="end",
        use_cache=False,
    )

    # Get current price closest to the provided timestamp
    asset_price: float = float(price_data["ReferenceRateUSD"].iloc[-1])

    # Return the current price of the asset as our point estimate
    bt.logging.info(f"Point estimate for {asset} at {provided_timestamp}: ${asset_price:.2f} (from CM API)")
    return asset_price


def get_prediction_interval(
    cm: CMData, timestamp: str, point_estimate: float, asset: str = "btc"
) -> Tuple[float, float]:
    """Make a reasonable prediction interval using hourly volatility

    Args:
        cm (CMData): The CoinMetrics API client
        timestamp (str): The current timestamp provided by the validator request
        point_estimate (float): The center of the prediction interval
        asset (str): The asset to predict (default: "btc")

    Returns:
        (float): The 90% prediction interval lower bound
        (float): The 90% prediction interval upper bound
    """
    try:
        # Get hourly data for the past 7 days to estimate realistic volatility
        start_time = get_before(timestamp, days=7, minutes=0, seconds=0)
        end_time = to_datetime(timestamp)

        # Query CM API for hourly data (much more appropriate for 1-hour predictions)
        historical_price_data: pd.DataFrame = cm.get_CM_ReferenceRate(
            assets=asset, start=to_str(start_time), end=to_str(end_time), frequency="1h"
        )

        if historical_price_data.empty or len(historical_price_data) < 24:
            bt.logging.warning(f"Insufficient data for {asset}, using fallback interval")
            # Fallback: ±10% of point estimate (increased from 5%)
            margin = point_estimate * 0.10
            return point_estimate - margin, point_estimate + margin

        # Calculate hourly returns (percentage changes)
        prices = historical_price_data["ReferenceRateUSD"]
        hourly_returns = prices.pct_change().dropna()

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


async def predict_asset(cm: CMData, timestamp: str, asset: str) -> Tuple[str, float, Tuple[float, float]]:
    """Predict a single asset asynchronously"""
    asset_start = time.perf_counter()

    # Get the naive point estimate
    point_estimate: float = get_point_estimate(cm=cm, timestamp=timestamp, asset=asset)

    # Get the naive prediction interval
    prediction_interval: Tuple[float, float] = get_prediction_interval(
        cm=cm, timestamp=timestamp, point_estimate=point_estimate, asset=asset
    )

    asset_time = time.perf_counter() - asset_start
    bt.logging.debug(f"⏱️ {asset} prediction took: {asset_time:.3f} seconds")

    return asset, point_estimate, prediction_interval


async def forward_async(synapse: Challenge, cm: CMData) -> Challenge:
    total_start_time = time.perf_counter()

    # Get list of assets to predict and ensure lowercase
    raw_assets = synapse.assets if hasattr(synapse, "assets") else ["btc"]
    assets = [asset.lower() for asset in raw_assets]

    bt.logging.info(
        f"👈 Received prediction request from: {synapse.dendrite.hotkey} for {assets} at timestamp: {synapse.timestamp}"
    )

    # Create prediction tasks for all assets in parallel
    tasks = [predict_asset(cm, synapse.timestamp, asset) for asset in assets]

    # Run all predictions in parallel
    results = await asyncio.gather(*tasks)

    # Collect results
    predictions = {}
    intervals = {}

    for asset, point_estimate, prediction_interval in results:
        predictions[asset] = point_estimate
        intervals[asset] = list(prediction_interval)

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
