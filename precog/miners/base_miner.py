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
    return asset_price


def get_prediction_interval(
    cm: CMData, timestamp: str, point_estimate: float, asset: str = "btc"
) -> Tuple[float, float]:
    """Make a naive multi-step prediction interval by estimating
    the sample standard deviation

    Args:
        cm (CMData): The CoinMetrics API client
        timestamp (str): The current timestamp provided by the validator request
        point_estimate (float): The center of the prediction interval
        asset (str): The asset to predict (default: "btc")

    Returns:
        (float): The 90% naive prediction interval lower bound
        (float): The 90% naive prediction interval upper bound

    Notes:
        Make reasonable assumptions that the 1s asset price residuals are
        uncorrelated and normally distributed
    """

    # Set the time range to be 24 hours
    # Ensure both timestamps are correctly typed and set to UTC
    start_time = get_before(timestamp, days=1, minutes=0, seconds=0)
    end_time = to_datetime(timestamp)

    # Query CM API for sample standard deviation of the 1s residuals
    historical_price_data: pd.DataFrame = cm.get_CM_ReferenceRate(
        assets=asset, start=to_str(start_time), end=to_str(end_time), frequency="1s"
    )
    residuals: pd.Series = historical_price_data["ReferenceRateUSD"].diff()
    sample_std_dev: float = float(residuals.std())

    # We have the standard deviation of the 1s residuals
    # We are forecasting forward 60m, which is 3600s
    # We must scale the 1s sample standard deviation to reflect a 3600s forecast
    # Make reasonable assumptions that the 1s residuals are uncorrelated and normally distributed
    # To do this naively, we multiply the std dev by the square root of the number of time steps
    time_steps: int = 3600
    naive_forecast_std_dev: float = sample_std_dev * (time_steps**0.5)

    # For a 90% prediction interval, we use the coefficient 1.64
    # Make reasonable assumptions that the 1s residuals are uncorrelated and normally distributed
    coefficient: float = 1.64

    # Calculate the lower bound and upper bound
    lower_bound: float = point_estimate - coefficient * naive_forecast_std_dev
    upper_bound: float = point_estimate + coefficient * naive_forecast_std_dev

    # Return the naive prediction interval for our forecast
    return lower_bound, upper_bound


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
