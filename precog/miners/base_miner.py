import time
from typing import Tuple

import bittensor as bt
import pandas as pd

from precog.protocol import Challenge
from precog.utils.cm_data import CMData
from precog.utils.timestamp import get_before, to_datetime, to_str


def get_point_estimate(cm: CMData, timestamp: str) -> float:
    """Make a naive forecast by predicting the most recent price

    Args:
        cm (CMData): The CoinMetrics API client
        timestamp (str): The current timestamp provided by the validator request

    Returns:
        (float): The current BTC price tied to the provided timestamp
    """

    # Ensure timestamp is correctly typed and set to UTC
    provided_timestamp = to_datetime(timestamp)

    # Query CM API for a pandas dataframe with only one record
    price_data: pd.DataFrame = cm.get_CM_ReferenceRate(
        assets="BTC",
        start=None,
        end=to_str(provided_timestamp),
        frequency="1s",
        limit_per_asset=1,
        paging_from="end",
        use_cache=False,
    )

    # Get current price closest to the provided timestamp
    btc_price: float = float(price_data["ReferenceRateUSD"].iloc[-1])

    # Return the current price of BTC as our point estimate
    return btc_price


def get_prediction_interval(cm: CMData, timestamp: str, point_estimate: float) -> Tuple[float, float]:
    """Make a naive multi-step prediction interval by estimating
    the sample standard deviation

    Args:
        cm (CMData): The CoinMetrics API client
        timestamp (str): The current timestamp provided by the validator request
        point_estimate (float): The center of the prediction interval

    Returns:
        (float): The 90% naive prediction interval lower bound
        (float): The 90% naive prediction interval upper bound

    Notes:
        Make reasonable assumptions that the 1s BTC price residuals are
        uncorrelated and normally distributed
    """

    # Set the time range to be 24 hours
    # Ensure both timestamps are correctly typed and set to UTC
    start_time = get_before(timestamp, days=1, minutes=0, seconds=0)
    end_time = to_datetime(timestamp)

    # Query CM API for sample standard deviation of the 1s residuals
    historical_price_data: pd.DataFrame = cm.get_CM_ReferenceRate(
        assets="BTC", start=to_str(start_time), end=to_str(end_time), frequency="1s"
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


def forward(synapse: Challenge, cm: CMData) -> Challenge:
    total_start_time = time.perf_counter()
    bt.logging.info(
        f"👈 Received prediction request from: {synapse.dendrite.hotkey} for timestamp: {synapse.timestamp}"
    )

    point_estimate_start = time.perf_counter()
    # Get the naive point estimate
    point_estimate: float = get_point_estimate(cm=cm, timestamp=synapse.timestamp)

    point_estimate_time = time.perf_counter() - point_estimate_start
    bt.logging.debug(f"⏱️ Point estimate function took: {point_estimate_time:.3f} seconds")

    interval_start = time.perf_counter()
    # Get the naive prediction interval
    prediction_interval: Tuple[float, float] = get_prediction_interval(
        cm=cm, timestamp=synapse.timestamp, point_estimate=point_estimate
    )

    interval_time = time.perf_counter() - interval_start
    bt.logging.debug(f"⏱️ Prediction interval function took: {interval_time:.3f} seconds")

    synapse.prediction = point_estimate
    synapse.interval = prediction_interval

    total_time = time.perf_counter() - total_start_time
    bt.logging.debug(f"⏱️ Total forward call took: {total_time:.3f} seconds")

    if synapse.prediction is not None:
        bt.logging.success(f"Predicted price: {synapse.prediction}  |  Predicted Interval: {synapse.interval}")
    else:
        bt.logging.info("No prediction for this request.")
    return synapse
