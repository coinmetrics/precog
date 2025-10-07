from typing import List

import bittensor as bt
import numpy as np
from pandas import DataFrame

from precog import constants
from precog.protocol import Challenge
from precog.utils.cm_data import CMData
from precog.utils.general import get_average_weights_for_ties, pd_to_dict, rank
from precog.utils.timestamp import get_before, to_datetime, to_str


def _calculate_interval_score(prediction_time, eval_time, interval_bounds, cm_data, uid=None):  # noqa: C901
    """Calculate interval score for a single asset prediction."""
    hour_prices = []
    items_checked = 0

    for price_time, price_value in cm_data.items():
        items_checked += 1

        if prediction_time <= price_time <= eval_time:
            hour_prices.append(price_value)

    if not hour_prices:
        return 0

    pred_min = min(interval_bounds)
    pred_max = max(interval_bounds)
    observed_min = min(hour_prices)
    observed_max = max(hour_prices)

    # Calculate effective top and bottom
    effective_top = min(pred_max, observed_max)
    effective_bottom = max(pred_min, observed_min)

    # Calculate width factor (f_w)
    if pred_max == pred_min:
        width_factor = 0
    else:
        width_factor = (effective_top - effective_bottom) / (pred_max - pred_min)

    # Calculate inclusion factor (f_i)
    prices_in_bounds = sum(1 for price in hour_prices if pred_min <= price <= pred_max)
    inclusion_factor = prices_in_bounds / len(hour_prices)

    return inclusion_factor * width_factor


def _process_asset_predictions(
    uid, current_miner, assets, all_cm_data, eval_time, asset_point_errors, asset_interval_scores, prediction_time
):
    """Process predictions for all assets for a single miner."""
    for asset in assets:
        cm_data = all_cm_data[asset]

        if prediction_time not in current_miner.predictions:
            asset_point_errors[asset].append(np.inf)
        else:
            stored_predictions = current_miner.predictions[prediction_time]

            if not stored_predictions or asset not in stored_predictions:
                asset_point_errors[asset].append(np.inf)
            else:
                prediction_value = stored_predictions[asset]
                if eval_time not in cm_data:
                    asset_point_errors[asset].append(np.inf)
                else:
                    actual_price = cm_data[eval_time]
                    current_point_error = abs(prediction_value - actual_price) / actual_price
                    asset_point_errors[asset].append(current_point_error)
                    bt.logging.debug(f"UID: {uid} | {asset} | Point error: {current_point_error:.4f}")

        if prediction_time not in current_miner.intervals:
            asset_interval_scores[asset].append(0)
        else:
            stored_intervals = current_miner.intervals[prediction_time]

            if not stored_intervals or asset not in stored_intervals:
                asset_interval_scores[asset].append(0)
            else:
                interval_bounds = stored_intervals[asset]
                interval_score_value = _calculate_interval_score(
                    prediction_time, eval_time, interval_bounds, cm_data, uid=uid
                )
                asset_interval_scores[asset].append(interval_score_value)
                bt.logging.debug(f"UID: {uid} | {asset} | Interval score: {interval_score_value:.4f}")


def calc_rewards(  # noqa: C901
    self,
    responses: List[Challenge],
) -> np.ndarray:
    prediction_future_hours = constants.PREDICTION_FUTURE_HOURS

    # preallocate
    asset_point_errors = {}
    asset_interval_scores = {}
    decay = 0.8
    timestamp = responses[0].timestamp
    cm = CMData()

    # Current evaluation time and when prediction was made
    eval_time = to_datetime(timestamp)
    prediction_time = get_before(timestamp=timestamp, hours=prediction_future_hours, minutes=0)

    # Get price data for the past hour (the hour that was predicted)
    # Miners predicted at prediction_time for the period [prediction_time, eval_time]
    start_time: str = to_str(prediction_time)
    end_time: str = to_str(eval_time)

    # Get assets from the first response
    assets = responses[0].assets

    # Fetch price data for all assets in one API call
    historical_price_data: DataFrame = cm.get_CM_ReferenceRate(
        assets=assets, start=start_time, end=end_time, frequency="1s"
    )

    # Split data by asset
    all_cm_data = {}
    if not historical_price_data.empty:
        for asset in assets:
            asset_data = historical_price_data[historical_price_data["asset"] == asset]
            if not asset_data.empty:
                all_cm_data[asset] = pd_to_dict(asset_data)  # noqa
            else:
                all_cm_data[asset] = {}
                bt.logging.warning(f"No CM data returned for {asset}")
    else:
        bt.logging.warning("No CM data returned for any assets")
        for asset in assets:
            all_cm_data[asset] = {}

    # Initialize error tracking for each asset
    for asset in assets:
        asset_point_errors[asset] = []
        asset_interval_scores[asset] = []

    for uid, response in zip(self.available_uids, responses):
        current_miner = self.MinerHistory[uid]
        self.MinerHistory[uid].add_prediction(response.timestamp, response.predictions, response.intervals)

        _process_asset_predictions(
            uid,
            current_miner,
            assets,
            all_cm_data,
            eval_time,
            asset_point_errors,
            asset_interval_scores,
            prediction_time,
        )

    # Score, rank, and weight each task independently
    task_weights = {}

    for asset in assets:
        # Point prediction task
        point_ranks = rank(np.array(asset_point_errors[asset]))
        point_task_weights = get_average_weights_for_ties(point_ranks, decay)
        task_name = f"{asset}_point"
        task_weights[task_name] = point_task_weights

        bt.logging.trace(f"{task_name}_weights: {point_task_weights}")

        # Interval prediction task
        interval_ranks = rank(-np.array(asset_interval_scores[asset]))  # Flip for higher=better
        interval_task_weights = get_average_weights_for_ties(interval_ranks, decay)
        task_name = f"{asset}_interval"
        task_weights[task_name] = interval_task_weights

        bt.logging.trace(f"{task_name}_weights: {interval_task_weights}")

    # Combine weighted tasks (weights sum to 1.0)
    final_rewards = np.zeros(len(self.available_uids))

    for asset in assets:
        point_weight = constants.TASK_WEIGHTS.get(asset, {}).get("point", 0.0)
        interval_weight = constants.TASK_WEIGHTS.get(asset, {}).get("interval", 0.0)

        final_rewards += point_weight * task_weights[f"{asset}_point"]
        final_rewards += interval_weight * task_weights[f"{asset}_interval"]

    bt.logging.trace(f"final_rewards: {final_rewards}")
    return final_rewards
