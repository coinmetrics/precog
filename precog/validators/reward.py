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

    # Debug: Check the first few iterations to see what's happening (only for UIDs 8 and 30)
    debug_enabled = uid in [8, 30]
    debug_first_items = 5
    debug_count = 0

    for price_time, price_value in cm_data.items():
        items_checked += 1

        # Debug first few items for specific UIDs only
        if debug_enabled and debug_count < debug_first_items:
            in_range = prediction_time <= price_time <= eval_time
            bt.logging.debug(
                f"_calculate_interval_score item {debug_count}: "
                f"time={price_time}, value={price_value:.2f}, "
                f"in_range={in_range}"
            )
            if not in_range:
                bt.logging.debug(f"  Range check details: pred={prediction_time} <= {price_time} <= eval={eval_time}")
                bt.logging.debug(
                    f"  pred <= price: {prediction_time <= price_time}, " f"price <= eval: {price_time <= eval_time}"
                )
            debug_count += 1

        in_range_actual = prediction_time <= price_time <= eval_time
        if in_range_actual:
            hour_prices.append(price_value)
            if debug_enabled and debug_count <= 5:
                bt.logging.debug(f"_calculate_interval_score UID {uid}: Added price {price_value} at {price_time}")
        elif debug_enabled and debug_count <= 5:
            bt.logging.debug(f"_calculate_interval_score UID {uid}: REJECTED price {price_value} at {price_time}")

    if debug_enabled:
        bt.logging.debug(f"_calculate_interval_score UID {uid}: Found {len(hour_prices)} prices in range")

    if not hour_prices:
        if debug_enabled:
            bt.logging.debug(f"_calculate_interval_score UID {uid}: Checked {items_checked} items, found 0 in range")
            bt.logging.debug(
                f"_calculate_interval_score UID {uid}: prediction_time type: {type(prediction_time)}, value: {prediction_time}"
            )
            bt.logging.debug(
                f"_calculate_interval_score UID {uid}: eval_time type: {type(eval_time)}, value: {eval_time}"
            )

            # Check if the times are in cm_data
            if prediction_time in cm_data:
                bt.logging.debug(f"_calculate_interval_score UID {uid}: prediction_time IS in cm_data")
            else:
                bt.logging.debug(f"_calculate_interval_score UID {uid}: prediction_time NOT in cm_data")

            if eval_time in cm_data:
                bt.logging.debug(f"_calculate_interval_score UID {uid}: eval_time IS in cm_data")
            else:
                bt.logging.debug(f"_calculate_interval_score UID {uid}: eval_time NOT in cm_data")

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
    uid, response, assets, all_cm_data, eval_time, asset_point_errors, asset_interval_scores, prediction_time
):
    """Process predictions for all assets for a single miner."""
    for asset in assets:
        cm_data = all_cm_data[asset]

        # Debug logging for UIDs 8 and 30
        if uid in [8, 30]:
            bt.logging.debug(f"UID {uid} | {asset} | cm_data has {len(cm_data)} entries")

        # Handle point predictions
        if not response.predictions or asset not in response.predictions:
            asset_point_errors[asset].append(np.inf)
            bt.logging.debug(f"UID: {uid} | {asset} | No prediction provided")
        else:
            prediction_value = response.predictions[asset]
            if eval_time not in cm_data:
                asset_point_errors[asset].append(np.inf)
                bt.logging.debug(f"UID: {uid} | {asset} | No price data at {eval_time}")
            else:
                actual_price = cm_data[eval_time]
                current_point_error = abs(prediction_value - actual_price) / actual_price
                asset_point_errors[asset].append(current_point_error)
                bt.logging.debug(
                    f"UID: {uid} | {asset} | Prediction: {prediction_value} | Actual: {actual_price} | Error: {current_point_error}"
                )

        # Handle interval predictions
        if not response.intervals or asset not in response.intervals:
            asset_interval_scores[asset].append(0)
            bt.logging.debug(f"UID: {uid} | {asset} | No interval prediction provided")
        else:
            interval_bounds = response.intervals[asset]
            interval_score_value = _calculate_interval_score(
                prediction_time, eval_time, interval_bounds, cm_data, uid=uid
            )
            asset_interval_scores[asset].append(interval_score_value)

            if interval_score_value > 0:
                pred_min = min(interval_bounds)
                pred_max = max(interval_bounds)
                bt.logging.debug(
                    f"UID: {uid} | {asset} | Interval: [{pred_min}, {pred_max}] | Score: {interval_score_value:.3f}"
                )
            else:
                bt.logging.debug(f"UID: {uid} | {asset} | No price data for interval evaluation")


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
    bt.logging.debug(f"Calculating rewards for timestamp: {timestamp}")
    cm = CMData()

    # Current evaluation time and when prediction was made
    eval_time = to_datetime(timestamp)
    prediction_time = get_before(timestamp=timestamp, hours=prediction_future_hours, minutes=0)

    bt.logging.info(f"Timestamp from response: {timestamp}")
    bt.logging.info(f"Eval time (converted): {eval_time}")
    bt.logging.info(f"Prediction time (eval_time - {prediction_future_hours}h): {prediction_time}")
    bt.logging.info(f"prediction_future_hours constant: {prediction_future_hours}")

    # Get price data for the past hour (the hour that was predicted)
    # Miners predicted at prediction_time for the period [prediction_time, eval_time]
    start_time: str = to_str(prediction_time)
    end_time: str = to_str(eval_time)

    # Get assets from the first response
    assets = responses[0].assets

    # Fetch price data for all assets in one API call
    bt.logging.info(f"Fetching CM data from {start_time} to {end_time} for assets: {assets}")
    historical_price_data: DataFrame = cm.get_CM_ReferenceRate(
        assets=assets, start=start_time, end=end_time, frequency="1s"
    )

    # Split data by asset
    all_cm_data = {}
    if not historical_price_data.empty:
        for asset in assets:
            asset_data = historical_price_data[historical_price_data["asset"] == asset]
            if not asset_data.empty:
                bt.logging.info(
                    f"Asset {asset}: DataFrame shape={asset_data.shape}, index={list(asset_data.index[:5])}, columns={list(asset_data.columns)}"
                )
                # Debug: Check DataFrame time range before conversion
                bt.logging.debug(
                    f"Asset {asset} DataFrame time range: {asset_data['time'].min()} to {asset_data['time'].max()}"
                )
                bt.logging.debug(f"Asset {asset} DataFrame has {len(asset_data)} rows")

                all_cm_data[asset] = pd_to_dict(asset_data)  # noqa
                bt.logging.info(f"CM data fetched for {asset}: {len(all_cm_data[asset])} price points")

                # Debug: Verify dict has expected entries
                dict_times = list(all_cm_data[asset].keys())
                if dict_times:
                    bt.logging.debug(f"Asset {asset} dict time range: {min(dict_times)} to {max(dict_times)}")
                    if len(dict_times) != len(asset_data):
                        bt.logging.warning(
                            f"Asset {asset}: DataFrame had {len(asset_data)} rows but dict has {len(dict_times)} entries!"
                        )
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

    # Debug: Log initial all_cm_data state
    bt.logging.debug("Initial all_cm_data sizes after fetching:")
    for asset in assets:
        bt.logging.debug(f"  {asset}: {len(all_cm_data[asset])} entries, id: {id(all_cm_data[asset])}")

    for uid, response in zip(self.available_uids, responses):
        # Store multi-asset predictions in MinerHistory
        self.MinerHistory[uid].add_prediction(response.timestamp, response.predictions, response.intervals)

        # Debug logging for specific UIDs
        if uid in [8, 30]:
            bt.logging.debug(f"UID {uid} - Before processing, all_cm_data sizes:")
            for asset in assets:
                bt.logging.debug(f"  {asset}: {len(all_cm_data[asset])} entries")
                if len(all_cm_data[asset]) <= 5:
                    bt.logging.debug(f"    Keys: {list(all_cm_data[asset].keys())}")

        # Process all asset predictions for this miner
        _process_asset_predictions(
            uid, response, assets, all_cm_data, eval_time, asset_point_errors, asset_interval_scores, prediction_time
        )

        # Debug logging after processing
        if uid in [8, 30]:
            bt.logging.debug(f"UID {uid} - After processing, all_cm_data sizes:")
            for asset in assets:
                bt.logging.debug(f"  {asset}: {len(all_cm_data[asset])} entries")
                if len(all_cm_data[asset]) <= 5:
                    bt.logging.debug(f"    Keys: {list(all_cm_data[asset].keys())}")

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
