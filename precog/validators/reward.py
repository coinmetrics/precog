from typing import List

import bittensor as bt
import numpy as np
from pandas import DataFrame

from precog.protocol import Challenge
from precog.utils.cm_data import CMData
from precog.utils.general import pd_to_dict, rank
from precog.utils.timestamp import align_timepoints, get_before, mature_dictionary, to_str


################################################################################
def calc_rewards(
    self,
    responses: List[Challenge],
) -> np.ndarray:
    evaluation_window_hours = self.config.evaluation_window_hours
    prediction_future_hours = self.config.prediction_future_hours

    # preallocate
    point_errors = []
    interval_errors = []
    decay = 0.9
    weights = np.linspace(0, len(self.available_uids) - 1, len(self.available_uids))
    decayed_weights = decay**weights
    timestamp = responses[0].timestamp
    bt.logging.debug(f"Calculating rewards for timestamp: {timestamp}")
    cm = CMData()
    # Adjust time window to look at predictions that have had time to mature
    # Start: (evaluation_window + prediction) hours ago
    # End: 1 hour ago (to ensure all predictions have matured)
    start_time: str = to_str(get_before(timestamp=timestamp, hours=evaluation_window_hours + prediction_future_hours))
    end_time: str = to_str(get_before(timestamp=timestamp, hours=prediction_future_hours))
    # Query CM API for sample standard deviation of the 1s residuals
    historical_price_data: DataFrame = cm.get_CM_ReferenceRate(
        assets="BTC", start=start_time, end=end_time, frequency="1s"
    )
    cm_data = pd_to_dict(historical_price_data)

    bt.logging.debug(f"CM data length: {len(cm_data)}")
    if len(cm_data) > 0:
        bt.logging.debug(f"CM data first timestamp: {list(cm_data.keys())[0]}")
        bt.logging.debug(f"CM data last timestamp: {list(cm_data.keys())[-1]}")

    for uid, response in zip(self.available_uids, responses):
        current_miner = self.MinerHistory[uid]
        self.MinerHistory[uid].add_prediction(response.timestamp, response.prediction, response.interval)
        # Get predictions from the evaluation window that have had time to mature
        prediction_dict, interval_dict = current_miner.format_predictions(
            get_before(timestamp, hours=prediction_future_hours),
            hours=evaluation_window_hours,
        )

        # Mature the predictions (shift forward by 1 hour)
        mature_time_dict = mature_dictionary(prediction_dict, hours=prediction_future_hours)

        bt.logging.debug(
            f"UID: {uid} | LENGTHS: prediction_dict={len(prediction_dict)}, mature_time_dict={len(mature_time_dict)}"
        )
        bt.logging.debug(
            f"UID: {uid} | LENGTHS: prediction_dict={len(prediction_dict)}, interval_dict={len(interval_dict)}, mature_time_dict={len(mature_time_dict)}"
        )
        if len(mature_time_dict) > 0:
            bt.logging.debug(f"UID: {uid} | Mature dict first timestamp: {list(mature_time_dict.keys())[0]}")
            bt.logging.debug(f"UID: {uid} | Mature dict last timestamp: {list(mature_time_dict.keys())[-1]}")
        preds, price, aligned_pred_timestamps = align_timepoints(mature_time_dict, cm_data)
        bt.logging.debug(
            f"UID: {uid} | AFTER ALIGNMENT: preds={len(preds)}, price={len(price)}, aligned_timestamps={len(aligned_pred_timestamps)}"
        )
        for i, j, k in zip(preds, price, aligned_pred_timestamps):
            bt.logging.debug(f"Prediction: {i} | Price: {j} | Aligned Prediction: {k}")
        inters, interval_prices, aligned_int_timestamps = align_timepoints(interval_dict, cm_data)
        for i, j, k in zip(inters, interval_prices, aligned_int_timestamps):
            bt.logging.debug(f"Interval: {i} | Interval Price: {j} | Aligned TS: {k}")
        point_errors.append(point_error(preds, price))
        try:
            if len(inters) == 0 or len(interval_prices) == 0:
                interval_errors.append(np.inf)
            else:
                # Let interval_error handle NaN values
                interval_errors.append(interval_error(inters, interval_prices))
        except Exception as e:
            bt.logging.debug(f"Exception in interval error calculation: {e}")
            interval_errors.append(np.inf)
        bt.logging.debug(f"UID: {uid} | point_errors: {point_errors[-1]} | interval_errors: {interval_errors[-1]}")

    point_errors_array = np.array(point_errors)
    interval_errors_array = np.array(interval_errors)

    # Replace any remaining NaN values with appropriate defaults
    point_errors_array = np.nan_to_num(point_errors_array, nan=999.0)
    interval_errors_array = np.nan_to_num(interval_errors_array, nan=0.0)

    point_ranks = rank(np.array(point_errors_array))
    interval_ranks = rank(-np.array(interval_errors_array))  # 1 is best, 0 is worst, so flip it
    rewards = (decayed_weights[point_ranks] + decayed_weights[interval_ranks]) / 2

    bt.logging.debug(f"Point errors: {point_errors}")
    bt.logging.debug(f"Point ranks: {point_ranks}")
    bt.logging.debug(f"Interval errors: {interval_errors}")
    bt.logging.debug(f"Interval ranks: {interval_ranks}")
    bt.logging.debug(f"Decayed weights: {decayed_weights}")
    bt.logging.debug(f"Final rewards: {rewards}")
    return rewards


def interval_error(intervals, cm_prices):
    if intervals is None or len(intervals) <= 1:
        return 0.0

    interval_errors = []
    for i, interval_to_evaluate in enumerate(intervals[:-1]):
        if i + 1 >= len(cm_prices):
            continue

        lower_bound_prediction = np.min(interval_to_evaluate)
        upper_bound_prediction = np.max(interval_to_evaluate)

        if upper_bound_prediction <= lower_bound_prediction:
            continue

        effective_min = np.max([lower_bound_prediction, np.min(cm_prices[i + 1 :])])
        effective_max = np.min([upper_bound_prediction, np.max(cm_prices[i + 1 :])])

        if effective_max <= effective_min:
            continue

        f_w = (effective_max - effective_min) / (upper_bound_prediction - lower_bound_prediction)
        f_i = sum(
            (cm_prices[i + 1 :] >= lower_bound_prediction) & (cm_prices[i + 1 :] <= upper_bound_prediction)
        ) / len(cm_prices[i + 1 :])

        interval_errors.append(f_w * f_i)

    error_array = np.array(interval_errors)

    if len(error_array) == 0 or np.all(np.isnan(error_array)):
        return 0.0

    return np.nanmean(error_array).item()


def point_error(predictions, cm_prices) -> np.ndarray:
    if predictions is None or len(predictions) == 0 or len(cm_prices) == 0:
        return np.inf

    # Convert to numpy arrays
    pred_array = np.array(predictions)
    price_array = np.array(cm_prices)

    # Calculate absolute percentage errors, handling division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        abs_pct_errors = np.abs(pred_array - price_array) / price_array

    # Check if all values are NaN
    if np.all(np.isnan(abs_pct_errors)):
        return np.inf

    # Calculate mean ignoring NaN values
    point_error = np.nanmean(abs_pct_errors)

    # Handle if result is still NaN
    if np.isnan(point_error):
        return np.inf

    return point_error.item()
