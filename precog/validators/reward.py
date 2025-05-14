from typing import List

import bittensor as bt
import numpy as np
from pandas import DataFrame

from precog import constants
from precog.protocol import Challenge
from precog.utils.cm_data import CMData
from precog.utils.general import get_average_weights_for_ties, pd_to_dict, rank
from precog.utils.timestamp import align_timepoints, get_before, mature_dictionary, to_datetime, to_str


################################################################################
def calc_rewards(
    self,
    responses: List[Challenge],
) -> np.ndarray:
    evaluation_window_hours = constants.EVALUATION_WINDOW_HOURS
    prediction_future_hours = constants.PREDICTION_FUTURE_HOURS
    prediction_interval_minutes = constants.PREDICTION_INTERVAL_MINUTES

    expected_timepoints = evaluation_window_hours * 60 / prediction_interval_minutes

    # preallocate
    point_errors = []
    interval_scores = []
    completeness_scores = []
    decay = 0.9
    timestamp = responses[0].timestamp
    bt.logging.debug(f"Calculating rewards for timestamp: {timestamp}")
    cm = CMData()
    # Adjust time window to look at predictions that have had time to mature
    # Start: (evaluation_window + prediction) hours ago
    # End: prediction_future_hours ago (to ensure all predictions have matured)
    start_time: str = to_str(get_before(timestamp=timestamp, hours=evaluation_window_hours + prediction_future_hours))
    end_time: str = to_str(to_datetime(get_before(timestamp=timestamp, hours=prediction_future_hours)))
    # Query CM API for sample standard deviation of the 1s residuals
    historical_price_data: DataFrame = cm.get_CM_ReferenceRate(
        assets="BTC", start=start_time, end=end_time, frequency="1s"
    )
    cm_data = pd_to_dict(historical_price_data)

    for uid, response in zip(self.available_uids, responses):
        current_miner = self.MinerHistory[uid]
        self.MinerHistory[uid].add_prediction(response.timestamp, response.prediction, response.interval)
        # Get predictions from the evaluation window that have had time to mature
        prediction_dict, interval_dict = current_miner.format_predictions(
            reference_timestamp=get_before(timestamp, hours=prediction_future_hours),
            hours=evaluation_window_hours,
        )

        # Mature the predictions (shift forward by 1 hour)
        mature_time_dict = mature_dictionary(prediction_dict, hours=prediction_future_hours)

        preds, price, aligned_pred_timestamps = align_timepoints(mature_time_dict, cm_data)

        num_predictions = len(preds) if preds is not None else 0

        # Ensure a maximum ratio of 1.0
        completeness_ratio = min(num_predictions / expected_timepoints, 1.0)
        completeness_scores.append(completeness_ratio)
        bt.logging.debug(
            f"UID: {uid} | Completeness: {completeness_ratio:.2f} ({num_predictions}/{expected_timepoints})"
        )

        # for i, j, k in zip(preds, price, aligned_pred_timestamps):
        #     bt.logging.debug(f"Prediction: {i} | Price: {j} | Aligned Prediction: {k}")
        inters, interval_prices, aligned_int_timestamps = align_timepoints(interval_dict, cm_data)
        # for i, j, k in zip(inters, interval_prices, aligned_int_timestamps):
        #     bt.logging.debug(f"Interval: {i} | Interval Price: {j} | Aligned TS: {k}")

        # Penalize miners with missing predictions by increasing their point error
        if preds is None or len(preds) == 0:
            point_errors.append(np.inf)  # Maximum penalty for no predictions
        else:
            # Calculate error as normal, but apply completeness penalty
            base_point_error = point_error(preds, price)
            # Apply penalty inversely proportional to completeness
            # This will increase error for incomplete prediction sets
            adjusted_point_error = base_point_error / completeness_ratio
            point_errors.append(adjusted_point_error)

        if any([np.isnan(inters).any(), np.isnan(interval_prices).any()]):
            interval_scores.append(0)
        else:
            base_interval_score = interval_score(inters, interval_prices)
            adjusted_interval_score = base_interval_score * completeness_ratio
            interval_scores.append(adjusted_interval_score)

        bt.logging.debug(f"UID: {uid} | point_errors: {point_errors[-1]} | interval_scores: {interval_scores[-1]}")

    point_ranks = rank(np.array(point_errors))
    interval_ranks = rank(-np.array(interval_scores))  # 1 is best, 0 is worst, so flip it

    point_weights = get_average_weights_for_ties(point_ranks, decay)
    interval_weights = get_average_weights_for_ties(interval_ranks, decay)

    base_rewards = (point_weights + interval_weights) / 2
    rewards = base_rewards * np.array(completeness_scores)

    return rewards


def interval_score(intervals, cm_prices):
    if intervals is None:
        return np.array([0])
    else:
        intervals_per_hour = 60 / constants.PREDICTION_INTERVAL_MINUTES
        prediction_window = int(constants.PREDICTION_FUTURE_HOURS * intervals_per_hour)

        interval_scores = []

        for i, interval_to_evaluate in enumerate(intervals[:-1]):
            # Check if we have enough future data for a complete prediction window
            if i + 1 + prediction_window > len(cm_prices):
                break

            # Get bounds of the prediction interval
            lower_bound_prediction = np.min(interval_to_evaluate)
            upper_bound_prediction = np.max(interval_to_evaluate)

            # Get only the next hour of prices (the prediction window)
            future_prices = cm_prices[i + 1 : i + 1 + prediction_window]

            # Calculate effective bounds (overlap between prediction and actual)
            effective_min = np.max([lower_bound_prediction, np.min(future_prices)])
            effective_max = np.min([upper_bound_prediction, np.max(future_prices)])

            # Calculate width factor (f_w): how much of the prediction was "used"
            f_w = (effective_max - effective_min) / (upper_bound_prediction - lower_bound_prediction)

            # Calculate inclusion factor (f_i): what % of prices fell within prediction
            f_i = sum((future_prices >= lower_bound_prediction) & (future_prices <= upper_bound_prediction)) / len(
                future_prices
            )

            # Final score for this interval
            interval_score = f_w * f_i
            interval_scores.append(interval_score)

        if len(interval_scores) == 0:
            mean_score = 0.0
        else:
            mean_score = np.nanmean(np.array(interval_scores)).item()

        return mean_score


def point_error(predictions, cm_prices) -> np.ndarray:
    if predictions is None:
        point_error = np.inf
    else:
        point_error = np.mean(np.abs(np.array(predictions) - np.array(cm_prices)) / np.array(cm_prices))
    return point_error.item()
