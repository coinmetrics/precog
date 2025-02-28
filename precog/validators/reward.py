from typing import List

import bittensor as bt
import numpy as np
from pandas import DataFrame

from precog.protocol import Challenge
from precog.utils.cm_data import CMData
from precog.utils.general import pd_to_dict, rank
from precog.utils.timestamp import align_timepoints, get_before, mature_dictionary, to_datetime, to_str


################################################################################
def calc_rewards(
    self,
    responses: List[Challenge],
) -> np.ndarray:
    evaluation_window_hours = self.config.evaluation_window
    # preallocate
    point_errors = []
    interval_errors = []
    decay = 0.9
    weights = np.linspace(0, len(self.available_uids) - 1, len(self.available_uids))
    decayed_weights = decay**weights
    timestamp = responses[0].timestamp
    cm = CMData()
    start_time: str = to_str(get_before(timestamp=timestamp, hours=evaluation_window_hours))
    end_time: str = to_str(to_datetime(timestamp))  # built-ins handle CM API's formatting
    # Query CM API for sample standard deviation of the 1s residuals
    historical_price_data: DataFrame = cm.get_CM_ReferenceRate(
        assets="BTC", start=start_time, end=end_time, frequency="1s"
    )
    cm_data = pd_to_dict(historical_price_data)

    # TODO: Do we need to set a minimum amount of intervals for evaluation if we are increasing the evaluation window?
    for uid, response in zip(self.available_uids, responses):
        current_miner = self.MinerHistory[uid]
        self.MinerHistory[uid].add_prediction(response.timestamp, response.prediction, response.interval)
        prediction_dict, interval_dict = current_miner.format_predictions(
            response.timestamp, hours=evaluation_window_hours
        )
        mature_time_dict = mature_dictionary(prediction_dict, hours=evaluation_window_hours)
        preds, price, aligned_pred_timestamps = align_timepoints(mature_time_dict, cm_data)
        for i, j, k in zip(preds, price, aligned_pred_timestamps):
            bt.logging.debug(f"Prediction: {i} | Price: {j} | Aligned Prediction: {k}")
        inters, interval_prices, aligned_int_timestamps = align_timepoints(interval_dict, cm_data)
        for i, j, k in zip(inters, interval_prices, aligned_int_timestamps):
            bt.logging.debug(f"Interval: {i} | Interval Price: {j} | Aligned TS: {k}")
        point_errors.append(point_error(preds, price))
        try:
            if (
                any([np.isnan(inters).any(), np.isnan(interval_prices).any()])
                or len(inters) == 0
                or len(interval_prices) == 0
            ):
                interval_errors.append(np.inf)
            else:
                interval_errors.append(interval_error(inters, interval_prices))
        except Exception:
            interval_errors.append(np.inf)
        bt.logging.debug(f"UID: {uid} | point_errors: {point_errors[-1]} | interval_errors: {interval_errors[-1]}")

    point_ranks = rank(np.array(point_errors))
    interval_ranks = rank(-np.array(interval_errors))  # 1 is best, 0 is worst, so flip it
    rewards = (decayed_weights[point_ranks] + decayed_weights[interval_ranks]) / 2
    return rewards


def interval_error(intervals, cm_prices):
    if intervals is None or len(intervals) <= 1:
        return np.inf

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
        return np.inf

    return np.nanmean(error_array).item()


def point_error(predictions, cm_prices) -> np.ndarray:
    if predictions is None or len(predictions) == 0 or len(cm_prices) == 0:
        return np.inf
    else:
        point_error = np.mean(np.abs(np.array(predictions) - np.array(cm_prices)) / np.array(cm_prices))
    return point_error.item()
