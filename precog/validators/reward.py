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
    # preallocate
    point_errors = []
    interval_errors = []
    decay = 0.9
    weights = np.linspace(0, len(self.available_uids) - 1, len(self.available_uids))
    decayed_weights = decay**weights
    timestamp = responses[0].timestamp
    cm = CMData()
    start_time: str = to_str(get_before(timestamp=timestamp, hours=1))
    end_time: str = to_str(to_datetime(timestamp))  # built-ins handle CM API's formatting
    # Query CM API for sample standard deviation of the 1s residuals
    historical_price_data: DataFrame = cm.get_CM_ReferenceRate(
        assets="BTC", start=start_time, end=end_time, frequency="1s"
    )
    cm_data = pd_to_dict(historical_price_data)
    for uid, response in zip(self.available_uids, responses):
        current_miner = self.MinerHistory[uid]
        self.MinerHistory[uid].add_prediction(response.timestamp, response.prediction, response.interval)
        prediction_dict, interval_dict = current_miner.format_predictions(response.timestamp)
        mature_time_dict = mature_dictionary(prediction_dict)
        preds, price, aligned_pred_timestamps = align_timepoints(mature_time_dict, cm_data)
        inters, interval_prices, aligned_int_timestamps = align_timepoints(interval_dict, cm_data)

        # Calculate point error metrics
        if preds is not None and len(preds) > 0:
            absolute_errors = np.abs(np.array(preds) - np.array(price))
            relative_errors = absolute_errors / np.array(price)
            current_point_error = np.mean(relative_errors)
        else:
            current_point_error = np.inf

        # Calculate interval error metrics
        if not any([np.isnan(inters).any(), np.isnan(interval_prices).any()]) and len(inters) > 0:
            lower_bound = np.min(inters[0])
            upper_bound = np.max(inters[0])
            future_prices = interval_prices[1:]
            
            effective_min = np.max([lower_bound, np.min(future_prices)])
            effective_max = np.min([upper_bound, np.max(future_prices)])
            f_w = (effective_max - effective_min) / (upper_bound - lower_bound)
            
            inside_mask = (future_prices >= lower_bound) & (future_prices <= upper_bound)
            percent_inside = (np.sum(inside_mask) / len(future_prices)) * 100
            current_interval_error = f_w * (percent_inside / 100)
        else:
            current_interval_error = 0

        bt.logging.debug("")  # Add blank line
        bt.logging.debug(f"""uid: {uid}
timestamp: {aligned_pred_timestamps[0] if aligned_pred_timestamps else 'N/A'}

Point Prediction Metrics:
prediction: {preds[0] if preds is not None and len(preds) > 0 else 'N/A'}
actual_price: {price[0] if len(price) > 0 else 'N/A'}
absolute_error: {absolute_errors[0]:.2f if preds is not None and len(preds) > 0 else 'N/A'}
relative_error: {relative_errors[0]:.4f if preds is not None and len(preds) > 0 else 'N/A'}
point_error_score: {current_point_error:.4f}

Interval Prediction Metrics:
upper_bound: {upper_bound if 'upper_bound' in locals() else 'N/A'}
lower_bound: {lower_bound if 'lower_bound' in locals() else 'N/A'}
price_max_interval: {np.max(future_prices) if 'future_prices' in locals() else 'N/A'}
price_min_interval: {np.min(future_prices) if 'future_prices' in locals() else 'N/A'}
percent_inside: {percent_inside:.2f if 'percent_inside' in locals() else 'N/A'}%
percent_outside: {(100 - percent_inside):.2f if 'percent_inside' in locals() else 'N/A'}%
width_factor: {f_w:.4f if 'f_w' in locals() else 'N/A'}
interval_error_score: {current_interval_error:.4f}""")

        point_errors.append(current_point_error)
        interval_errors.append(current_interval_error)

    point_ranks = rank(np.array(point_errors))
    interval_ranks = rank(-np.array(interval_errors))  # 1 is best, 0 is worst, so flip it
    rewards = (decayed_weights[point_ranks] + decayed_weights[interval_ranks]) / 2
    return rewards


def interval_error(intervals, cm_prices, timestamps=None):
    if intervals is None:
        return np.array([0])
    else:
        interval_errors = []
        for i, interval_to_evaluate in enumerate(intervals[:-1]):
            ts = timestamps[i] if timestamps is not None else f"interval_{i}"
            
            lower_bound_prediction = np.min(interval_to_evaluate)
            upper_bound_prediction = np.max(interval_to_evaluate)
            future_prices = cm_prices[i + 1:]

            effective_min = np.max([lower_bound_prediction, np.min(future_prices)])
            effective_max = np.min([upper_bound_prediction, np.max(future_prices)])
            f_w = (effective_max - effective_min) / (upper_bound_prediction - lower_bound_prediction)
            # print(f"f_w: {f_w} | t: {effective_max} | b: {effective_min} | _pmax: {upper_bound_prediction} | _pmin: {lower_bound_prediction}")
            inside_mask = (future_prices >= lower_bound_prediction) & (future_prices <= upper_bound_prediction)
            percent_inside = (np.sum(inside_mask) / len(future_prices)) * 100
            percent_outside = 100 - percent_inside
            
            f_i = percent_inside / 100

            interval_errors.append(f_w * f_i)
            # print(f"lower: {lower_bound_prediction} | upper: {upper_bound_prediction} | cm_prices: {cm_prices[i:]} | error: {f_w * f_i}")
        if len(interval_errors) == 1:
            mean_error = interval_errors[0]
        else:
            mean_error = np.nanmean(np.array(interval_errors)).item()
        return mean_error


def point_error(predictions, cm_prices) -> np.ndarray:
    if predictions is None:
        point_error = np.inf
    else:
        absolute_errors = np.abs(np.array(predictions) - np.array(cm_prices))
        relative_errors = absolute_errors / np.array(cm_prices)
        point_error = np.mean(relative_errors)
    return point_error.item()
