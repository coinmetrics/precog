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
    interval_errors = []
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

        if uid == 30:
            bt.logging.debug(f"\nDebugging interval evaluation for UID {uid}:")
            bt.logging.debug(f"Number of aligned intervals: {len(inters) if inters is not None else 0}")
            bt.logging.debug(
                f"Number of aligned interval prices: {len(interval_prices) if interval_prices is not None else 0}"
            )

        if any([np.isnan(inters).any(), np.isnan(interval_prices).any()]):
            interval_errors.append(0)
        else:
            base_interval_error = interval_error(inters, interval_prices, uid=uid, timestamps=aligned_int_timestamps)
            adjusted_interval_error = base_interval_error * completeness_ratio
            interval_errors.append(adjusted_interval_error)

        # bt.logging.debug(f"UID: {uid} | point_errors: {point_errors[-1]} | interval_errors: {interval_errors[-1]}")

    point_ranks = rank(np.array(point_errors))
    interval_ranks = rank(-np.array(interval_errors))  # 1 is best, 0 is worst, so flip it

    point_weights = get_average_weights_for_ties(point_ranks, decay)
    interval_weights = get_average_weights_for_ties(interval_ranks, decay)

    base_rewards = (point_weights + interval_weights) / 2
    rewards = base_rewards * np.array(completeness_scores)

    return rewards


def interval_error(intervals, cm_prices, uid=None, timestamps=None):  # noqa: C901
    if intervals is None:
        return np.array([0])
    else:
        # Only log for UID 30
        should_log = uid == 30

        # Calculate expected intervals based on constants
        intervals_per_hour = 60 / constants.PREDICTION_INTERVAL_MINUTES  # Should be 12
        prediction_window = int(constants.PREDICTION_FUTURE_HOURS * intervals_per_hour)  # Should be 12

        if should_log:
            bt.logging.debug("=" * 50)
            bt.logging.debug(f"INTERVAL ERROR COMPARISON FOR UID {uid}")
            bt.logging.debug("Constants:")
            bt.logging.debug(f"  - PREDICTION_INTERVAL_MINUTES: {constants.PREDICTION_INTERVAL_MINUTES}")
            bt.logging.debug(f"  - PREDICTION_FUTURE_HOURS: {constants.PREDICTION_FUTURE_HOURS}")
            bt.logging.debug(f"  - Calculated intervals_per_hour: {intervals_per_hour}")
            bt.logging.debug(f"  - Calculated prediction_window: {prediction_window}")
            bt.logging.debug("Input data:")
            bt.logging.debug(f"  - Number of intervals: {len(intervals)}")
            bt.logging.debug(f"  - Number of prices: {len(cm_prices)}")
            bt.logging.debug("=" * 50)

        old_interval_errors = []
        new_interval_errors = []

        for i, interval_to_evaluate in enumerate(intervals[:-1]):
            lower_bound_prediction = np.min(interval_to_evaluate)
            upper_bound_prediction = np.max(interval_to_evaluate)

            # OLD METHOD - evaluating against all future prices
            old_prices_slice = cm_prices[i + 1 :]
            old_slice_size = len(old_prices_slice)

            # NEW METHOD - only next prediction_window prices
            new_end_index = min(i + 1 + prediction_window, len(cm_prices))
            new_prices_slice = cm_prices[i + 1 : new_end_index]
            new_slice_size = len(new_prices_slice)

            # Skip if we don't have enough future data for new method
            if new_slice_size < prediction_window:
                if should_log:
                    bt.logging.debug(f"\nInterval {i}: Skipping - only {new_slice_size} future prices available")
                # For old method, still calculate if there's any data
                if old_slice_size > 0:
                    old_effective_min = np.max([lower_bound_prediction, np.min(old_prices_slice)])
                    old_effective_max = np.min([upper_bound_prediction, np.max(old_prices_slice)])
                    old_f_w = (old_effective_max - old_effective_min) / (
                        upper_bound_prediction - lower_bound_prediction
                    )
                    old_f_i = sum(
                        (old_prices_slice >= lower_bound_prediction) & (old_prices_slice <= upper_bound_prediction)
                    ) / len(old_prices_slice)
                    old_interval_errors.append(old_f_w * old_f_i)
                break

            # Calculate OLD method scores
            old_effective_min = np.max([lower_bound_prediction, np.min(old_prices_slice)])
            old_effective_max = np.min([upper_bound_prediction, np.max(old_prices_slice)])
            old_f_w = (old_effective_max - old_effective_min) / (upper_bound_prediction - lower_bound_prediction)
            old_f_i = sum(
                (old_prices_slice >= lower_bound_prediction) & (old_prices_slice <= upper_bound_prediction)
            ) / len(old_prices_slice)
            old_score = old_f_w * old_f_i
            old_interval_errors.append(old_score)

            # Calculate NEW method scores
            new_effective_min = np.max([lower_bound_prediction, np.min(new_prices_slice)])
            new_effective_max = np.min([upper_bound_prediction, np.max(new_prices_slice)])
            new_f_w = (new_effective_max - new_effective_min) / (upper_bound_prediction - lower_bound_prediction)
            new_f_i = sum(
                (new_prices_slice >= lower_bound_prediction) & (new_prices_slice <= upper_bound_prediction)
            ) / len(new_prices_slice)
            new_score = new_f_w * new_f_i
            new_interval_errors.append(new_score)

            if should_log:
                bt.logging.debug(f"\nInterval {i}:")
                if timestamps and i < len(timestamps):
                    bt.logging.debug(f"  Timestamp: {timestamps[i]}")
                bt.logging.debug(f"  Interval bounds: [{lower_bound_prediction:.2f}, {upper_bound_prediction:.2f}]")
                bt.logging.debug("  OLD METHOD:")
                bt.logging.debug(f"    - Evaluating against {old_slice_size} future prices")
                bt.logging.debug(f"    - Price range: [{np.min(old_prices_slice):.2f}, {np.max(old_prices_slice):.2f}]")
                bt.logging.debug(f"    - Effective range: [{old_effective_min:.2f}, {old_effective_max:.2f}]")
                bt.logging.debug(f"    - f_w: {old_f_w:.4f}, f_i: {old_f_i:.4f}, score: {old_score:.4f}")
                bt.logging.debug("  NEW METHOD:")
                bt.logging.debug(f"    - Evaluating against {new_slice_size} future prices")
                bt.logging.debug(f"    - Price range: [{np.min(new_prices_slice):.2f}, {np.max(new_prices_slice):.2f}]")
                bt.logging.debug(f"    - Effective range: [{new_effective_min:.2f}, {new_effective_max:.2f}]")
                bt.logging.debug(f"    - f_w: {new_f_w:.4f}, f_i: {new_f_i:.4f}, score: {new_score:.4f}")
                bt.logging.debug(
                    f"  DIFFERENCE: {new_score - old_score:.4f} ({((new_score - old_score) / old_score * 100):.1f}%)"
                )

        # Calculate mean errors
        old_mean_error = np.nanmean(np.array(old_interval_errors)).item() if old_interval_errors else 0
        new_mean_error = np.nanmean(np.array(new_interval_errors)).item() if new_interval_errors else 0

        if should_log:
            bt.logging.debug("\n" + "=" * 50)
            bt.logging.debug("FINAL COMPARISON:")
            bt.logging.debug("OLD METHOD:")
            bt.logging.debug(f"  - Intervals evaluated: {len(old_interval_errors)}")
            bt.logging.debug(f"  - Mean score: {old_mean_error:.4f}")
            bt.logging.debug("NEW METHOD:")
            bt.logging.debug(f"  - Intervals evaluated: {len(new_interval_errors)}")
            bt.logging.debug(f"  - Mean score: {new_mean_error:.4f}")
            bt.logging.debug(
                f"DIFFERENCE: {new_mean_error - old_mean_error:.4f} ({((new_mean_error - old_mean_error) / old_mean_error * 100):.1f}%)"
            )
            bt.logging.debug("=" * 50)

        # Return the OLD method for now (to maintain current behavior)
        # Change this to new_mean_error when ready to switch
        return old_mean_error


def point_error(predictions, cm_prices) -> np.ndarray:
    if predictions is None:
        point_error = np.inf
    else:
        point_error = np.mean(np.abs(np.array(predictions) - np.array(cm_prices)) / np.array(cm_prices))
    return point_error.item()
