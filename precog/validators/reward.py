from typing import List

import bittensor as bt
import numpy as np
from pandas import DataFrame

from precog import constants
from precog.protocol import Challenge
from precog.utils.cm_data import CMData
from precog.utils.general import get_average_weights_for_ties, pd_to_dict, rank
from precog.utils.timestamp import get_before, to_datetime, to_str


################################################################################
def calc_rewards(
    self,
    responses: List[Challenge],
) -> np.ndarray:
    prediction_future_hours = constants.PREDICTION_FUTURE_HOURS

    # preallocate
    point_errors = []
    interval_scores = []
    decay = 0.9
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

    bt.logging.info(f"Fetching CM data from {start_time} to {end_time} for interval evaluation")

    # Query CM API for price data
    historical_price_data: DataFrame = cm.get_CM_ReferenceRate(
        assets="BTC", start=start_time, end=end_time, frequency="1s"
    )
    cm_data = pd_to_dict(historical_price_data)
    bt.logging.info(f"CM data fetched: {len(cm_data)} price points")

    for uid, response in zip(self.available_uids, responses):
        current_miner = self.MinerHistory[uid]
        self.MinerHistory[uid].add_prediction(response.timestamp, response.prediction, response.interval)

        # Get single prediction made at prediction_time (1 hour ago)
        if prediction_time not in current_miner.predictions:
            point_errors.append(np.inf)  # No prediction penalty
            bt.logging.debug(f"UID: {uid} | No prediction found at {prediction_time}")
        else:
            prediction_value = current_miner.predictions[prediction_time]

            # Get actual price at eval_time
            if eval_time not in cm_data:
                point_errors.append(np.inf)  # No price data penalty
                bt.logging.debug(f"UID: {uid} | No price data at {eval_time}")
            else:
                actual_price = cm_data[eval_time]
                current_point_error = abs(prediction_value - actual_price) / actual_price
                point_errors.append(current_point_error)
                bt.logging.debug(
                    f"UID: {uid} | Prediction: {prediction_value} | Actual: {actual_price} | Error: {current_point_error}"
                )

        # Get single interval prediction made at prediction_time
        if prediction_time not in current_miner.intervals:
            interval_scores.append(0)  # No interval prediction
            bt.logging.debug(f"UID: {uid} | No interval prediction found at {prediction_time}")
        else:
            interval_bounds = current_miner.intervals[prediction_time]

            # Evaluate interval over the past hour (from prediction_time to eval_time)
            hour_prices = []

            # Collect all price points in the predicted hour
            for price_time, price_value in cm_data.items():
                if prediction_time <= price_time <= eval_time:
                    hour_prices.append(price_value)

            if not hour_prices:
                interval_scores.append(0)  # No price data for interval evaluation
                bt.logging.debug(f"UID: {uid} | No price data for interval evaluation")
            else:
                # Calculate interval score using both width factor and inclusion factor
                bt.logging.debug(f"UID: {uid} | Raw interval_bounds: {interval_bounds}")
                bt.logging.debug(f"UID: {uid} | Number of hour_prices: {len(hour_prices)}")
                pred_min = min(interval_bounds)
                pred_max = max(interval_bounds)

                # Get observed min and max prices
                observed_min = min(hour_prices)
                observed_max = max(hour_prices)
                bt.logging.debug(
                    f"UID: {uid} | Predicted: [{pred_min}, {pred_max}], Observed: [{observed_min}, {observed_max}]"
                )

                # Calculate effective top and bottom
                effective_top = min(pred_max, observed_max)
                effective_bottom = max(pred_min, observed_min)
                bt.logging.debug(f"UID: {uid} | Effective top: {effective_top}, Effective bottom: {effective_bottom}")

                # Calculate width factor (f_w)
                if pred_max == pred_min:
                    width_factor = 0  # Invalid interval
                else:
                    width_factor = (effective_top - effective_bottom) / (pred_max - pred_min)
                    bt.logging.debug(
                        f"UID: {uid} | Width calculation: ({effective_top} - {effective_bottom}) / ({pred_max} - {pred_min}) = {width_factor}"
                    )

                # Calculate inclusion factor (f_i)
                prices_in_bounds = sum(1 for price in hour_prices if pred_min <= price <= pred_max)
                inclusion_factor = prices_in_bounds / len(hour_prices)

                # Final interval score is the product
                interval_score_value = inclusion_factor * width_factor
                interval_scores.append(interval_score_value)

                bt.logging.debug(
                    f"UID: {uid} | Interval: [{pred_min}, {pred_max}] | "
                    f"Width Factor: {width_factor:.3f} | Inclusion Factor: {inclusion_factor:.3f} | "
                    f"Score: {interval_score_value:.3f}"
                )

        bt.logging.debug(f"UID: {uid} | point_errors: {point_errors[-1]} | interval_scores: {interval_scores[-1]}")

    point_ranks = rank(np.array(point_errors))
    interval_ranks = rank(-np.array(interval_scores))  # 1 is best, 0 is worst, so flip it

    point_weights = get_average_weights_for_ties(point_ranks, decay)
    interval_weights = get_average_weights_for_ties(interval_ranks, decay)
    bt.logging.trace(f"point_weights: {point_weights}")
    bt.logging.trace(f"interval_weights: {point_weights}")

    rewards = (point_weights + interval_weights) / 2
    bt.logging.trace(f"base_rewards: {rewards}")

    return rewards
