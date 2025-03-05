from datetime import timedelta
from typing import List

from precog.utils.timestamp import get_now, get_timezone, round_minute_down, to_datetime


class MinerHistory:
    """This class is used to store miner predictions along with their timestamps.
    Allows for easy formatting, filtering, and lookup of predictions by timestamp.
    """

    def __init__(self, uid: int, timezone=get_timezone()):
        self.predictions = {}
        self.intervals = {}
        self.uid = uid
        self.timezone = timezone

    def add_prediction(self, timestamp, prediction: float, interval: List[float]):
        if isinstance(timestamp, str):
            timestamp = to_datetime(timestamp)
        timestamp = round_minute_down(timestamp)
        if prediction is not None:
            self.predictions[timestamp] = prediction
        if interval is not None:
            self.intervals[timestamp] = interval

    def clear_old_predictions(self):
        # deletes predictions older than 24 hours
        start_time = round_minute_down(get_now()) - timedelta(hours=24)
        filtered_pred_dict = {key: value for key, value in self.predictions.items() if start_time <= key}
        self.predictions = filtered_pred_dict
        filtered_interval_dict = {key: value for key, value in self.intervals.items() if start_time <= key}
        self.intervals = filtered_interval_dict

    def format_predictions(self, reference_timestamp=None, hours: int = 1, prediction_interval_minutes: int = 5):
        if reference_timestamp is None:
            reference_timestamp = round_minute_down(get_now())
        if isinstance(reference_timestamp, str):
            reference_timestamp = to_datetime(reference_timestamp)

        # Round reference timestamp down to nearest interval
        reference_timestamp = round_minute_down(reference_timestamp)

        # Calculate start time
        start_time = round_minute_down(reference_timestamp) - timedelta(hours=hours + 1)

        # Filter actual predictions made within the window
        filtered_pred_dict = {
            key: value for key, value in self.predictions.items() if start_time <= key <= reference_timestamp
        }
        filtered_interval_dict = {
            key: value for key, value in self.intervals.items() if start_time <= key <= reference_timestamp
        }

        return filtered_pred_dict, filtered_interval_dict
