from datetime import timedelta
from typing import Dict, List

from precog.utils.timestamp import get_now, get_timezone, round_minute_down, to_datetime


class MinerHistory:
    """This class is used to store miner predictions along with their timestamps.
    Allows for easy formatting, filtering, and lookup of predictions by timestamp.
    Now supports multiple assets per timestamp.
    """

    def __init__(self, uid: int, timezone=get_timezone()):
        self.predictions = {}  # {timestamp: {asset: prediction}}
        self.intervals = {}  # {timestamp: {asset: interval}}
        self.uid = uid
        self.timezone = timezone

    def add_prediction(self, timestamp, predictions: Dict[str, float], intervals: Dict[str, List[float]]):
        if isinstance(timestamp, str):
            timestamp = to_datetime(timestamp)
        timestamp = round_minute_down(timestamp)

        if predictions:
            self.predictions[timestamp] = predictions.copy()
        if intervals:
            self.intervals[timestamp] = intervals.copy()

    def clear_old_predictions(self):
        # deletes predictions older than 24 hours
        start_time = round_minute_down(get_now()) - timedelta(hours=24)
        filtered_pred_dict = {key: value for key, value in self.predictions.items() if start_time <= key}
        self.predictions = filtered_pred_dict
        filtered_interval_dict = {key: value for key, value in self.intervals.items() if start_time <= key}
        self.intervals = filtered_interval_dict

    def format_predictions(self, reference_timestamp=None, hours: int = 1):
        """
        Filter and format prediction and interval data based on a reference timestamp and time window.

        This function filters the prediction and interval dictionaries to include only entries
        within a specified time window, ending at the reference timestamp and extending back
        by the specified number of hours.

        Parameters:
        -----------
        reference_timestamp : datetime or str, optional
            The end timestamp for the time window. If None, the current time rounded down
            to the nearest minute is used. If a string is provided, it will be converted
            to a datetime object.
        hours : int, default=1
            The number of hours to look back from the reference timestamp.

        Returns:
        --------
        tuple
            A tuple containing two dictionaries:
            - filtered_pred_dict: Dictionary of filtered predictions where keys are timestamps
            and values are the corresponding prediction values.
            - filtered_interval_dict: Dictionary of filtered intervals where keys are timestamps
            and values are the corresponding interval values.

        Notes:
        ------
        The actual time window used is (hours + 1) to ensure complete coverage of the requested period.
        """
        if reference_timestamp is None:
            reference_timestamp = round_minute_down(get_now())
        if isinstance(reference_timestamp, str):
            reference_timestamp = to_datetime(reference_timestamp)

        start_time = round_minute_down(reference_timestamp) - timedelta(hours=hours + 1)

        filtered_pred_dict = {
            key: value for key, value in self.predictions.items() if start_time <= key <= reference_timestamp
        }
        filtered_interval_dict = {
            key: value for key, value in self.intervals.items() if start_time <= key <= reference_timestamp
        }

        return filtered_pred_dict, filtered_interval_dict
