import argparse
from datetime import datetime, timedelta
from typing import List

from precog.utils.timestamp import get_now, get_timezone, round_minute_down, to_datetime


class Config:
    def __init__(self, args):
        # Add command-line arguments to the Config object
        for key, value in vars(args).items():
            setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def to_str(self):
        arguments = ""
        for key, value in vars(self).items():
            if isinstance(value, bool):
                if value:
                    arguments = arguments + f" --{key}"
            else:
                try:
                    if isinstance(value, NestedNamespace):
                        for key_2, value_2 in vars(value).items():
                            if isinstance(value_2, bool):
                                if value_2:
                                    arguments = arguments + f" --{key}.{key_2}"
                            else:
                                arguments = arguments + f" --{key}.{key_2} {value_2}"
                    else:
                        arguments = arguments + (f" --{key} {value}")
                except Exception:
                    continue
        return arguments


class NestedNamespace(argparse.Namespace):
    def __setattr__(self, name, value):
        if "." in name:
            group, name = name.split(".", 1)
            ns = getattr(self, group, NestedNamespace())
            setattr(ns, name, value)
            self.__dict__[group] = ns
        else:
            self.__dict__[name] = value

    def get(self, key, default=None):
        if "." in key:
            group, key = key.split(".", 1)
            return getattr(self, group, NestedNamespace()).get(key, default)
        return self.__dict__.get(key, default)


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

    def format_predictions(self, reference_timestamp=None, hours: int = 1):
        # intervals = []
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

    def get_relevant_timestamps(self, reference_timestamp: datetime):
        # returns a list of aligned timestamps
        # round down reference to nearest 5m
        round_down_now = round_minute_down(reference_timestamp)
        # get the timestamps for the past 12 epochs
        timestamps = [round_down_now - timedelta(minutes=5 * i) for i in range(12)]
        # remove any timestamps that are not in the dicts
        filtered_list = [
            item for item in timestamps if item in self.predictions.keys() and item in self.intervals.keys()
        ]
        return filtered_list
