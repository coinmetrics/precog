from datetime import datetime, timedelta
from typing import Optional, Union

from pytz import timezone

###############################
#           GETTERS           #
###############################


def get_timezone() -> timezone:
    """
    Set the Global shared timezone for all timestamp manipulation
    """
    return timezone("UTC")


def get_now() -> datetime:
    """
    Get the current datetime
    """
    return datetime.now(get_timezone())


def get_before(
    timestamp: Optional[Union[datetime, str, float]] = None,
    days: int = 0,
    hours: int = 0,
    minutes: int = 5,
    seconds: int = 0,
) -> datetime:
    """
    Get the datetime x minutes before now
    """
    if timestamp is None:
        timestamp = get_now()
    else:
        timestamp = to_datetime(timestamp)

    # Perform the time subtraction
    before_timestamp = timestamp - timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)

    return before_timestamp


def get_midnight() -> datetime:
    """
    Get the most recent instance of midnight
    """
    return get_now().replace(hour=0, minute=0, second=0, microsecond=0)


def get_posix() -> float:
    """
    Get the current POSIX time, seconds that have elapsed since Jan 1 1970
    """
    return to_posix(get_now())


def get_str() -> str:
    """
    Get the current timestamp as a string, convenient for requests
    """
    return to_str(get_now())


###############################
#         CONVERTERS          #
###############################


def to_posix(timestamp: Union[datetime, str, float]) -> float:
    """
    Convert datetime to seconds that have elapsed since Jan 1 1970
    """

    # Verify datetime object and convert to UTC
    utc_datetime = to_datetime(timestamp)

    # Convert to posix time float
    posix_timestamp = utc_datetime.timestamp()

    return float(posix_timestamp)


def to_str(timestamp: Union[datetime, str, float]) -> str:
    """
    Convert datetime to iso 8601 string
    """

    # Verify datetime object and convert to UTC
    utc_datetime = to_datetime(timestamp)

    # Convert to iso8601 string
    str_datetime = utc_datetime.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    return str(str_datetime)


def to_datetime(timestamp: Union[str, float]) -> datetime:
    """
    Convert iso 8601 string, or a POSIX time float, to datetime
    """
    if isinstance(timestamp, str):
        # Assume the proper iso 8601 string format is used
        # `strptime` will trigger an error as needed
        return datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=get_timezone())

    elif isinstance(timestamp, float):
        # Assume proper float value
        # `fromtimestamp` will trigger errors as needed
        return datetime.fromtimestamp(timestamp, tz=get_timezone())

    elif isinstance(timestamp, datetime):
        # Already a datetime object
        # Return as UTC
        return timestamp.astimezone(get_timezone())

    else:
        # Invalid typing
        raise TypeError(
            "Must pass a timestamp that is either a iso 8601 string, POSIX time float, or a datetime object"
        )


###############################
#          FUNCTIONS          #
###############################


def elapsed_seconds(timestamp1: datetime, timestamp2: datetime) -> float:
    """
    Absolute number of seconds between two timestamps
    """
    return abs((timestamp1 - timestamp2).total_seconds())


def round_minute_down(timestamp: datetime, base: int = 5) -> datetime:
    """
    Round the timestamp down to the nearest 5 minutes

    Example:
        >>> timestamp = datetime.now(timezone("UTC"))
        >>> timestamp
        datetime.datetime(2024, 11, 14, 18, 18, 16, 719482, tzinfo=<UTC>)
        >>> round_minute_down(timestamp)
        datetime.datetime(2024, 11, 14, 18, 15, tzinfo=<UTC>)
    """
    # Round the minute down to the nearest multiple of `base`
    correct_minute: int = timestamp.minute // base * base
    return timestamp.replace(minute=correct_minute, second=0, microsecond=0)


def is_query_time(prediction_interval: int, timestamp: str, tolerance: int = 120) -> bool:
    """
    Tolerance - in seconds, how long to allow after epoch start
    prediction_interval - in minutes, how often to predict

    Notes:
        This function will be called multiple times
        First, check that we are in a new epoch
        Then, check if we already sent a request in the current epoch
    """
    now: datetime = get_now()
    provided_timestamp: datetime = to_datetime(timestamp)

    # The provided timestamp is the last time a request was made. If this timestamp
    # is from the current epoch, we do not want to make a request. One way to check
    # this is by checking that `now` and `provided_timestamp` are more than `tolerance`
    # apart from each other. When true, this means the `provided_timestamp` is from
    # the previous epoch
    been_long_enough = elapsed_seconds(now, provided_timestamp) > tolerance

    # return false early if we already know it has not been long enough
    if not been_long_enough:
        return False

    # If it has been long enough, let's check the epoch start time
    midnight = get_midnight()
    sec_since_open = elapsed_seconds(now, midnight)

    # To check if this is a new epoch, compare the current timestamp
    # to the expected beginning of an epoch. If we are within `tolerance`
    # seconds of a new epoch, then we are willing to send a request
    sec_since_epoch_start = sec_since_open % (prediction_interval * 60)
    beginning_of_epoch = sec_since_epoch_start < tolerance

    # We know a request hasn't been sent yet, so simply return T/F based
    # on beginning of epoch
    return beginning_of_epoch


def align_timepoints(filtered_pred_dict, cm_dict):
    """Takes in a dictionary of predictions and aligns them to a list of coinmetrics prices.

    Args:
        filtered_pred_dict (dict): {datetime: float} dictionary of predictions.
        cm_data (dict): {datetime: float} dictionary of prices.


    Returns:
        aligned_pred_values (List[float]): The values in filtered_pred_dict with timestamp keys that match cm_timestamps.
        aligned_cm_data (List[float]): The values in cm_data where cm_timestamps matches the timestamps in filtered_pred_dict.
        aligned_timestamps (List[datetime]): The timestamps corresponding to the values in aligned_pred_values and aligned_cm_data.
    """
    aligned_pred_values = []
    aligned_cm_data = []
    aligned_timestamps = []
    for timestamp, value in filtered_pred_dict.items():
        if timestamp in cm_dict:
            aligned_pred_values.append(value)
            aligned_timestamps.append(timestamp)
            aligned_cm_data.append(cm_dict[timestamp])
    return aligned_pred_values, aligned_cm_data, aligned_timestamps


def mature_dictionary(dict, hours=1, minutes=0) -> dict:
    """Used to turn prediction timestamps into maturation timestamps

    Args:
        dict (dict): {datetime: value} dictionary where keys are all datetimes.
        hours (int, optional): the offset in hours to add to the datetime keys. Defaults to 1.
        minutes (int, optional): Allows for fine-tuned maturation of timestamps. Defaults to 0.

    Returns:
        dict: The same as dict but with the keys offset by hours and minutes.
    """
    new_dict = {}
    for timestamp, value in dict.items():
        new_dict[timestamp + timedelta(hours=hours, minutes=minutes)] = value
    return new_dict
