import asyncio
import re
import time
from typing import Any, Callable, Optional

import bittensor as bt
import git
import numpy as np
import requests
from numpy import argsort, array, concatenate, cumsum, empty_like
from pandas import DataFrame


def get_version() -> Optional[str]:
    """Pulls the version of the precog-subnet repo from GitHub.

    Returns:
        Optional[str]: Repo version in the format: '1.1.0'
    """
    repo = git.Repo(search_parent_directories=True)
    branch_name = repo.active_branch.name
    url = f"https://github.com/coinmetrics/precog/blob/{branch_name}/precog/__init__.py"
    response = requests.get(url, timeout=10)
    if not response.ok:
        bt.logging.error("Failed to get version from GitHub")
        return None
    match = re.search(r"__version__ = (.{1,10})", response.text)

    version_match = re.search(r"\d+\.\d+\.\d+", match.group(1))
    if not version_match:
        raise Exception("Version information not found")

    return version_match.group()


def rank(vector):
    if vector is None or len(vector) <= 1:
        return array([0])
    else:
        # Sort the array and get the indices that would sort it
        sorted_indices = argsort(vector)
        sorted_vector = vector[sorted_indices]
        # Create a mask for where each new unique value starts in the sorted array
        unique_mask = concatenate(([True], sorted_vector[1:] != sorted_vector[:-1]))
        # Use cumulative sum of the unique mask to get the ranks, then assign back in original order
        ranks = cumsum(unique_mask) - 1
        rank_vector = empty_like(vector, dtype=int)
        rank_vector[sorted_indices] = ranks
        return rank_vector


def get_average_weights_for_ties(ranks, decay):
    """
    Corrected implementation that properly averages weights for tied positions.
    """
    n = len(ranks)
    base_weights = decay ** np.arange(n)

    weights = np.zeros_like(ranks, dtype=float)

    sorted_indices = np.argsort(ranks)
    sorted_ranks = ranks[sorted_indices]

    pos = 0
    while pos < n:
        current_rank = sorted_ranks[pos]

        tie_indices = np.nonzero(sorted_ranks == current_rank)[0]
        tie_size = len(tie_indices)

        start_pos = tie_indices[0]
        end_pos = start_pos + tie_size - 1

        position_weights = base_weights[start_pos : end_pos + 1]
        avg_weight = np.mean(position_weights)

        for idx in sorted_indices[tie_indices]:
            weights[idx] = avg_weight

        pos += tie_size

    return weights


async def loop_handler(self, func: Callable, sleep_time: float = 120):
    try:
        while not self.stop_event.is_set():
            async with self.lock:
                await func()
            await asyncio.sleep(sleep_time)
    except asyncio.CancelledError:
        bt.logging.error(f"{func.__name__} cancelled")
        raise
    except KeyboardInterrupt:
        raise
    except Exception as e:
        bt.logging.error(f"{func.__name__} raised error: {e}")
        raise e
    finally:
        async with self.lock:
            self.stop_event.set()


def func_with_retry(func: Callable, max_attempts: int = 3, delay: float = 1, *args, **kwargs) -> Any:
    attempt = 0
    while attempt < max_attempts:
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            attempt += 1
            bt.logging.debug(f"Function {func} failed: Attempt {attempt} of {max_attempts} with error: {e}")
            if attempt == max_attempts:
                bt.logging.error(f"Function {func} failed {max_attempts} times, skipping.")
                raise
            else:
                time.sleep(delay)


def pd_to_dict(data: DataFrame) -> dict:
    price_dict = {}
    for i in range(len(data)):
        price_dict[data.time[i].to_pydatetime()] = data.iloc[i]["ReferenceRateUSD"].item()
    return price_dict
