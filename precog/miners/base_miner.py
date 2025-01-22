from typing import Tuple

import bittensor as bt
import pandas as pd

from precog.protocol import Challenge
from precog.utils.cm_data import CMData
from precog.utils.timestamp import get_before, to_datetime, to_str
import time


def get_point_estimate(timestamp: str) -> float:
   start = time.perf_counter()
   
   cm = CMData()
   init_time = time.perf_counter() - start
   
   provided_timestamp = to_datetime(timestamp) 
   parse_time = time.perf_counter() - start - init_time
   
   price_data = cm.get_CM_ReferenceRate(
       assets="BTC", start=None, end=to_str(provided_timestamp), 
       frequency="1s", limit_per_asset=1, paging_from="end"
   )
   api_time = time.perf_counter() - start - init_time - parse_time
   
   btc_price = float(price_data["ReferenceRateUSD"].iloc[-1])
   process_time = time.perf_counter() - start - init_time - parse_time - api_time
   
   bt.logging.debug(f"Point estimate times - Init: {init_time:.3f}s, Parse: {parse_time:.3f}s, " 
                   f"API: {api_time:.3f}s, Process: {process_time:.3f}s")
   return btc_price

def get_prediction_interval(timestamp: str, point_estimate: float) -> Tuple[float, float]:
   start = time.perf_counter()
   
   cm = CMData()
   init_time = time.perf_counter() - start

   start_time = get_before(timestamp, days=1, minutes=0, seconds=0)
   end_time = to_datetime(timestamp)
   parse_time = time.perf_counter() - start - init_time
   
   historical_price_data = cm.get_CM_ReferenceRate(
       assets="BTC", start=to_str(start_time), end=to_str(end_time), frequency="1s"
   )
   api_time = time.perf_counter() - start - init_time - parse_time
   
   residuals = historical_price_data["ReferenceRateUSD"].diff()
   sample_std_dev = float(residuals.std())
   time_steps = 3600
   naive_forecast_std_dev = sample_std_dev * (time_steps**0.5)
   coefficient = 1.64
   lower_bound = point_estimate - coefficient * naive_forecast_std_dev
   upper_bound = point_estimate + coefficient * naive_forecast_std_dev
   
   calc_time = time.perf_counter() - start - init_time - parse_time - api_time
   
   bt.logging.debug(f"Interval times - Init: {init_time:.3f}s, Parse: {parse_time:.3f}s, "
                   f"API: {api_time:.3f}s, Calc: {calc_time:.3f}s")
   return lower_bound, upper_bound


def forward(synapse: Challenge) -> Challenge:
   start_time = time.perf_counter()
   
   bt.logging.info(
       f"👈 Received prediction request from: {synapse.dendrite.hotkey} for timestamp: {synapse.timestamp}"
   )

   point_estimate: float = get_point_estimate(timestamp=synapse.timestamp)
   point_time = time.perf_counter() - start_time
   
   prediction_interval: Tuple[float, float] = get_prediction_interval(
       timestamp=synapse.timestamp, point_estimate=point_estimate
   )
   interval_time = time.perf_counter() - start_time - point_time

   synapse.prediction = point_estimate
   synapse.interval = prediction_interval

   total_time = time.perf_counter() - start_time
   bt.logging.info(f"Times - Point: {point_time:.3f}s, Interval: {interval_time:.3f}s, Total: {total_time:.3f}s")

   if synapse.prediction is not None:
       bt.logging.success(f"Predicted price: {synapse.prediction}  |  Predicted Interval: {synapse.interval}")
   else:
       bt.logging.info("No prediction for this request.")
   return synapse
