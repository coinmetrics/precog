import pandas as pd
from typing import Optional, Union
from datetime import datetime, date
from coinmetrics.api_client import CoinMetricsClient


class CMData:
    def __init__(self, api_key: str = "") -> None:
        self._api_key = api_key
        self._client = CoinMetricsClient(self.api_key)

    @property
    def api_key(self):
        return self._api_key

    @property
    def client(self):
        return self._client

    def get_CM_ReferenceRate(
        self,
        assets: Union[list, str],
        start: Optional[Union[datetime, date, str]] = None,
        end: Optional[Union[datetime, date, str]] = None,
        end_inclusive: bool = True,
        frequency: str = "1s",
        page_size: int = 10000,
        parallelize: bool = False,
        time_inc_parallel: pd.Timedelta = pd.Timedelta("1h"),
        **kwargs,
    ) -> pd.DataFrame:
        """Fetches CM Reference Rate for specific asset ticker or list of tickers from CoinMetrics Python client.

        Returns:
            DataFrame: Reference Rate of assets over time, with columns
                ['asset', 'time', 'ReferenceRateUSD']

        Notes:
            CM API Reference: https://coinmetrics.github.io/api-client-python/site/api_client.html#get_pair_candles
        """

        reference_rate = self.client.get_asset_metrics(
            assets,
            metrics="ReferenceRateUSD",
            start_time=start,
            end_time=end,
            end_inclusive=end_inclusive,
            frequency=frequency,
            page_size=page_size,
            **kwargs,
        )

        if parallelize:
            reference_rate_df = reference_rate.parallel(time_increment=time_inc_parallel).to_dataframe()
        else:
            reference_rate_df = reference_rate.to_dataframe()

        reference_rate_df = reference_rate_df.sort_values("time").reset_index(drop=True)
        return reference_rate_df

    def get_pair_candles(
        self,
        pairs: Union[list, str],
        start: Optional[Union[datetime, date, str]] = None,
        end: Optional[Union[datetime, date, str]] = None,
        end_inclusive: bool = True,
        frequency: str = "1h",
        page_size: int = 10000,
        parallelize: bool = False,
        time_inc_parallel: pd.Timedelta = pd.Timedelta("1d"),
        **kwargs,
    ) -> pd.DataFrame:
        """Fetches candles for specific asset pairs from CoinMetrics Python client.
            Note 'pair' must be in format {base}-{quote} (ie. pair='btc-usd')

        Returns:
            DataFrame: Available pair candles with columns:
                ['pair', 'time', 'price_open', 'price_close', 'price_high', 'price_low']

        Notes:
            CM API Reference: https://coinmetrics.github.io/api-client-python/site/api_client.html#get_pair_candles
        """

        pair_candles = self.client.get_pair_candles(
            pairs,
            start_time=start,
            end_time=end,
            end_inclusive=end_inclusive,
            frequency=frequency,
            page_size=page_size,
            **kwargs,
        )

        if parallelize:
            pair_candles_df = pair_candles.parallel(time_increment=time_inc_parallel).to_dataframe()
        else:
            pair_candles_df = pair_candles.to_dataframe()

        pair_candles_df = pair_candles_df.sort_values("time").reset_index(drop=True)
        return pair_candles_df

    def get_open_interest_catalog(self, base: str = "btc", quote: str = "usd", market_type: str = "future", **kwargs):
        """Returns the CM Catalog for active markets by base asset, quote asset, and type ('spot', 'option', or 'future')

        Args:
            base (str, optional): Base Asset of Market. Defaults to "btc".
            quote (str, optional): Quote Asset of Market. Defaults to "usd".
            market_type (str, optional): Market type ('spot', 'option', 'future'). Defaults to "spot".
            
        Returns:
            catalog (pd.DataFrame): Dataframe containing active markets with columns
                ['market', 'min_time', 'max_time']
        """
        catalog = self.client.catalog_market_open_interest_v2(
            base=base, quote=quote, market_type=market_type, page_size=10000, paging_from="end"
        ).to_dataframe()
        
        return catalog

    def get_market_open_interest(
        self, markets: list, page_size: int = 10000, parallelize=False, **kwargs
    ) -> pd.DataFrame:
        """Fetches available market open interest from CoinMetrics Python client. 
            Possible markets can be obtained from the get_open_interest_catalog() method

        Returns:
            DataFrame: Open Interest of unsettled derivatives contracts. Columns are:
                [market, time, contract_count, value_usd, database_time, exchange_time]

        Notes:
            CM API Reference: https://coinmetrics.github.io/api-client-python/site/api_client.html#get_market_open_interest
        """

        market_open_interest = self.client.get_market_open_interest(markets, page_size=page_size, **kwargs)

        if parallelize:
            return market_open_interest.parallel().to_dataframe()
        else:
            return market_open_interest.to_dataframe()

    def get_market_funding_rates(self, markets: list, page_size: int = 10000, **kwargs) -> pd.DataFrame:
        """Fetches available market funding rates from CoinMetrics Python client.

        Returns:
            DataFrame: Available market funding rates

        Notes:
            CM API Reference: https://coinmetrics.github.io/api-client-python/site/api_client.html#get_market_funding_rates
        """

        market_funding_rates = self.client.get_market_funding_rates(markets, page_size=page_size, **kwargs)
        return market_funding_rates.to_dataframe()

    # def get_time_reference_rate(self):
    #     """Retrieves reference rates for the 'btc' asset from the CoinMetrics API.

    #     Returns:
    #     - api_time (float): The time taken to make the API call in seconds.
    #     - df_time (float): The time taken to create the DataFrame from the API response in seconds.
    #     - total_time (float): The total time taken for the API call and DataFrame creation in seconds.
    #     - num_points (int): The number of data points in the DataFrame.
    #     """
    #     api_start_time = time.time()
    #     reference_rates = self.client.get_asset_metrics(
    #         assets="btc", metrics="ReferenceRateUSD", frequency="1s", page_size=10000
    #     )

    #     api_end_time = time.time()
    #     api_time = api_end_time - api_start_time

    #     df_start_time = time.time()
    #     # df = reference_rates.export_to_csv()
    #     # df = reference_rates.export_to_json()
    #     df = reference_rates.to_dataframe()
    #     print(df)
    #     df_end_time = time.time()
    #     df_time = df_end_time - df_start_time

    #     total_time = api_time + df_time
    #     return api_time, df_time, total_time, len(df)

    # def dump(self):
    #     """Runs the 'time_reference_rate_api' method multiple times and calculates the average times.

    #     Prints the API call time, DataFrame creation time, total time, and number of data points for each run.
    #     Prints the average times for the API call, DataFrame creation, total time, and number of data points.
    #     """
    #     # Number of times to run the API call
    #     num_runs = 5

    #     # Lists to store times
    #     api_times = []
    #     df_times = []
    #     total_times = []
    #     data_points = []

    #     for i in range(num_runs):
    #         print(f"\nRun {i+1}/{num_runs}")
    #         api_time, df_time, total_time, num_points = self.time_reference_rate_api()
    #         api_times.append(api_time)
    #         df_times.append(df_time)
    #         total_times.append(total_time)
    #         data_points.append(num_points)
    #         print(f"API call time: {api_time:.2f} seconds")
    #         print(f"DataFrame creation time: {df_time:.2f} seconds")
    #         print(f"Total time: {total_time:.2f} seconds")
    #         print(f"Number of data points: {num_points}")

    #     # Calculate averages
    #     avg_api_time = statistics.mean(api_times)
    #     avg_df_time = statistics.mean(df_times)
    #     avg_total_time = statistics.mean(total_times)
    #     avg_data_points = statistics.mean(data_points)

    #     print("\nAverage times:")
    #     print(f"API call: {avg_api_time:.2f} seconds")
    #     print(f"DataFrame creation: {avg_df_time:.2f} seconds")
    #     print(f"Total: {avg_total_time:.2f} seconds")
    #     print(f"Average number of data points: {avg_data_points:.0f}")
