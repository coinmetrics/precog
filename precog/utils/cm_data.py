from datetime import date, datetime
from typing import Optional, Union

import pandas as pd
from coinmetrics.api_client import CoinMetricsClient


class CMData:
    def __init__(self, api_key: str = "") -> None:
        self._api_key = api_key
        self._client = CoinMetricsClient(self.api_key)
        self._cache = pd.DataFrame()
        self._last_update = None

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
        use_cache: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """Fetches CM Reference Rate for specific asset ticker or list of tickers from CoinMetrics Python client.

        Args:
            assets (Union[list, str]): Asset ticker or list of tickers to retrieve CM Reference Rates for
            start (Optional[Union[datetime, date, str]], optional): Start time of data, if None will return earliest available. Defaults to None.
            end (Optional[Union[datetime, date, str]], optional): End time of data, if None will return earliest available. Defaults to None.
            end_inclusive (bool, optional): Whether to include a data point occuring at the "end" time. Defaults to True.
            frequency (str, optional): Frequency of prices - '200ms', '1s', '1m', '1m', '1d'. Defaults to "1s".
            page_size (int, optional): Page size of return, recommended 10000. Defaults to 10000.
            parallelize (bool, optional): Whether to parallelize query into multiple queries.
                Can speed up retrieval but may go over usage limits. Defaults to False.
            time_inc_parallel (pd.Timedelta, optional): If using parallelize, time interval queried by each thread. Defaults to pd.Timedelta("1h").
            use_cache (bool, optional): Whether to use cached data. Defaults to True.

        Returns:
            pd.DataFrame: Reference Rate of assets over time, with columns
                ['asset', 'time', 'ReferenceRateUSD']

        Notes:
            CM API Reference: https://coinmetrics.github.io/api-client-python/site/api_client.html#get_pair_candles
        """
        if not use_cache:
            return self._fetch_reference_rate(
                assets, start, end, end_inclusive, frequency, page_size, parallelize, time_inc_parallel, **kwargs
            )

        end_time = pd.to_datetime(end)

        if self._cache.empty or "asset" not in self._cache.columns:
            self._cache = self._fetch_reference_rate(
                assets, start, end, end_inclusive, frequency, page_size, parallelize, time_inc_parallel, **kwargs
            )
            self._last_update = end_time
            return self._cache

        latest_cached = self._cache["time"].max()
        if end_time > latest_cached:
            new_data = self._fetch_reference_rate(
                assets,
                latest_cached,
                end,
                end_inclusive,
                frequency,
                page_size,
                parallelize,
                time_inc_parallel,
                **kwargs,
            )

            self._cache = pd.concat([self._cache, new_data]).drop_duplicates(subset=["time"])
            self._cache.sort_values("time", inplace=True)
            self._last_update = end_time

        # Remove data older than 24 hours from latest point
        cutoff_time = end_time - pd.Timedelta(days=1)
        self._cache = self._cache[self._cache["time"] >= cutoff_time]

        # Filter data for requested time range
        if start:
            start_time = pd.to_datetime(start)
            return self._cache[(self._cache["time"] >= start_time) & (self._cache["time"] <= end_time)]

        return self._cache[self._cache["time"] <= end_time]

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

        Args:
            markets (list): List of derivatives markets to get the Open Interest for.
            Note there is a character limit to the query, so may need to be done in chunks for a long list

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

    def _fetch_reference_rate(
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
        """Internal method to fetch reference rate data from CM API"""
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

        return reference_rate_df.sort_values("time").reset_index(drop=True)

    def clear_cache(self):
        """Clear the cache if needed"""
        self._cache = pd.DataFrame()
        self._last_update = None
