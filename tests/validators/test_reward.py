import pickle
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from pandas import DataFrame

from precog import constants
from precog.protocol import Challenge
from precog.utils.classes import MinerHistory
from precog.utils.timestamp import to_str
from precog.validators.reward import calc_rewards


class TestReward(unittest.TestCase):
    """Simple sanity check tests for the calc_rewards function."""

    def setUp(self):
        """Set up test fixtures with real Bitcoin price data."""
        # Create a mock validator instance with 250 miners (realistic scale)
        self.mock_validator = MagicMock()
        self.mock_validator.available_uids = list(range(250))

        # Create MinerHistory instances for all 250 miners
        self.mock_validator.MinerHistory = {uid: MinerHistory(uid) for uid in range(250)}

        # Load real price data for all assets
        self.all_asset_data = self._load_real_price_data()

    def _load_real_price_data(self):
        """Load real price data for all supported assets."""
        all_asset_data = {}

        # Load data for each asset
        for asset in constants.SUPPORTED_ASSETS:
            test_data_path = Path(f"tests/data/{asset}_test_data.pkl")

            with open(test_data_path, "rb") as f:
                data = pickle.load(f)
                all_asset_data[asset] = data["dict"]

        # Use the first asset (BTC) for timing info
        btc_data_path = Path("tests/data/btc_test_data.pkl")
        with open(btc_data_path, "rb") as f:
            btc_data = pickle.load(f)

        self.eval_time = btc_data["eval_time"]
        self.prediction_time = btc_data["prediction_time"]
        self.timestamp_str = to_str(self.eval_time)

        return all_asset_data

    @patch("precog.validators.reward.CMData")
    @patch("precog.validators.reward.pd_to_dict")
    def test_calc_rewards(self, mock_pd_to_dict, mock_cm_data):
        """Simple test that calc_rewards works and miners with better predictions get higher rewards."""
        # Setup mock data - create a proper DataFrame that matches expected structure
        mock_cm_instance = MagicMock()
        mock_cm_data.return_value = mock_cm_instance

        # Create a mock DataFrame with the expected structure
        mock_df_data = []
        for asset in constants.SUPPORTED_ASSETS:
            for timestamp, price in self.all_asset_data[asset].items():
                mock_df_data.append({"asset": asset, "time": timestamp, "ReferenceRateUSD": price})

        mock_df = DataFrame(mock_df_data)
        mock_cm_instance.get_CM_ReferenceRate.return_value = mock_df
        mock_pd_to_dict.side_effect = lambda df: self.all_asset_data[df["asset"].iloc[0]]

        # Get actual prices for all assets at eval time
        actual_prices = {asset: self.all_asset_data[asset][self.eval_time] for asset in constants.SUPPORTED_ASSETS}

        # Create 250 miners: first 3 with known different prediction qualities for testing
        responses = []

        for uid in range(250):
            predictions = {}
            intervals = {}

            for asset in constants.SUPPORTED_ASSETS:
                actual_price = actual_prices[asset]

                if uid == 0:  # Perfect prediction
                    prediction = actual_price
                    interval = [actual_price * 0.99, actual_price * 1.01]  # 1% range around actual
                elif uid == 1:  # Poor prediction (off by ~5%)
                    prediction = actual_price * 0.95
                    interval = [actual_price * 0.93, actual_price * 0.97]
                elif uid == 2:  # Terrible prediction (off by ~20%)
                    prediction = actual_price * 0.8
                    interval = [actual_price * 0.75, actual_price * 0.85]
                else:  # Random predictions for remaining miners (within reasonable range)
                    noise_factor = np.random.normal(1.0, 0.05)  # 5% std dev
                    prediction = actual_price * noise_factor
                    interval_width = actual_price * 0.02  # 2% range
                    interval = [prediction - interval_width, prediction + interval_width]

                predictions[asset] = prediction
                intervals[asset] = interval

            # Create response - using supported assets from constants
            responses.append(
                Challenge(
                    timestamp=self.timestamp_str,
                    assets=constants.SUPPORTED_ASSETS,
                    predictions=predictions,
                    intervals=intervals,
                )
            )

            # Pre-populate MinerHistory with predictions from 1 hour ago
            # The reward calculation expects to find predictions at prediction_time
            self.mock_validator.MinerHistory[uid].add_prediction(to_str(self.prediction_time), predictions, intervals)

        # Calculate rewards
        rewards = calc_rewards(self.mock_validator, responses)

        # Basic structure checks
        self.assertEqual(len(rewards), 250, "Should return rewards for all 250 miners")
        self.assertIsInstance(rewards, np.ndarray, "Should return numpy array")

        # Sanity checks
        self.assertTrue(all(r >= 0 for r in rewards), "All rewards should be non-negative")
        self.assertTrue(all(r <= 1 for r in rewards), "All rewards should be <= 1")

        # Better predictions should get higher rewards
        perfect_reward = rewards[0]
        poor_reward = rewards[1]
        terrible_reward = rewards[2]

        self.assertGreater(perfect_reward, poor_reward, "Perfect prediction should beat poor prediction")
        self.assertGreater(poor_reward, terrible_reward, "Poor prediction should beat terrible prediction")

    @patch("precog.validators.reward.CMData")
    @patch("precog.validators.reward.pd_to_dict")
    def test_miners_with_same_predictions_get_same_rewards(self, mock_pd_to_dict, mock_cm_data):
        """Test that miners with identical predictions receive identical rewards."""
        # Setup mock data - create a proper DataFrame that matches expected structure
        mock_cm_instance = MagicMock()
        mock_cm_data.return_value = mock_cm_instance

        # Create a mock DataFrame with the expected structure
        mock_df_data = []
        for asset in constants.SUPPORTED_ASSETS:
            for timestamp, price in self.all_asset_data[asset].items():
                mock_df_data.append({"asset": asset, "time": timestamp, "ReferenceRateUSD": price})

        mock_df = DataFrame(mock_df_data)
        mock_cm_instance.get_CM_ReferenceRate.return_value = mock_df
        mock_pd_to_dict.side_effect = lambda df: self.all_asset_data[df["asset"].iloc[0]]

        # Get actual prices for all assets at eval time
        actual_prices = {asset: self.all_asset_data[asset][self.eval_time] for asset in constants.SUPPORTED_ASSETS}

        responses = []

        for uid in range(250):
            predictions = {}
            intervals = {}

            for asset in constants.SUPPORTED_ASSETS:
                actual_price = actual_prices[asset]

                if uid < 5:  # First 5 miners: identical predictions
                    prediction = actual_price * 0.95  # 5% below actual
                    interval = [actual_price * 0.9, actual_price * 1.0]
                else:  # Different predictions for other miners
                    noise_factor = np.random.normal(1.0, 0.1)  # 10% std dev
                    prediction = actual_price * noise_factor
                    interval_width = actual_price * 0.05  # 5% range
                    interval = [prediction - interval_width, prediction + interval_width]

                predictions[asset] = prediction
                intervals[asset] = interval

            # Create response - using supported assets from constants
            responses.append(
                Challenge(
                    timestamp=self.timestamp_str,
                    assets=constants.SUPPORTED_ASSETS,
                    predictions=predictions,
                    intervals=intervals,
                )
            )

            # Pre-populate MinerHistory with predictions from 1 hour ago
            # The reward calculation expects to find predictions at prediction_time
            self.mock_validator.MinerHistory[uid].add_prediction(to_str(self.prediction_time), predictions, intervals)

        # Calculate rewards
        rewards = calc_rewards(self.mock_validator, responses)

        # Check that first 5 miners (with identical predictions) get identical rewards
        tied_rewards = rewards[:5]
        self.assertTrue(
            all(abs(r - tied_rewards[0]) < 1e-10 for r in tied_rewards),
            f"Miners with identical predictions should get identical rewards. Got: {tied_rewards}",
        )


if __name__ == "__main__":
    unittest.main()
