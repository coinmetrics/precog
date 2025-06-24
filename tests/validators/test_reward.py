import pickle
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from pandas import DataFrame

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

        # Load real Bitcoin price data
        self.real_price_data = self._load_real_price_data()

        if self.real_price_data is None:
            self.skipTest("Real price data not available")

    def _load_real_price_data(self):
        """Load real Bitcoin price data from the saved file."""
        test_data_path = Path("tests/data/btc_test_data.pkl")

        try:
            with open(test_data_path, "rb") as f:
                data = pickle.load(f)

            self.eval_time = data["eval_time"]
            self.prediction_time = data["prediction_time"]
            self.timestamp_str = to_str(self.eval_time)

            return data["dict"]

        except Exception as e:
            self.skipTest(f"Test data not available: {e}")

    @patch("precog.validators.reward.CMData")
    @patch("precog.validators.reward.pd_to_dict")
    def test_calc_rewards(self, mock_pd_to_dict, mock_cm_data):
        """Simple test that calc_rewards works and miners with better predictions get higher rewards."""
        # Setup mock data
        mock_cm_instance = MagicMock()
        mock_cm_data.return_value = mock_cm_instance
        mock_cm_instance.get_CM_ReferenceRate.return_value = DataFrame()
        mock_pd_to_dict.return_value = self.real_price_data

        actual_price = self.real_price_data[self.eval_time]

        # Create 250 miners: first 3 with known different prediction qualities for testing
        responses = []

        for uid in range(250):
            miner = self.mock_validator.MinerHistory[uid]

            if uid == 0:  # Perfect prediction
                prediction = actual_price
                interval = [actual_price - 100, actual_price + 100]
            elif uid == 1:  # Poor prediction
                prediction = actual_price - 5000  # $5000 off
                interval = [actual_price - 6000, actual_price - 4000]
            elif uid == 2:  # Terrible prediction
                prediction = actual_price / 2  # 50% off
                interval = [actual_price / 2 - 1000, actual_price / 2 + 1000]
            else:  # Random predictions for remaining miners
                prediction = actual_price + np.random.normal(0, 1000)
                interval = [prediction - 500, prediction + 500]

            # Set up miner prediction history
            miner.predictions[self.prediction_time] = prediction
            miner.intervals[self.prediction_time] = interval

            # Create response
            responses.append(Challenge(timestamp=self.timestamp_str, prediction=prediction, interval=interval))

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
        # Setup mock data
        mock_cm_instance = MagicMock()
        mock_cm_data.return_value = mock_cm_instance
        mock_cm_instance.get_CM_ReferenceRate.return_value = DataFrame()
        mock_pd_to_dict.return_value = self.real_price_data

        actual_price = self.real_price_data[self.eval_time]

        # Create identical predictions for first 5 miners
        identical_prediction = actual_price - 200
        identical_interval = [actual_price - 400, actual_price]

        responses = []

        for uid in range(250):
            miner = self.mock_validator.MinerHistory[uid]

            if uid < 5:  # First 5 miners: identical predictions
                prediction = identical_prediction
                interval = identical_interval
            else:  # Different predictions for other miners
                prediction = actual_price + np.random.normal(0, 1000)
                interval = [prediction - 500, prediction + 500]

            # Set up miner prediction history
            miner.predictions[self.prediction_time] = prediction
            miner.intervals[self.prediction_time] = interval

            # Create response
            responses.append(Challenge(timestamp=self.timestamp_str, prediction=prediction, interval=interval))

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
