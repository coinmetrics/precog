import asyncio
import os
import pickle

import bittensor as bt
from numpy import array
from pytz import timezone
from substrateinterface import SubstrateInterface

from precog import __spec_version__
from precog.protocol import Challenge
from precog.utils.bittensor import check_uid_availability, print_info, setup_bittensor_objects
from precog.utils.classes import MinerHistory
from precog.utils.timestamp import elapsed_seconds, get_before, get_now, is_query_time, iso8601_to_datetime
from precog.utils.wandb import log_wandb, setup_wandb
from precog.validators.reward import calc_rewards


class weight_setter:
    def __init__(self, config=None, loop=None):
        self.config = config
        self.loop = loop
        self.lock = asyncio.Lock()
        setup_bittensor_objects(self)
        self.timezone = timezone("UTC")
        self.prediction_interval = self.config.prediction_interval  # in minutes
        self.N_TIMEPOINTS = self.config.N_TIMEPOINTS  # number of timepoints to predict
        self.last_sync = 0
        self.set_weights_rate = 100  # in blocks
        self.resync_metagraph_rate = 20  # in blocks
        bt.logging.info(
            f"Running validator for subnet: {self.config.netuid} on network: {self.config.subtensor.network}"
        )
        self.available_uids = asyncio.run(self.get_available_uids())
        self.hotkeys = self.metagraph.hotkeys
        if self.config.reset_state:
            self.scores = [0.0] * len(self.metagraph.S)
            self.moving_average_scores = {uid: 0 for uid in self.metagraph.uids}
            self.MinerHistory = {uid: MinerHistory(uid, timezone=self.timezone) for uid in self.available_uids}
            self.save_state()
        else:
            self.load_state()
        self.node = SubstrateInterface(url=self.config.subtensor.chain_endpoint)
        self.current_block = self.subtensor.get_current_block()
        self.blocks_since_last_update = (
            self.current_block - self.node_query("SubtensorModule", "LastUpdate", [self.config.netuid])[self.my_uid]
        )
        self.tempo = self.node_query("SubtensorModule", "Tempo", [self.config.netuid])
        if self.config.wandb_on:
            setup_wandb(self)
        self.stop_event = asyncio.Event()
        bt.logging.info("Setup complete, starting loop")
        self.loop.create_task(
            self.loop_handler(self.scheduled_prediction_request, sleep_time=self.config.print_cadence)
        )
        self.loop.create_task(self.loop_handler(self.resync_metagraph, sleep_time=self.resync_metagraph_rate))
        self.loop.create_task(self.loop_handler(self.set_weights, sleep_time=self.set_weights_rate))

    def __exit__(self, exc_type, exc_value, traceback):
        self.save_state()
        try:
            pending = asyncio.all_tasks(self.loop)
            for task in pending:
                task.cancel()
            asyncio.gather(*pending)
        except Exception as e:
            bt.logging.error(f"Error on __exit__ function: {e}")
        self.loop.stop()

    async def loop_handler(self, func, sleep_time=120):
        try:
            while not self.stop_event.is_set():
                await func()
                await asyncio.sleep(sleep_time)
        except asyncio.exceptions.CancelledError:
            raise
        except KeyboardInterrupt:
            raise
        except Exception:
            raise
        finally:
            async with self.lock:
                self.stop_event.set()
                self.__exit__(None, None, None)

    async def get_available_uids(self):
        miner_uids = []
        for uid in range(len(self.metagraph.S)):
            uid_is_available = check_uid_availability(self.metagraph, uid, self.config.vpermit_tao_limit)
            if uid_is_available:
                miner_uids.append(uid)
        return miner_uids

    async def resync_metagraph(self, force=False):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        async with self.lock:
            self.blocks_since_sync = self.current_block - self.last_sync
            if self.blocks_since_sync >= self.resync_metagraph_rate or force:
                bt.logging.info("Syncing Metagraph...")
                self.metagraph.sync(subtensor=self.subtensor)
                bt.logging.info("Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages")
                # Zero out all hotkeys that have been replaced.
                self.available_uids = asyncio.run(self.get_available_uids())
                for uid, hotkey in enumerate(self.metagraph.hotkeys):
                    if (uid not in self.MinerHistory and uid in self.available_uids) or self.hotkeys[uid] != hotkey:
                        bt.logging.info(f"Replacing hotkey on {uid} with {self.metagraph.hotkeys[uid]}")
                        self.hotkeys[uid] = hotkey
                        self.scores[uid] = 0  # hotkey has been replaced
                        self.MinerHistory[uid] = MinerHistory(uid, timezone=self.timezone)
                        self.moving_average_scores[uid] = 0
                self.last_sync = self.subtensor.get_current_block()
                self.save_state()

    def query_miners(self):
        timestamp = get_now().isoformat()
        synapse = Challenge(timestamp=timestamp)
        responses = self.dendrite.query(
            # Send the query to selected miner axons in the network.
            axons=[self.metagraph.axons[uid] for uid in self.available_uids],
            synapse=synapse,
            deserialize=False,
        )
        return responses, timestamp

    def node_query(self, module, method, params):
        try:
            result = self.node.query(module, method, params).value
        except Exception:
            # reinitilize node
            self.node = SubstrateInterface(url=self.subtensor.chain_endpoint)
            result = self.node.query(module, method, params).value
        return result

    async def set_weights(self):
        if self.blocks_since_last_update >= self.set_weights_rate:
            async with self.lock:
                uids = array(self.available_uids)
                weights = [self.moving_average_scores[uid] for uid in self.available_uids]
            for i, j in zip(weights, self.available_uids):
                bt.logging.debug(f"UID: {j}  |  Weight: {i}")
            if sum(weights) == 0:
                weights = [1] * len(weights)
            # Convert to uint16 weights and uids.
            (
                uint_uids,
                uint_weights,
            ) = bt.utils.weight_utils.convert_weights_and_uids_for_emit(uids=uids, weights=array(weights))
            # Update the incentive mechanism on the Bittensor blockchain.
            result, msg = self.subtensor.set_weights(
                netuid=self.config.netuid,
                wallet=self.wallet,
                uids=uint_uids,
                weights=uint_weights,
                wait_for_inclusion=True,
                wait_for_finalization=True,
                version_key=__spec_version__,
            )
            if result:
                bt.logging.success("✅ Set Weights on chain successfully!")
            else:
                bt.logging.debug(
                    "Failed to set weights this iteration with message:",
                    msg,
                )
        async with self.lock:
            self.current_block = self.subtensor.get_current_block()
            self.blocks_since_last_update = (
                self.current_block - self.node_query("SubtensorModule", "LastUpdate", [self.config.netuid])[self.my_uid]
            )

    async def scheduled_prediction_request(self):
        if not hasattr(self, "timestamp"):
            self.timestamp = get_before(minutes=self.prediction_interval).isoformat()
        query_lag = elapsed_seconds(get_now(), iso8601_to_datetime(self.timestamp))
        if len(self.available_uids) == 0:
            bt.logging.info("No miners available. Sleeping for 10 minutes...")
            print_info(self)
            await asyncio.sleep(600)
        else:
            if is_query_time(self.prediction_interval, self.timestamp) or query_lag >= 60 * self.prediction_interval:
                responses, self.timestamp = self.query_miners()
                try:
                    rewards = calc_rewards(self, responses=responses)
                except Exception as e:
                    bt.logging.error(f"Failed to calculate rewards with error: {e}")
                # Adjust the scores based on responses from miners and update moving average.
                async with self.lock:
                    for i, value in zip(self.available_uids, rewards):
                        self.moving_average_scores[i] = (1 - self.config.alpha) * self.moving_average_scores[
                            i
                        ] + self.config.alpha * value
                        self.scores = list(self.moving_average_scores.values())
                if self.config.wandb_on:
                    log_wandb(responses, rewards, self.available_uids)
            else:
                print_info(self)

    def save_state(self):
        """Saves the state of the validator to a file."""

        state_path = os.path.join(self.config.full_path, "state.pt")
        state = {
            "scores": self.scores,
            "MinerHistory": self.MinerHistory,
            "moving_average_scores": self.moving_average_scores,
        }
        with open(state_path, "wb") as f:
            pickle.dump(state, f)
        bt.logging.info(f"Saved {self.config.neuron.name} state.")

    def load_state(self):
        """Loads the state of the validator from a file."""
        bt.logging.info("Loading validator state.")
        state_path = os.path.join(self.config.full_path, "state.pt")
        bt.logging.info(f"State path: {state_path}")
        if not os.path.exists(state_path):
            bt.logging.info("Skipping state load due to missing state.pt file.")
            self.scores = [0.0] * len(self.metagraph.S)
            self.moving_average_scores = {uid: 0 for uid in self.metagraph.uids}
            self.MinerHistory = {uid: MinerHistory(uid) for uid in self.available_uids}
            return
        try:
            with open(state_path, "rb") as f:
                state = pickle.load(f)
            self.scores = state["scores"]
            self.MinerHistory = state["MinerHistory"]
            self.moving_average_scores = state["moving_average_scores"]
        except Exception as e:
            bt.logging.error(f"Failed to load state with error: {e}")