import asyncio
import importlib
import time
import typing
from datetime import datetime

import bittensor as bt
from pytz import timezone
from substrateinterface import Keypair, SubstrateInterface

from precog.protocol import Challenge
from precog.utils.bittensor import print_info, setup_bittensor_objects
from precog.utils.classes import Config
from precog.utils.general import parse_arguments


class Miner:
    """
    Optimized Miner to handle ultra-fast requests with low latency.
    """

    def __init__(self, config=None):
        args = parse_arguments()
        config = Config(args)
        self.forward_module = importlib.import_module(f"precog.miners.{config.forward_function}")
        self.config = config
        self.config.neuron.type = "Miner"
        setup_bittensor_objects(self)
        # Attach determiners which functions are called when servicing a request.
        bt.logging.info("Attaching forward function to miner axon.")
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority,
            # verify_fn=self.verify,
        )
        self.nonces = {}
        self.current_block = self.node_query("System", "Number", [])
        self.current_prediction = [datetime.now(timezone("America/New_York")), [0]]
        self.loop = asyncio.get_event_loop()
        self.loop.create_task(self.run())

    async def run(self):
        bt.logging.info(
            f"Serving miner axon {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        self.axon.start()
        while True:
            try:
                print_info(self)
                await asyncio.sleep(self.config.print_cadence)
            except KeyboardInterrupt:
                bt.logging.info("Keyboard interrupt detected, shutting down miner.")
                self.axon.stop()
                break

    async def forward(self, synapse: Challenge) -> Challenge:
        synapse = self.forward_module.forward(synapse)
        if synapse.prediction is not None:
            bt.logging.success(f"Predicted price: {synapse.prediction}  |  Predicted Interval: {synapse.interval}")
        else:
            bt.logging.info("No price predicted for this request.")
        return synapse

    def save_state(self):
        pass

    def load_state(self):
        pass

    async def blacklist(self, synapse: Challenge) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contracted via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (template.protocol.Dummy): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """

        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return True, "Missing dendrite or hotkey"

        # TODO(developer): Define how miners should blacklist requests.
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if not self.config.blacklist.allow_non_registered and synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            # Ignore requests from un-registered entities.
            bt.logging.trace(f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}")
            return True, "Unrecognized hotkey"

        if self.config.blacklist.force_validator_permit:
            # If the config is set to force validator permit, then we should only allow requests from validators.
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}")
                return True, "Non-validator hotkey"

        bt.logging.trace(f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}")
        return False, "Hotkey recognized!"

    async def priority(self, synapse: Challenge) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (template.protocol.Dummy): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may receive messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return 0.0

        # TODO(developer): Define how miners should prioritize requests.
        caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)  # Get the caller index.
        priority = float(self.metagraph.S[caller_uid])  # Return the stake as the priority.
        bt.logging.trace(f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}")
        return priority

    async def verify(self, synapse: Challenge) -> None:
        # needs to replace the base miner verify logic
        bt.logging.debug(f"checking nonce: {synapse.dendrite}")
        # Build the keypair from the dendrite_hotkey
        if synapse.dendrite is not None:
            keypair = Keypair(ss58_address=synapse.dendrite.hotkey)
            # Build the signature messages.
            message = f"{synapse.dendrite.nonce}.{synapse.dendrite.hotkey}.{self.wallet.hotkey.ss58_address}.{synapse.dendrite.uuid}.{synapse.computed_body_hash}"
            # Build the unique endpoint key.
            endpoint_key = f"{synapse.dendrite.hotkey}:{synapse.dendrite.uuid}"
            # Requests must have nonces to be safe from replays
            if synapse.dendrite.nonce is None:
                raise Exception("Missing Nonce")
            if synapse.dendrite.version is not None:
                bt.logging.debug("Using custom synapse verification logic")
                # If we don't have a nonce stored, ensure that the nonce falls within
                # a reasonable delta.
                cur_time = time.time_ns()

                allowed_delta = self.config.timeout * 1_000_000_000  # nanoseconds

                latest_allowed_nonce = synapse.dendrite.nonce + allowed_delta
                bt.logging.debug(f"delta: {self._to_seconds(cur_time - synapse.dendrite.nonce)}")

                if self.nonces.get(endpoint_key) is None and synapse.dendrite.nonce > latest_allowed_nonce:
                    raise Exception(
                        f"Nonce is too old. Allowed delta in seconds: {self._to_seconds(allowed_delta)}, got delta: {self._to_seconds(cur_time - synapse.dendrite.nonce)}"
                    )
                if self.nonces.get(endpoint_key) is not None and synapse.dendrite.nonce <= self.nonces[endpoint_key]:
                    raise Exception(
                        f"Nonce is too small, already have a newer nonce in the nonce store, got: {synapse.dendrite.nonce}, already have: {self.nonces[endpoint_key]}"
                    )
            else:
                bt.logging.warning(f"Using synapse verification logic for version < 7.2.0: {synapse.dendrite.version}")
                if (
                    endpoint_key in self.nonces.keys()
                    and self.nonces[endpoint_key] is not None
                    and synapse.dendrite.nonce <= self.nonces[endpoint_key]
                ):
                    raise Exception(
                        f"Nonce is too small, already have a newer nonce in the nonce store, got: {synapse.dendrite.nonce}, already have: {self.nonces[endpoint_key]}"
                    )

            if not keypair.verify(message, synapse.dendrite.signature):
                raise Exception(
                    f"Signature mismatch with {message} and {synapse.dendrite.signature}, from hotkey {synapse.dendrite.hotkey}"
                )

            # Success
            self.nonces[endpoint_key] = synapse.dendrite.nonce  # type: ignore

    def node_query(self, module, method, params):
        try:
            result = self.node.query(module, method, params).value
        except Exception:
            # reinitilize node
            self.node = SubstrateInterface(url=self.subtensor.chain_endpoint)
            result = self.node.query(module, method, params).value
        return result

    def _to_seconds(self, nano: int) -> int:
        return nano / 1_000_000_000


# Run the miner
if __name__ == "__main__":
    args = parse_arguments()
    config = Config(args)
    miner = Miner(config=config)
    miner.loop.run_forever()
