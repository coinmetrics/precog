from pathlib import Path

import bittensor as bt


def setup_bittensor_objects(self):
    if self.config.logging.level == "trace":
        bt.logging.set_trace()
    elif self.config.logging.level == "debug":
        bt.logging.set_debug()
    else:
        # set to info by default
        pass
    bt.logging.info(f"Set logging level to {self.config.logging.level}")

    # if chain enpoint isn't set, use the network arg
    if self.config.subtensor.chain_endpoint is None:
        self.config.subtensor.chain_endpoint = bt.subtensor.determine_chain_endpoint_and_network(
            self.config.subtensor.network
        )[1]
    else:
        # if chain endpoint is set, overwrite network arg
        self.config.subtensor.network = self.config.subtensor.chain_endpoint
    # Initialize subtensor.
    self.subtensor = bt.subtensor(config=self.config, network=self.config.subtensor.chain_endpoint)
    self.metagraph = self.subtensor.metagraph(self.config.netuid)
    self.wallet = bt.wallet(config=self.config)
    self.dendrite = bt.dendrite(wallet=self.wallet)

    # Initialize axon config
    axon_config = bt.axon.config()
    axon_config.max_workers = self.config.axon.max_workers
    axon_config.port = self.config.axon.port
    axon_config.ip = self.config.axon.ip
    axon_config.external_ip = self.config.axon.external_ip
    axon_config.external_port = self.config.axon.external_port
    self.config.axon = axon_config

    # Debug prints
    bt.logging.debug(f"Axon config - port: {self.config.axon.port}")
    bt.logging.debug(f"Axon config - ip: {self.config.axon.ip}")
    bt.logging.debug(f"Axon config - external_ip: {self.config.axon.external_ip}")
    bt.logging.debug(f"Axon config - external_port: {self.config.axon.external_port}")
    bt.logging.debug(f"Axon config - max_workers: {self.config.axon.max_workers}")

    self.axon = bt.axon(
        wallet=self.wallet,
        config=self.config,
        port=self.config.axon.port,
        ip=self.config.axon.ip,
        external_ip=self.config.axon.external_ip,
        external_port=self.config.axon.external_port,
        max_workers=self.config.axon.max_workers,
    )
    # Connect the validator to the network.
    if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
        bt.logging.error(
            f"\nYour {self.config.neuron.type}: {self.wallet} is not registered to chain connection: {self.subtensor} \nRun 'btcli register' and try again."
        )
        exit()
    else:
        # Each validator gets a unique identity (UID) in the network.
        self.my_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        bt.logging.info(f"Running {self.config.neuron.type} on uid: {self.my_uid}")

    full_path = Path(
        f"~/.bittensor/{self.config.neuron.type}s/{self.config.wallet.name}/{self.config.wallet.hotkey}/netuid{self.config.netuid}/{self.config.neuron.type}"
    ).expanduser()
    full_path.mkdir(parents=True, exist_ok=True)
    self.config.full_path = str(full_path)


def print_info(self) -> None:
    if self.config.neuron.type == "Validator":
        weight_timing = self.hyperparameters.weights_rate_limit - self.blocks_since_last_update
        if weight_timing <= 0:
            weight_timing = "a few"  # hashtag aesthetic af
        log = (
            "Validator | "
            f"UID:{self.my_uid} | "
            f"Block:{self.current_block} | "
            f"Stake:{self.metagraph.S[self.my_uid]:.3f} | "
            f"VTrust:{self.metagraph.Tv[self.my_uid]:.3f} | "
            f"Dividend:{self.metagraph.D[self.my_uid]:.3f} | "
            f"Emission:{self.metagraph.E[self.my_uid]:.3f} | "
            f"Seting weights in {weight_timing} blocks"
        )
    elif self.config.neuron.type == "Miner":
        log = (
            "Miner | "
            f"UID:{self.my_uid} | "
            f"Block:{self.current_block} | "
            f"Stake:{self.metagraph.S[self.my_uid]:.3f} | "
            f"Trust:{self.metagraph.T[self.my_uid]:.3f} | "
            f"Incentive:{self.metagraph.I[self.my_uid]:.3f} | "
            f"Emission:{self.metagraph.E[self.my_uid]:.3f}"
        )
    else:
        log = f"Unknown Neuron Type: {self.config.neuron.type}\nPlease choose from: [Validator, Miner]"
    bt.logging.info(log)


def check_uid_availability(metagraph: "bt.metagraph.Metagraph", uid: int, vpermit_tao_limit: int) -> bool:
    """Check if uid is available. The UID should be available if it is serving and has less than vpermit_tao_limit stake
    Args:
        metagraph (:obj: bt.metagraph.Metagraph): Metagraph object
        uid (int): uid to be checked
        vpermit_tao_limit (int): Validator permit tao limit
    Returns:
        bool: True if uid is available, False otherwise
    """
    # Filter non serving axons.
    if not metagraph.axons[uid].is_serving:
        return False
    # Filter validator permit > 1024 stake.
    if metagraph.validator_permit[uid]:
        if metagraph.S[uid] > vpermit_tao_limit:
            return False
    # Available otherwise.
    return True
