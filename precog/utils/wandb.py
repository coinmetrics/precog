import os

import bittensor as bt
import wandb

from precog import __version__, constants


def setup_wandb(self) -> None:
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key is not None:
        wandb.init(
            project=f"sn{self.config.netuid}-validators",
            entity=constants.WANDB_PROJECT,
            config={
                "hotkey": self.wallet.hotkey.ss58_address,
                "uid": self.my_uid,
                "subnet_version": __version__,
            },
            name=f"validator-{self.my_uid}-{__version__}",
            resume="auto",
            dir=self.config.neuron.full_path,
            reinit=True,
        )
    else:
        bt.logging.error("WANDB_API_KEY not found in environment variables.")


def log_wandb(responses, rewards, miner_uids, hotkeys, moving_average_scores):
    try:
        miners_info = {}

        for miner_uid, response, reward in zip(miner_uids, responses, rewards.tolist()):
            miner_data = {
                "miner_hotkey": hotkeys[miner_uid],
                "miner_reward": reward,
                "miner_moving_average": float(moving_average_scores.get(miner_uid, 0)),
            }

            # Add predictions for each asset
            if hasattr(response, "predictions") and response.predictions:
                for asset, prediction in response.predictions.items():
                    miner_data[f"miner_{asset}_prediction"] = prediction

            # Add intervals for each asset
            if hasattr(response, "intervals") and response.intervals:
                for asset, interval in response.intervals.items():
                    miner_data[f"miner_{asset}_interval"] = interval

            miners_info[miner_uid] = miner_data

        wandb_val_log = {"miners_info": miners_info}

        bt.logging.trace(f"Attempting to log data to wandb: {wandb_val_log}")
        wandb.log(wandb_val_log)
    except Exception as e:
        bt.logging.error(f"Failed to log to wandb: {str(e)}")
        bt.logging.error("Full error: ", exc_info=True)
