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
        wandb_val_log = {
            "miners_info": {
                miner_uid: {
                    "miner_hotkey": hotkeys[miner_uid],
                    "miner_point_prediction": response.prediction,
                    "miner_interval_prediction": response.interval,
                    "miner_reward": reward,
                    "miner_moving_average": float(moving_average_scores.get(miner_uid, 0)),
                }
                for miner_uid, response, reward in zip(miner_uids, responses, rewards.tolist())
            },
        }

        bt.logging.trace(f"Attempting to log data to wandb: {wandb_val_log}")
        wandb.log(wandb_val_log)
    except Exception as e:
        bt.logging.error(f"Failed to log to wandb: {str(e)}")
        bt.logging.error("Full error: ", exc_info=True)
