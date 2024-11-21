import argparse
import re
from typing import Optional

import bittensor as bt
import git
import requests

from precog.utils.classes import NestedNamespace


def parse_arguments(parser: Optional[argparse.ArgumentParser] = None):
    """Used to overwrite defaults when params are passed into the script.

    Args:
        parser (Optional[argparse.ArgumentParser], optional): _description_. Default arguments shown below.

    Example:
        >>> python3 -m start_miner.py --netuid 2
        >>> args = parse_arguments()
        >>> print(args.subtensor.chain_endpoint)

    Returns:
        namespace (NestedNamespace): Returns a nested arparse.namespace object which contains all the arguments
    """
    if parser is None:
        parser = argparse.ArgumentParser(description="Configuration")
    parser.add_argument(
        "--subtensor.chain_endpoint", type=str, default=None
    )  # for testnet: wss://test.finney.opentensor.ai:443
    parser.add_argument(
        "--subtensor.network",
        choices=["finney", "test", "local"],
        default="finney",
    )
    parser.add_argument("--wallet.name", type=str, default="default", help="Coldkey name")
    parser.add_argument("--wallet.hotkey", type=str, default="default", help="Hotkey name")
    parser.add_argument("--netuid", type=int, default=1, help="Subnet netuid")
    parser.add_argument("--neuron.name", type=str, default="validator", help="What to call this process")
    parser.add_argument(
        "--neuron.type",
        type=str,
        choices=["Validator", "Miner"],
        default="Validator",
        help="What type of neuron this is",
    )
    parser.add_argument("--axon.port", type=int, default=8000)
    parser.add_argument("--logging.level", type=str, choices=["info", "debug", "trace"], default="info")
    parser.add_argument("--logging.logging_dir", type=str, default="~/.bittensor/validators")
    parser.add_argument(
        "--blacklist.force_validator_permit", action="store_false", dest="blacklist.force_validator_permit"
    )
    parser.add_argument("--blacklist.allow_non_registered", action="store_true", dest="blacklist.allow_non_registered")
    parser.add_argument("--autoupdate", action="store_true", dest="autoupdate")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--prediction_interval", type=int, default=5)
    parser.add_argument("--N_TIMEPOINTS", type=int, default=12)
    parser.add_argument("--vpermit_tao_limit", type=int, default=1024)
    parser.add_argument("--wandb_on", action="store_true", dest="wandb_on")
    parser.add_argument("--reset_state", action="store_true", dest="reset_state", help="Overwrites the state file")
    parser.add_argument("--timeout", type=int, default=16, help="allowable nonce delay time (seconds)")
    parser.add_argument("--print_cadence", type=float, default=12, help="how often to print stats (seconds)")
    parser.add_argument("--forward_function", type=str, default="forward", help="name of the forward function to use")
    return parser.parse_args(namespace=NestedNamespace())


def get_version() -> Optional[str]:
    """Pulls the version of the precog-subnet repo from GitHub.

    Returns:
        Optional[str]: Repo version in the format: '1.1.0'
    """
    repo = git.Repo(search_parent_directories=True)
    branch_name = repo.active_branch.name
    url = f"https://github.com/coinmetrics/precog/blob/{branch_name}/precog/__init__.py"
    response = requests.get(url, timeout=10)
    if not response.ok:
        bt.logging.error("Failed to get version from GitHub")
        return None
    match = re.search(r"__version__ = (.{1,10})", response.text)

    version_match = re.search(r"\d+\.\d+\.\d+", match.group(1))
    if not version_match:
        raise Exception("Version information not found")

    return version_match.group()