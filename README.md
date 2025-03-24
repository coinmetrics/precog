






<div align="center">
# **CoinMetrics Precog Subnet** <!-- omit in toc -->
<div align="center">

| **Testnet UID:**  256 <br> **Mainnet UID:**  55 |
| - |

</div>

<br/>

|     |     |
| :-: | :-: |
| **Status** | <img src="https://img.shields.io/github/v/release/coinmetrics/precog?label=Release" height="25"/> <img src="https://img.shields.io/github/actions/workflow/status/coinmetrics/precog/ci.yml?label=Build" height="25"/> <br> <a href="https://github.com/pre-commit/pre-commit" target="_blank"> <img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white&label=Pre-Commit" height="25"/> </a> <a href="https://github.com/psf/black" target="_blank"> <img src="https://img.shields.io/badge/code%20style-black-000000.svg?label=Code%20Style" height="25"/> </a> <br> <img src="https://img.shields.io/github/license/coinmetrics/precog?label=License" height="25"/> |
| **Activity** | <img src="https://img.shields.io/github/commit-activity/m/coinmetrics/precog?label=Commit%20Activity" height="25"/> <img src="https://img.shields.io/github/commits-since/coinmetrics/precog/latest/dev?label=Commits%20Since%20Latest%20Release" height="25"/> <br> <img src="https://img.shields.io/github/release-date/coinmetrics/precog?label=Latest%20Release%20Date" height="25"/> <img src="https://img.shields.io/github/last-commit/coinmetrics/precog/dev?label=Last%20Commit" height="25"/> <br> <img src="https://img.shields.io/github/contributors/coinmetrics/precog?label=Contributors" height="25"/> |
| **Compatibility** | <img src="https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Fcoinmetrics%2Fprecog%2Frefs%2Fheads%2Fdev%2Fpyproject.toml&query=%24.tool.poetry.dependencies.python&logo=python&label=Python&logoColor=yellow" height="25"/> <img src="https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Fcoinmetrics%2Fprecog%2Frefs%2Fheads%2Fdev%2Fpyproject.toml&query=%24.tool.poetry.dependencies.bittensor&prefix=v&label=Bittensor" height="25"/> <br> <img src="https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Fcoinmetrics%2Fprecog%2Frefs%2Fheads%2Fdev%2Fpyproject.toml&query=%24.tool.poetry.dependencies.coinmetrics-api-client&label=coinmetrics-api-client" height="25"/> |
| **Social** | <a href="" target="_blank"> <img src="https://img.shields.io/website?url=https%3A%2F%2Fcharts.coinmetrics.io%2Fcrypto-data%2F&up_message=CoinMetrics&label=Website" height="25"/> </a> |


</div>


---
## Introduction

The Precog Subnet will serve as an arena to identify the analysts with the best strategies and information to anticipate Bitcoin price movements, and recruit them to share their insight with the broader community.  By leveraging Bittensor’s subnet structure multiple perspectives can compete with the collective goal of filtering out noise in price signals.  The open ecosystem creates a mechanism for people with unique knowledge to benefit themselves in exchange for sharing their specialties with the public

---
## Design Decisions
The initial focus is on Bitcoin as it is the most well-established and arguably most decentralized crypto currency.  Bitcoin has an abundance of high-resolution data, while still retaining enough volatility to make price prediction a valuable challenge.

We decided to focus on high-frequency predictions (every 5 minutes) with short resolution times (every hour) because we believe this leverages the unique capabilities of Bittensor to do something that traditional derivatives markets cannot do.  While options and futures allow markets to express sentiments around asset price, they are unable to do so in a continuously and with such a short settlement time.

The incentive mechanism was specifically designed to reward precise statements of the future price level, as well as the most likely band the price will trade in. Compared to mechanisms with less precise “long” or “short” predictions, and pre-determined strike price intervals, we believe the metrics we query are closer to what traders and analysts truly want: the most likely price in the future with frequent updates.

---
## Compute Requirements

| Validator |   Miner   |
|---------- |-----------|
|  8gb RAM  |  8gb RAM  |
|  2 vCPUs  |  2 vCPUs  |

---

## Prerequisites

Before beginning, ensure you have:

1. **Python Installation**
   - Python version 3.9, 3.10, or 3.11 installed
   - We recommend using `pyenv` for Python version management

2. **Bittensor Knowledge**
   - Understanding of the Bittensor ecosystem and wallet management
   - Familiarity with creating and managing Bittensor wallets
   - Review the [Bittensor Wallet Documentation](https://docs.bittensor.com/getting-started/wallets)
   - For general understanding, see the [Bittensor Documentation](https://docs.bittensor.com/)

3. **Poetry**
   - Basic understanding of Poetry for dependency management
   - See the [Poetry Documentation](https://python-poetry.org/docs/) for installation and usage

4. **System Requirements**
   - NodeJS and NPM (for PM2 installation)
   - Basic understanding of command-line interfaces

## Installation

First, install PM2:
```
sudo apt update
sudo apt install nodejs npm
sudo npm install pm2@latest -g
```
Verify installation:
```
pm2 --version
```


Clone the repository:
```
git clone https://github.com/coinmetrics/precog.git
cd precog
```

Create and source a python virtual environment:
```
python3 -m venv .venv
source .venv/bin/activate
```

Install the requirements with poetry:
```
pip install poetry
poetry install
```

---
## Configuration


### .env Files
Copy the example `.env` files and edit all desired values. If you are running a validator, you will only need to copy the .env.validator file. If you are running a miner, you will only need to copy the .env.miner file:

#### Obtain & Setup WandB API Key (Validators Only)
Before starting the process, validators would be required to procure a WANDB API Key. Please follow the instructions mentioned below:<br>

- Log in to <a href="https://wandb.ai">Weights & Biases</a> and generate an API key in your account settings.
- Set the variable `WANDB_API_KEY` in the `.env.validator` file.
- Finally, run `wandb login` and paste your API key. Now you're all set with weights & biases.

For help finding your wandb api key, look <a href="https://docs.wandb.ai/support/find_api_key/">here</a>

#### .env.validator
```
cp .env.validator.example .env.validator
```
Edit `.env.validator` with your desired values.

```
# Network Configuration
# Options: localnet, testnet, finney
NETWORK=finney

# Wallet Configuration
COLDKEY=your_validator_coldkey
VALIDATOR_HOTKEY=your_validator_hotkey

# Node Configuration
VALIDATOR_NAME=validator
VALIDATOR_PORT=8091

# Auto Update Configuration
# Whether to run the auto update script
# Options: 1, 0
AUTO_UPDATE=1
# The name of the process to monitor for auto updates
AUTO_UPDATE_PROC_NAME=auto-updater

# The location of the validator entry script
SCRIPT_LOCATION=./precog/validators/validator.py

# Logging
# Options: info, debug, trace
LOGGING_LEVEL=debug

# Local Subtensor Configuration
# Only used if you run your own subtensor node
LOCALNET=ws://127.0.0.1:9945

#Wandb Setup
WANDB_API_KEY=your_wandb_api_key
```

#### .env.miner
```
cp .env.miner.example .env.miner
```
Edit `.env.miner` with your desired values.

```# Network Configuration
# Options: localnet, testnet, finney
NETWORK=finney

# Wallet Configuration
COLDKEY=your_miner_coldkey
MINER_HOTKEY=your_miner_hotkey

# Node Configuration
MINER_NAME=miner
# This port must be open to accept incoming TCP connections.
MINER_PORT=8092

# Miner Settings
TIMEOUT=16
VPERMIT_TAO_LIMIT=2

#Adjust this function if you would like to specify a custom forward function
FORWARD_FUNCTION=base_miner

# Logging
# Options: info, debug, trace
LOGGING_LEVEL=debug

# Local Subtensor Configuration
# Only used if you run your own subtensor node
LOCALNET=ws://127.0.0.1:9945
```

---
## Deployment

### Registering a Hotkey
Once you have configured your .env files as per the instructions above, you can register a miner with `make register ENV_FILE=.env.miner` or register a validator with `make register ENV_FILE=.env.validator`.

### Running a Miner
Base miner:
1. Run the command:
    ```
    make miner ENV_FILE=.env.miner
    ```

Custom miner:
1. Write a custom forward function stored in `precog/miners/your_file.py`
    - `miner.py` searches for a function called `forward` contained within your provided file `--forward_function your_file`
    - This function should handle how the miner responds to requests from the validator
    - Within the forward function, `synapse.predictions` and `synapse.interval` should be set.
    - See [base_miner.py](https://github.com/coinmetrics/precog/blob/master/precog/miners/base_miner.py) for an example
2. Add a command to Makefile.
    - copy the miner command and rename it (e.g. miner_custom) in Makefile
    - replace the `--forward_function base_miner` with `--forward_function your_file`
3. Run the Command:
    ```
    make miner_custom ENV_FILE=.env.custom
    ```


### Running a Validator
This will start the validator and the auto update script. Both run as independent pm2 processes. To opt out of auto updates, set `AUTO_UPDATE=0` in your .env.validator file.
>The auto update script will run every 5 minutes. This script will pull down the latest repository changes and will restart the validator and auto update script if changes are detected. This is to ensure that the validator is always running the latest version of the code. If you have made custom changes to the validator source code, you may want to disable auto updates to prevent overwriting your changes.

```
make validator ENV_FILE=.env.validator
```


## Incentive Mechanism
Please see the [Incentive mechanism Documentation](https://docs.coinmetrics.io/bittensor/precog-methodology) on the Coin Metrics docs site for an in-depth explanation of the mechanism.  These describe the process for evaluating each forecast type's performance and ranking Mines on that basis, including examples to grok the Interval Forecast.

Briefly, miners are rewarded based on two factors:
1. A point estimate for the price of BTC in USD one hour from prediction time
2. An interval estimate (the minimum and maximum price of BTC over the next hour based on a 1s frequency)

For more information, feel free to investigate [reward.py](https://github.com/coinmetrics/precog/blob/master/precog/validators/reward.py) or [base_miner.py](https://github.com/coinmetrics/precog/blob/master/precog/miners/base_miner.py)



---
## Roadmap

Our goal is to continuously improve the subnet and tune it to the goals and interests that will engage the community.  We have considered additions in the form of additional asset coverage, such as extending price analysis to TAO tokens.  Extensions can also mean incentive mechanisms to calculate new types of metrics such as anticipating volatility, transaction volumes, or the cost of different types of transfers.  Our greatest strength is our deep and professional data library, used by many of the largest financial institutions in crypto.  We expect these resources will allow the subnet scope to adapt quickly when conditions are right.

We hope to, on one hand, leverage our existing products and coverage to make generating new insights as frictionless as possible for Miners.  While on the other hand, we also hope to integrate new data streams into our catalog that only the Bittensor ecosystem can generate.  Our aim is for these novel outputs to ultimately bring new participants to Bittensor from the broader crypto community, as we serve metrics and analysis that can't be obtained anywhere else.
