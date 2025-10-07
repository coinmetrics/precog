include $(ENV_FILE)
export

finney = wss://entrypoint-finney.opentensor.ai:443
testnet = wss://test.finney.opentensor.ai:443
localnet = $(LOCALNET)

ifeq ($(NETWORK),localnet)
   netuid = 55
else ifeq ($(NETWORK),testnet)
   netuid = 256
else ifeq ($(NETWORK),finney)
   netuid = 55
endif

metagraph:
	btcli subnet metagraph --netuid $(netuid) --subtensor.chain_endpoint $($(NETWORK))

register:
	{ \
		read -p 'Wallet name?: ' wallet_name ;\
		read -p 'Hotkey?: ' hotkey_name ;\
		btcli subnet register --netuid $(netuid) --wallet.name "$$wallet_name" --wallet.hotkey "$$hotkey_name" --subtensor.chain_endpoint $($(NETWORK)) ;\
	}

miner:
	pm2 start --name $(MINER_NAME) python3 -- precog/miners/miner.py \
		--neuron.name $(MINER_NAME) \
		--wallet.name $(COLDKEY) \
		--wallet.hotkey $(MINER_HOTKEY) \
		--subtensor.chain_endpoint $($(NETWORK)) \
		--axon.port $(MINER_PORT) \
		--netuid $(netuid) \
		--logging.level $(LOGGING_LEVEL) \
		--timeout $(TIMEOUT) \
		--vpermit_tao_limit $(VPERMIT_TAO_LIMIT) \
		--forward_function $(FORWARD_FUNCTION) \

validator:
	pm2 start --name $(VALIDATOR_NAME) python3 -- precog/validators/validator.py \
		--neuron.name $(VALIDATOR_NAME) \
		--wallet.name $(COLDKEY) \
		--wallet.hotkey $(VALIDATOR_HOTKEY) \
		--subtensor.chain_endpoint $($(NETWORK)) \
		--axon.port $(VALIDATOR_PORT) \
		--netuid $(netuid) \
		--logging.level $(LOGGING_LEVEL)
