ENV_FILE ?= .env

include $(ENV_FILE)
export

finney = wss://entrypoint-finney.opentensor.ai:443
testnet = wss://test.finney.opentensor.ai:443
localnet = ws://127.0.0.1:9945

ifeq ($(NETWORK),localnet)
   netuid = 1
else ifeq ($(NETWORK),testnet)
   netuid = 256
else ifeq ($(NETWORK),finney)
   #netuid = 64
   $(error Finney network not supported yet)
endif

metagraph:
	btcli subnet metagraph --netuid $(netuid) --subtensor.chain_endpoint $($(NETWORK))

register:
	{ \
		read -p 'Wallet name?: ' wallet_name ;\
		read -p 'Hotkey?: ' hotkey_name ;\
		btcli subnet register --netuid $(netuid) --wallet.name "$$wallet_name" --wallet.hotkey "$$hotkey_name" --subtensor.chain_endpoint $($(NETWORK)) ;\
	}

validator:
	python start_validator.py \
		--neuron.name $(VALIDATOR_NAME) \
		--wallet.name $(COLDKEY) \
		--wallet.hotkey $(VALIDATOR_HOTKEY) \
		--subtensor.chain_endpoint $($(NETWORK)) \
		--axon.port $(VALIDATOR_PORT) \
		--axon.ip $(AXON_IP) \
		--axon.external_ip $(AXON_EXTERNAL_IP) \
		--netuid $(netuid) \
		--logging.level $(LOGGING_LEVEL)

miner:
	python start_miner.py \
		--neuron.name $(MINER_NAME) \
		--wallet.name $(COLDKEY) \
		--wallet.hotkey $(MINER_HOTKEY) \
		--subtensor.chain_endpoint $($(NETWORK)) \
		--axon.port $(MINER_PORT) \
		--axon.ip $(AXON_IP) \
		--axon.external_ip $(AXON_EXTERNAL_IP) \
		--netuid $(netuid) \
		--logging.level $(LOGGING_LEVEL) \
		--timeout $(TIMEOUT) \
		--vpermit_tao_limit $(VPERMIT_TAO_LIMIT) \
		--forward_function $(FORWARD_FUNCTION)
