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

	# Check if script is already running with pm2
	if pm2 status | grep -q $(VALIDATOR_NAME); then
		echo "The main is already running with pm2. Stopping and restarting..."
		pm2 delete $(VALIDATOR_NAME)
	fi

	# Check if the update check is already running with pm2
	if pm2 status | grep -q $(AUTO_UPDATE_PROC_NAME); then
		echo "The update check is already running with pm2. Stopping and restarting..."
		pm2 delete $(AUTO_UPDATE_PROC_NAME)
	fi

	# Create the pm2 config file
	echo "module.exports = {

	apps: [
		{
		name: $(VALIDATOR_NAME),
		script: 'poetry',
		interpreter: 'python3',
		min_uptime: '5m',
		max_restarts: '5',
		args: [
			'run',
			'python',
			$(SCRIPT_LOCATION),
			'--neuron.name $(VALIDATOR_NAME)',
			'--wallet.name $(COLDKEY)',
			'--wallet.hotkey $(VALIDATOR_HOTKEY)',
			'--subtensor.chain_endpoint $($(NETWORK))',
			'--axon.port $(VALIDATOR_PORT)',
			'--netuid $(netuid)',
			'--logging.level $(LOGGING_LEVEL)'
			]
		}" > app.config.js


	# Append the pm2 config file if we want to use auto updater
	if $(AUTO_UPDATE); then
		echo "Adding auto updater"
		echo ",
		{
		name: 'auto_updater',
		script: './scripts/autoupdater.sh',
		interpreter: '/bin/bash',
		min_uptime: '5m',
		max_restarts: '5',
		env: {
			'UPDATE_CHECK_INTERVAL': '300',
			'GIT_BRANCH': 'main'
		}
		}" >> app.config.js

	# Append the closing bracket to the pm2 config file
	echo "
	]
	};" >> app.config.js

	# Run the Python script with the arguments using pm2
	echo "Running $(SCRIPT_LOCATION) with the following pm2 config:"

	# Print configuration to be used
	cat app.config.js

	pm2 start app.config.js
