#!/bin/bash

# Create the pm2 config file
echo "module.exports = {

apps: [
    {
    name: '$VALIDATOR_NAME',
    script: 'poetry',
    interpreter: 'python3',
    min_uptime: '5m',
    max_restarts: '5',
    args: [
        'run',
        'python',
        '$SCRIPT_LOCATION',
        '--neuron.name', '$VALIDATOR_NAME',
        '--wallet.name', '$COLDKEY',
        '--wallet.hotkey', '$VALIDATOR_HOTKEY',
        '--subtensor.chain_endpoint', '${!NETWORK}',
        '--axon.port', '$VALIDATOR_PORT',
        '--netuid', '$netuid',
        '--logging.level', '$LOGGING_LEVEL'
        ]
    }" > app.config.js


# Append the pm2 config file if we want to use auto updater
if [[ $AUTO_UPDATE == 1 ]]; then
    echo "Adding auto updater"
    echo ",
    {
    name: '$AUTO_UPDATE_PROC_NAME',
    autorestart: false,
    instances: 1,
    cron_restart: '2/5 * * * *',
    script: 'poetry',
    interpreter: 'python3',
    args: [
        'run',
        'python',
        './precog/validators/scripts/auto_updater.py'
        ]
    }" >> app.config.js
else
    echo "Not using auto updater"
fi

# Append the closing bracket to the pm2 config file
echo "
]
};" >> app.config.js

# Run the Python script with the arguments using pm2
echo "Running $SCRIPT_LOCATION with the following pm2 config:"

# Print configuration to be used
cat app.config.js
