#!/bin/bash

# Check if the update check is already running with pm2
if pm2 status | grep -q $AUTO_UPDATE_PROC_NAME; then
    echo "The update check is already running with pm2. Stopping now."
    pm2 delete $AUTO_UPDATE_PROC_NAME || true
else
    echo "The update check is not running with pm2."
fi

# Check if script is already running with pm2
if pm2 status | grep -q $VALIDATOR_NAME; then
    echo "The main is already running with pm2. Stopping now."
    pm2 delete $VALIDATOR_NAME || true
else
    echo "The main is not running with pm2."
fi
