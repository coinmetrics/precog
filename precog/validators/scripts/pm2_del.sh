#!/bin/bash

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
