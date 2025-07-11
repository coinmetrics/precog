Release Notes
=============
2.5.3
-----
Released on July 10th 2025
- Decrease moving average alpha to 0.05

2.5.2
-----
Released on July 8th 2025
- Decrease ranking decay to 0.8
- Increase moving average alpha to 0.1

2.5.1
-----
Released on June 27th 2025
- Increase moving average alpha parameter to have a half life of 2 hours

2.5.0
-----
Released on June 25th 2025
- Streamline rewards calculation
    - Weights are now calculated solely from the moving average of rewards
- Improvements to validation logic
    - Ensure that requests are not sent to miners immediately on validator restart

2.4.2
-----
Released on May 19th 2025
- Ensure that interval predictions are being evaluated against an hour of prices

2.4.1
-----
Released on April 30th 2025
- Update how rewards mechanism handles tied evaluations

2.4.0
-----
Released on March 25th 2025
- Update dependencies to be compatible with latest Bittensor v9.1.0
- Enhance auto updater reliability and stashing mechanics

2.3.0
-----
Released on March 10th 2025
- Increase the evaluation window to include 6 hours of miner predictions

2.2.2
-----
Released on March 5th 2025
- Periodically clear the MinerHistory object
- Minimize subtensor instantiation by reusing the same subtensor initialized on miner/validator startup
- Only use the cm_data cache for the prediction interval data (24 hours in the base miner)

2.2.1
-----
Released on March 5th 2025
- Resolved timestamp rounding bug triggered by approaching midnight

2.2.0
-----
Released on March 3rd 2025
- Implemented validator auto updater to pull latest changes from GitHub

2.1.0
-----
Released on February 24th 2025
- Improved validator state management
- Validator request timestamps are now identical

2.0.0
-----
Released on February 13th 2025
- Upgrade to bittensor v9.0.0
- Upgrade to bittensor-cli v9.0.0
- Make minimal changes to support dTAO logic

1.0.4
-----
Released on February 3rd 2025
- Add wandb logging
- Add API cache for 24 hours of data to speed up base miner responses
- Implement dendrite.forward to improve validator performance

1.0.3
-----
Released on January 16th 2025
- Ensure metagraph is properly synced when new validators join the network.

1.0.2
-----
Released on January 16th 2025
- Ensure metagraph is properly synced when new miners join the network

1.0.1
-----
Released on January 15th 2025
- Update to support finney network
- Add support for running local subtensor node

1.0.0
-----
Released on January 14th 2025
- Release to mainnet

0.3.0
-----
Released on January 7th 2025
- Update to bittensor v8.5.1
- Implement CR3

0.2.0
-----
Released on December 20th 2024
- Finalized README instructions
- Cleaned and documented code base


0.1.0
-----
Released on December 10th 2024
- Successful deployment on testnest
- Implemented naive base miner
- Leveraged poetry dependency management
