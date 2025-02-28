import importlib.resources as pkg_resources
import logging
import subprocess
import time
from datetime import timedelta

import git

import precog
from precog.utils.timestamp import elapsed_seconds, get_now

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TIME_INTERVAL = 5


def git_pull_change(path) -> bool:
    repo = git.Repo(path)
    current_hash = repo.head.commit

    # Check if there are local changes
    must_stash = repo.is_dirty()

    # Stash the changes
    if must_stash:
        logger.info("Local changes detected. Stashing...")
        repo.git.stash("save")

    # Pull the latest changes from github
    repo.remotes.origin.pull(rebase=True)
    logger.info("Pull complete.")
    new_hash = repo.head.commit

    # Apply stash if exists
    if must_stash:
        logger.info("Applying stash...")
        repo.git.stash("pop")

    logger.info(f"Current hash: {current_hash}")
    logger.info(f"New hash: {new_hash}")

    # Return True if the hash has changed
    return current_hash != new_hash


if __name__ == "__main__":
    logger.info("Starting auto updater...")

    # Get the path to the precog directory
    with pkg_resources.path(precog, "..") as p:
        git_repo_path = p

    # Loop until we observe github activity
    while True:

        # Get current timestamp
        now = get_now()

        # Check if the current minute is 2 minutes past anticipated validator query time
        if now.minute % TIME_INTERVAL == 2:

            logger.info(f"Current time: {now}")

            logger.info("Checking for repository changes...")

            # Pull the latest changes from github
            has_changed = git_pull_change(git_repo_path)

            # If the repo has changed, break the loop
            if has_changed:
                logger.info("Repository has changed!")
                break

            # If the repo has not changed, sleep
            else:
                logger.info("Repository has not changed. Sleep mode activated.")

                # Calculate the time of the next git pull check
                next_check = now + timedelta(minutes=TIME_INTERVAL)
                next_check = next_check.replace(second=0)

                # Determine the number of seconds to sleep
                seconds_to_sleep = elapsed_seconds(get_now(), next_check)

                # Sleep for the exact number of seconds to the next git pull check
                time.sleep(seconds_to_sleep)
        else:

            # Sleep for 45 seconds
            # This is to prevent the script from checking for changes too frequently
            # This specific `else` block should not be reach too often since we sleep for the exact time of the anticipated validator query time
            logger.info("Sleeping for 45 seconds")
            time.sleep(45)

    # This code is only reached when the repo has changed
    # We can now restart both pm2 processes, including the auto updater
    # Let the script simply end and the new process will be restarted by pm2
    logger.info("Restarting pm2 processes...")
    subprocess.run(["pm2", "restart", "app.config.js"], cwd=git_repo_path)
