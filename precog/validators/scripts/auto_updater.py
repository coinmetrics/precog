import importlib.resources as pkg_resources
import subprocess
import time
from datetime import timedelta

import bittensor as bt
import git

import precog
from precog.utils.timestamp import elapsed_seconds, get_now

# Frequency of the auto updater in minutes
TIME_INTERVAL = 5


def git_pull_change(path, max_retries=3, retry_delay=5) -> bool:
    # Load the git repository
    repo = git.Repo(path)
    current_hash = repo.head.commit

    # Kill auto update if there are local changes
    if repo.is_dirty():
        bt.logging.debug("Local changes detected.")
        bt.logging.debug("Only run auto update if there are no local changes.")
        bt.logging.debug("Killing the auto updater.")
        raise RuntimeError("Local changes detected. Auto update killed")

    # Try pulling with retries
    for attempt in range(max_retries):
        try:
            # Pull the latest changes from github
            repo.remotes.origin.pull(rebase=True)
            bt.logging.debug("Pull complete.")
            break
        except Exception as e:
            if attempt < max_retries - 1:  # Don't sleep on the last attempt
                bt.logging.debug(f"Pull attempt {attempt + 1} failed: {str(e)}")
                bt.logging.debug(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                bt.logging.debug(f"All pull attempts failed. Last error: {str(e)}")
                raise  # Re-raise the last exception if all retries failed

    new_hash = repo.head.commit

    bt.logging.debug(f"Current hash: {current_hash}")
    bt.logging.debug(f"New hash: {new_hash}")

    # Return True if the hash has changed
    return current_hash != new_hash


if __name__ == "__main__":
    bt.logging.set_debug()
    bt.logging.debug("Starting auto updater...")

    # Get the path to the precog directory
    with pkg_resources.path(precog, "..") as p:
        git_repo_path = p

    # Loop until we observe github activity
    while True:

        # Get current timestamp
        now = get_now()

        # Check if the current minute is 2 minutes past anticipated validator query time
        if now.minute % TIME_INTERVAL == 2:

            bt.logging.debug("Checking for repository changes...")

            # Pull the latest changes from github
            has_changed = git_pull_change(git_repo_path)

            # If the repo has changed, break the loop
            if has_changed:
                bt.logging.debug("Repository has changed!")
                break

            # If the repo has not changed, sleep
            else:
                bt.logging.debug("Repository has not changed. Sleep mode activated.")

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
            bt.logging.debug("Sleeping for 45 seconds")
            time.sleep(45)

    # This code is only reached when the repo has changed
    # We can now restart both pm2 processes, including the auto updater
    # Let the script simply end and the new process will be restarted by pm2
    bt.logging.debug("Installing dependencies...")
    subprocess.run(["poetry", "install"], cwd=git_repo_path)
    bt.logging.debug("Restarting pm2 processes...")
    subprocess.run(["pm2", "restart", "app.config.js"], cwd=git_repo_path)
