import importlib.resources as pkg_resources
import subprocess
import time

import bittensor as bt
import git

import precog
from precog.utils.timestamp import elapsed_seconds, get_now
from git.exc import GitCommandError

# Frequency of the auto updater in minutes
TIME_INTERVAL = 5



def git_pull_change(path, max_retries=3, retry_delay=5) -> bool:
    # Load the git repository
    repo = git.Repo(path)
    current_hash = repo.head.commit

    # Notify of observed local changes
    if repo.is_dirty():
        bt.logging.debug("Local changes detected.")

    # Try pulling with retries
    for attempt in range(max_retries):
        try:
            # Pull the latest changes from github
            repo.remotes.origin.pull(rebase=True)
            bt.logging.debug("Pull complete.")
            break

        except GitCommandError as e:
            if "merge conflicts" in str(e):
                bt.logging.debug("Merge conflicts detected. Reverting to the previous state.")
                repo.git.merge('--abort')
                bt.logging.debug("Stopping the pm2 process for the auto updater.")
                return

            else:
                bt.logging.debug(f"An error occurred: {e}")
                if attempt < max_retries - 1:  # Don't sleep on the last attempt
                    bt.logging.debug(f"Pull attempt {attempt + 1} failed: {str(e)}")
                    bt.logging.debug(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    bt.logging.debug(f"All pull attempts failed. Last error: {str(e)}")
                    raise  # Re-raise the last exception if all retries failed

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

    bt.logging.debug("Checking for repository changes...")

    # Pull the latest changes from github
    has_changed = git_pull_change(git_repo_path)

    # If the repo has not changed
    if not has_changed:
        bt.logging.debug("Repository has not changed. Sleep mode activated.")

    # If the repository has changed
    else:
        bt.logging.debug("Repository has changed!")

        # We can now restart both pm2 processes, including the auto updater
        bt.logging.debug("Installing dependencies...")
        subprocess.run(["poetry", "install"], cwd=git_repo_path)
        bt.logging.debug("Restarting pm2 processes...")
        subprocess.run(["pm2", "restart", "app.config.js"], cwd=git_repo_path)
