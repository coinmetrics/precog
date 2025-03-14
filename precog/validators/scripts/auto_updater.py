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

    # Check for unstaged changes and cache if needed
    if repo.is_dirty():
        bt.logging.debug("Local changes detected. Stashing changes now.")
        repo.git.stash('push')
        stashed = True
    else:
        stashed = False

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

    # If there are no changes observed on GitHub
    if current_hash == new_hash:
        bt.logging.debug("No new commits on GitHub.")

        # Reapply the stash
        if stashed:
            repo.git.stash("apply", "--index")
            bt.logging.debug("Successfully reapplied stashed changes.")

        # Return False if the hash has not changed
        return False
    
    else:
        bt.logging.debug("New commits observed on GitHub.")

        # Reapply the stash
        if stashed:
            try:
                repo.git.stash("apply", "--index")
                bt.logging.debug("Successfully reapplied stashed changes.")
            except GitCommandError as e:
                bt.logging.debug("Conflicts while reapplying stash. Rolling back...")
                
                repo.git.reset('--hard', current_hash)  # Reset to original state
                repo.git.stash("apply", "--index")  # Restore original changes
                
                bt.logging.debug("Rolled back to original state with local changes.")
                bt.logging.debug(f"Currently on commit hash: {current_hash}")
                bt.logging.debug("Ending the auto update pm2 process. Human intervention is required to resolve merge conflicts.")
                
                return None
            
        # Return True if the hash has changed
        return True


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
