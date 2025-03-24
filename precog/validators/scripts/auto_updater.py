import importlib.resources as pkg_resources
import subprocess
import time

import bittensor as bt
import git
from git.exc import GitCommandError

import precog

# Frequency of the auto updater in minutes
TIME_INTERVAL = 5


def git_apply_stash(repo: git.Repo, old_commit_hash: str) -> bool:

    try:
        # Apply the most recent stash
        # Preserve staged versus unstaged changes
        repo.git.stash("apply", "--index")

    except GitCommandError as e:
        bt.logging.debug(f"Error observed while applying stash: `{str(e)}`")
        bt.logging.debug("Rolling back...")

        bt.logging.debug(f"Rolling back to commit hash `{old_commit_hash}`")
        repo.git.reset("--hard", old_commit_hash)  # Reset to original state

        repo.git.stash("apply", "--index")  # Restore original changes
        bt.logging.debug("Reapplied stashed changes. Dropping stash now.")
        repo.git.stash("drop")  # Drop the most recent stash

        bt.logging.debug("Rolled back to original state with local changes.")
        bt.logging.debug(f"Currently on commit hash: {old_commit_hash}")

        # Return False if rollback was required
        return False

    # Applying the stash was successful
    else:
        bt.logging.debug("Successfully reapplied stashed changes. Dropping stash now.")

        # Drop the most recent stash
        repo.git.stash("drop")

        # Return True if stash was successfully applied
        return True


def git_pull(repo: git.Repo, max_retries: int = 3, retry_delay: int = 5) -> bool:
    # Try pulling with retries
    for attempt in range(max_retries):
        try:
            # Pull the latest changes from github
            repo.remotes.origin.pull(rebase=True)

        except Exception as e:
            if attempt < max_retries - 1:  # Don't sleep on the last attempt
                bt.logging.debug(f"Pull attempt {attempt + 1} failed: {str(e)}")
                bt.logging.debug(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                bt.logging.debug(f"All pull attempts failed. Last error: {str(e)}")

                return False

        else:
            bt.logging.debug("Pull complete.")
            return True


def main(path) -> bool:
    # Load the git repository
    repo = git.Repo(path)
    current_hash = repo.head.commit

    # Check for unstaged changes and cache if needed
    if repo.is_dirty():
        bt.logging.debug("Local changes detected. Stashing changes now.")
        repo.git.stash("push")
        stashed = True
    else:
        bt.logging.debug("No local changes detected. Stashing not required.")
        stashed = False

    # Try pulling with retries
    pull_success = git_pull(repo)

    stash_success = True
    if stashed:
        # Reapply the stash
        stash_success = git_apply_stash(repo, current_hash)

    # If we could not pull
    if not pull_success:

        # TODO: End the pm2 process?
        pass

    # If we pulled successfully but had to rollback
    elif not stash_success:
        raise RuntimeError(
            "Local changes are not compatible with new commits observed on GitHub. Manual intervention to update the code is required. Killing the auto updater pm2 process."
        )

    # Pull and stash succeeded
    else:
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
    has_changed = main(git_repo_path)

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
