import subprocess
import time

import precog
from precog.utils.classes import Config
from precog.utils.general import get_version, parse_arguments

webhook_url = ""
current_version = precog.__version__


def update_and_restart(config):
    global current_version
    start_command = ["pm2", "start", "--name", f"{config.neuron.name}"]
    arguments = "python3 -m precog.validators.validator" + config.to_str()

    start_command.append(arguments)
    subprocess.run(start_command)
    if config.autoupdate:
        while True:
            latest_version = get_version()
            print(f"Current version: {current_version}")
            print(f"Latest version: {latest_version}")
            if current_version != latest_version and latest_version is not None:
                print("Updating to the latest version...")
                subprocess.run(["pm2", "delete", config.neuron.name])
                subprocess.run(["git", "reset", "--hard"])
                subprocess.run(["git", "pull"])
                subprocess.run(["pip", "install", "-e", "."])
                subprocess.run(start_command)
                current_version = latest_version
            print("All up to date!")
            time.sleep(300)


if __name__ == "__main__":
    args = parse_arguments()
    config = Config(args)
    try:
        update_and_restart(config)
    except Exception as e:
        print(e)
