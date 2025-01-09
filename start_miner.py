import subprocess

from precog.utils.classes import Config
from precog.utils.general import parse_arguments


def main(config):
    start_command = ["pm2", "start", "--name", f"{config.neuron.name}"]
    arguments = "python3 -m precog.miners.miner" + config.to_str()

    start_command.append(arguments)
    subprocess.run(start_command)


if __name__ == "__main__":
    args = parse_arguments()
    config = Config(args)
    config.neuron.type = "Miner"
    try:
        main(config)
    except Exception as e:
        print(e)
