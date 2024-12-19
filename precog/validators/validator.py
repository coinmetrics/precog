import argparse
import asyncio
from pathlib import Path

from precog.utils.config import config
from precog.validators.weight_setter import weight_setter


class Validator:
    def __init__(self):
        self.config = config(parser=argparse.ArgumentParser(), neuron_type="validator")
        print(self.config)
        full_path = Path(
            f"{self.config.logging.logging_dir}/{self.config.wallet.name}/{self.config.wallet.hotkey}/netuid{self.config.netuid}/validator"
        ).expanduser()
        full_path.mkdir(parents=True, exist_ok=True)
        self.config.full_path = str(full_path)

    async def main(self):
        loop = asyncio.get_event_loop()
        self.weight_setter = weight_setter(config=self.config, loop=loop)

    async def reset_instance(self):
        self.__init__()
        asyncio.run(self.main())


# Run the validator.
if __name__ == "__main__":
    validator = Validator()
    asyncio.run(validator.main())
