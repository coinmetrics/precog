import argparse
import asyncio
from pathlib import Path

import bittensor as bt

from precog.utils.config import config
from precog.validators.weight_setter import weight_setter


class Validator:
    def __init__(self, config):
        self.config = config
        full_path = Path(
            f"{self.config.logging.logging_dir}/{self.config.wallet.name}/{self.config.wallet.hotkey}/netuid{self.config.netuid}/validator"
        ).expanduser()
        full_path.mkdir(parents=True, exist_ok=True)
        self.config.full_path = str(full_path)

    async def main(self):
        loop = asyncio.get_event_loop()
        self.weight_setter = await weight_setter.create(config=self.config, loop=loop)
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            # Clean shutdown: cancel all tasks except current task
            current_task = asyncio.current_task()
            tasks = [t for t in asyncio.all_tasks() if t is not current_task]

            # Cancel all other tasks
            for task in tasks:
                task.cancel()

            # Wait for all tasks to complete cancellation
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            # Re-raise the CancelledError to allow proper cleanup
            raise
        except Exception as e:
            import traceback

            bt.logging.error(f"Validator main loop error: {e}")
            bt.logging.error(f"Traceback: {traceback.format_exc()}")
            raise
        finally:
            # Properly cleanup weight_setter
            if hasattr(self, "weight_setter"):
                self.weight_setter.__exit__(None, None, None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = config(parser, neuron_type="validator")
    validator = Validator(config)
    try:
        asyncio.run(validator.main())
    except KeyboardInterrupt:
        pass
