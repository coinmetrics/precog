# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Dict, List, Optional

import bittensor as bt
import pydantic


class Challenge(bt.Synapse):
    """ """

    # Required request input, filled by sending dendrite caller.
    timestamp: str = pydantic.Field(
        ...,
        title="Timestamps",
        description="The timestamp to predict from",
        allow_mutation=False,
    )

    # Assets to predict, filled by sending dendrite caller.
    assets: List[str] = pydantic.Field(
        default=["btc"],
        title="Assets",
        description="The list of assets to predict (e.g., ['btc', 'eth', 'tao_bittensor'])",
        allow_mutation=False,
    )

    # Optional request output, filled by recieving axon.
    predictions: Optional[Dict[str, float]] = pydantic.Field(
        default=None,
        title="Predictions",
        description="The predictions for each asset",
    )

    # Optional request output, filled by recieving axon.
    intervals: Optional[Dict[str, List[float]]] = pydantic.Field(
        default=None,
        title="Intervals",
        description="The predicted intervals for each asset. Formatted as {asset: [min, max]}",
    )

    def deserialize(self) -> Dict[str, float]:
        """
        Deserialize the predictions output.

        Returns:
        - Dict[str, float]: The deserialized response containing predictions for each asset.
        """
        return self.predictions
