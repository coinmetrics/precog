import asyncio
import base64
import random
import traceback
from concurrent.futures import ThreadPoolExecutor

import bittensor as bt
import httpx
import uvicorn
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from fastapi import Depends, FastAPI, HTTPException, Request

from precog import __spec_version__
from precog.protocol import Challenge


class Server:
    def __init__(self, validator):
        self.validator = validator
        # self.get_credentials()
        self.app = FastAPI()
        self.app.add_api_route(
            "/predict",
            self.organic_predict,
            methods=["POST"],
            dependencies=[Depends(self.get_self)],
        )
        self.start_server()

    async def organic_predict(self, request: Request):
        authorization: str = request.headers.get("authorization")
        if not authorization:
            # raise HTTPException(status_code=401, detail="Authorization header missing")
            bt.logging.debug("No authorization header. Processing request anyway")
        else:
            self.authenticate_token(authorization)
        bt.logging.debug("Received organic request")
        payload = await request.json()

        timestamp = payload["timestamp"]
        synapse = Challenge(timestamp=timestamp)
        future = asyncio.run_coroutine_threadsafe(
                self.validator.dendrite.forward(
                # Send the query to selected miner axons in the network.
                axons=[self.validator.metagraph.axons[uid] for uid in self.validator.available_uids],
                synapse=synapse,
                deserialize=False,
                timeout=self.validator.config.neuron.timeout,
            ),
            loop=self.validator.loop,
        )
        results = future.result()
        data = {uid: {"prediction": synapse.prediction, "interval": synapse.interval} for uid, synapse in zip(self.validator.available_uids, results)}
        return data

    def authenticate_token(self, public_key_bytes):
        public_key_bytes = base64.b64decode(public_key_bytes)
        try:
            self.verify_credentials(public_key_bytes)
            bt.logging.info("Successfully authenticated token")
            return public_key_bytes
        except Exception as e:
            bt.logging.error(f"Exception occured in authenticating token: {e}")
            bt.logging.error(traceback.print_exc())
            raise HTTPException(status_code=401, detail="Error getting authentication token")

    def get_credentials(self):
        with httpx.Client(timeout=httpx.Timeout(30)) as client:
            response = client.post(
                f"{self.validator.config.server.proxy_client_url}/get-credentials",
                json={
                    "postfix": (
                        f":{self.validator.config.server.port}/validator_proxy"
                        if self.validator.config.server.port
                        else ""
                    ),
                    "uid": self.validator.uid,
                },
            )
        response.raise_for_status()
        response = response.json()
        message = response["message"]
        signature = response["signature"]
        signature = base64.b64decode(signature)

        def verify_credentials(public_key_bytes):
            public_key = Ed25519PublicKey.from_public_bytes(public_key_bytes)
            try:
                public_key.verify(signature, message.encode("utf-8"))
            except InvalidSignature:
                raise Exception("Invalid signature")

        self.verify_credentials = verify_credentials

    async def get_self(self):
        return self

    def start_server(self):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.executor.submit(uvicorn.run, self.app, host="0.0.0.0", port=self.validator.config.server.port)
