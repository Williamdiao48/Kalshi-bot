"""
Test whether Kalshi accepts a plain API key header instead of RSA-PSS.

Usage:
    KALSHI_KEY_ID=your-key-id venv/bin/python scripts/test_kalshi_auth.py

Set KALSHI_ENVIRONMENT=demo to hit the demo environment instead of production.
"""

import asyncio
import os
import aiohttp
from dotenv import load_dotenv

load_dotenv()

KEY_ID = os.environ.get("KALSHI_KEY_ID", "")
ENV = os.environ.get("KALSHI_ENVIRONMENT", "production")
BASE_URL = (
    "https://demo-api.kalshi.co/trade-api/v2"
    if ENV == "demo"
    else "https://api.elections.kalshi.com/trade-api/v2"
)
TEST_PATH = "/markets?limit=1"


async def main():
    if not KEY_ID:
        print("ERROR: KALSHI_KEY_ID not set")
        return

    url = BASE_URL + TEST_PATH

    # Attempt 1: plain Authorization Bearer
    print("=== Attempt 1: Authorization: Bearer <key_id> ===")
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers={"Authorization": f"Bearer {KEY_ID}"}) as r:
            print(f"Status: {r.status}")
            print(await r.text())

    print()

    # Attempt 2: no auth at all (baseline — should definitely fail)
    print("=== Attempt 2: No auth headers (baseline) ===")
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as r:
            print(f"Status: {r.status}")
            print(await r.text())

    print()

    # Attempt 3: KALSHI-ACCESS-KEY header only (no signature)
    print("=== Attempt 3: KALSHI-ACCESS-KEY only (no signature) ===")
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers={"KALSHI-ACCESS-KEY": KEY_ID}) as r:
            print(f"Status: {r.status}")
            print(await r.text())


asyncio.run(main())
