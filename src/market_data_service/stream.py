
"""
Stream live BTC/USDT trades from Binance and log features to a CSV file.
"""

import asyncio
import csv
import json
import os
import time
from collections import deque

import websockets

from market_data_service.features import compute_features

SYMBOL = "btcusdt"
URL = f"wss://stream.binance.com:9443/ws/{SYMBOL}@trade"
WINDOW_SIZE = 1000        # number of most recent trades kept
LOG_INTERVAL_S = 1.0      # write at most one feature row per second
CSV_PATH = "data/features.csv"
FIELDS = [
    "time_ms", "last_price", "vwap",
    "price_vs_vwap", "imbalance", "volume_rate",
]


async def stream_once(writer, csv_file):
    """Connect, read trades until the connection drops, log features."""
    # A fresh window for new trades (FIFO)
    trades = deque(maxlen=WINDOW_SIZE)
    last_log = 0.0

    async with websockets.connect(URL) as websocket:
        async for message in websocket:
            data = json.loads(message)

            # Skip anything that is not a trade message.
            if "p" not in data or "q" not in data or "m" not in data:
                continue

            price = float(data["p"])
            qty = float(data["q"])
            time_ms = int(data.get("T", data.get("E")))
            # "m" is True when the buyer is the market maker, i.e. the
            # seller started the trade. So a buyer-started trade is "not m".
            is_buy = not data["m"]
            trades.append((price, qty, time_ms, is_buy))

            now = time.monotonic()
            if now - last_log >= LOG_INTERVAL_S:
                features = compute_features(trades)
                if features is not None:
                    writer.writerow(features)
                    csv_file.flush()
                    print(features)
                last_log = now


async def main():
    os.makedirs("data", exist_ok=True)
    is_new_file = not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0
    with open(CSV_PATH, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        if is_new_file:
            writer.writeheader()

        # Binance closes connections now and then, so reconnect forever.
        while True:
            try:
                await stream_once(writer, csv_file)
            except (websockets.exceptions.ConnectionClosed, OSError) as err:
                print(f"Connection problem ({err!r}), reconnecting in 1s")
                await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
