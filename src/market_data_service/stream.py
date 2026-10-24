import asyncio
import json
import websockets
from collections import deque

async def listening_to_binance_stream():
    symbol = "btcusdt"
    url = f"wss://stream.binance.com:9443/ws/{symbol}@trade"

    trade_data = deque(maxlen=1000)
    async with websockets.connect(url) as websocket:
         async for message in websocket:
            data = json.loads(message)
            p = float(data.get("p"))
            q = float(data.get("q"))
            trade_data.append((p, q))
            print(rolling_vwap(trade_data))


def rolling_vwap(trade_data):
    total_quantity = 0
    sum_pq = 0
    for price,quantity in trade_data:
        total_quantity += quantity
        sum_pq += price * quantity

    if total_quantity == 0:
        return 0.0

    return sum_pq/total_quantity

if __name__ == "__main__":
    asyncio.run(listening_to_binance_stream())