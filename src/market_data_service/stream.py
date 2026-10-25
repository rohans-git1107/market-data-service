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
            e_time = int(data.get("E")) 
            
            trade_data.append((p, q, e_time))
            print(MarketTradeData(trade_data))


def MarketTradeData(trade_data):

    if len(trade_data) < 2:
        return {
            "vwap": 0.0,
            "imbalance": 0.0,
            "velocity": 0.0
        }

    total_quantity = 0.0
    sum_pq = 0.0
    
    buy_quantity = 0.0
    sell_quantity = 0.0
    
    prev_price = 0
    oldest_time = 0
    newest_time = 0


    for item in trade_data:
        p, q, e_time = item
        

        if oldest_time == 0:
            oldest_time = e_time
        newest_time = e_time


        total_quantity += q
        sum_pq += p * q
        

        if prev_price != 0:
            if p > prev_price:
                buy_quantity += q
            elif p < prev_price:
                sell_quantity += q
            else:
                buy_quantity += q / 2.0
                sell_quantity += q / 2.0
        else:

            buy_quantity += q / 2.0
            sell_quantity += q / 2.0
            
        prev_price = p

    if total_quantity > 0:
        vwap = (sum_pq / total_quantity)
    else:
        vwap = 0.0

    total_buy_sell = buy_quantity + sell_quantity
    if total_buy_sell > 0:
        imbalance = ((buy_quantity - sell_quantity) / total_buy_sell) 
    else: 
        imbalance = 0.0

    time_delta_ms = newest_time - oldest_time
    if time_delta_ms > 0:
        velocity = total_quantity / (time_delta_ms / 1000.0)
    else:
        velocity = 0.0

    return {
        "vwap": vwap,
        "imbalance": imbalance,
        "velocity": velocity
    }


if __name__ == "__main__":
    asyncio.run(listening_to_binance_stream())