
"""
Feature computation for a sliding window of trades. (used deque)

A trade is a tuple: (price, quantity, time_ms, is_buy)
where is_buy is True when the buyer started the trade (the taker was a buyer).
"""


def compute_features(trades):
    """
    Return a dict of features for the given trades, or None if there is
    not enough data yet. Returning None (instead of zeros) stops fake
    "all zero" rows from ending up in the training data.
    """
    if len(trades) < 2:
        return None

    total_qty = 0.0
    sum_price_qty = 0.0
    buy_qty = 0.0

    for price, qty, _, is_buy in trades:
        total_qty += qty
        sum_price_qty += price * qty
        if is_buy:
            buy_qty += qty

    if total_qty <= 0:
        return None

    sell_qty = total_qty - buy_qty
    vwap = sum_price_qty / total_qty
    last_price = trades[-1][0]

    # Ranges from -1 (all selling) to +1 (all buying).
    imbalance = (buy_qty - sell_qty) / total_qty

    # How far the latest price sits above/below the VWAP, as a fraction.
    
    price_vs_vwap = (last_price - vwap) / vwap

    # Volume traded per second across the window.
    elapsed_ms = trades[-1][2] - trades[0][2]
    volume_rate = total_qty / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0.0

    return {
        "time_ms": trades[-1][2],
        "last_price": last_price,
        "vwap": vwap,
        "price_vs_vwap": price_vs_vwap,
        "imbalance": imbalance,
        "volume_rate": volume_rate,
    }
