# market-data-service

Streams live BTC/USDT trades from Binance and turns them into features
(VWAP distance, buy/sell imbalance, trading volume rate) logged to a CSV file.
The logged data is intended as training data for a small neural network that
predicts short-term price direction.

> Current Status: Collecting data for ML Model (SEE ROADMAP)


## How it works

1. Connects to Binance's public trade stream (no API key needed).
2. Keeps a sliding window of the 1,000 most recent trades.
3. Computes features over that window and writes a row to `data/features.csv`
   about once per second.
4. Reconnects automatically if the connection drops.

### Features

| `time_ms` | Time of the latest trade (Unix milliseconds) |
| `last_price` | Price of the latest trade |
| `vwap` | Volume-weighted average price over the window |
| `price_vs_vwap` | How far the latest price is above/below VWAP, as a fraction |
| `imbalance` | `(buy volume - sell volume) / total volume`, from -1 (all selling) to +1 (all buying) |
| `volume_rate` | BTC traded per second across the window |

Buy and sell volume use Binance's own flag for which side started each trade,
not a guess from price movement.

## Setup

Requires Python and [uv]

```bash
git clone https://github.com/rohans-git1107/market-data-service.git
cd market-data-service
uv sync
```

## Usage

Start logging features:

```bash
uv run python -m market_data_service.stream
```

Rows are printed as they are written to `data/features.csv`. Stop with Ctrl+C.
The `data/` folder is not tracked by git, so run the logger to generate your
own data.

Run the tests:

```bash
uv run pytest
```

## Project layout

```
src/market_data_service/
    features.py   # (pure functions, unit tested)
    stream.py     # Binance websocket client and CSV logging
tests/
    test_features.py
```

## Notes and limitations

- The first few minutes after each start are a warm-up period: the window
  holds few trades, so features are unreliable. Drop these rows before
  training. A reconnect also starts a fresh window.
- Rows are logged when a trade arrives after the interval has passed, so
  spacing is roughly 1 to 3 seconds, not exactly 1.
- Short-horizon crypto price prediction is very noisy. The aim of this project
  is to have a clean pipeline and an honest evalutaion. It is not meant to be a profitable strategy.

## Roadmap

- [x] Use sliding window for live trade stream
- [x] Feature calculation with unit tests
- [x] CSV logging with automatic reconnect
- [ ] Collect data for ML Model
- [ ] Label rows (price up or down after a fixed time horizon)
- [ ] Baselines: majority class and logistic regression
- [ ] Small PyTorch neural network, evaluated with a time-based train/test split