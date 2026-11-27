# market-data-service

Streams live BTC/USDT trades from Binance, turns them into features (VWAP
distance, buy/sell imbalance, trading volume rate), and tests whether a small
neural network can predict short-term price direction better than simple
baselines.

> **Status:** the data pipeline, labelling, and a first evaluation are done.
> On about one day of data, none of the models reliably beat a coin flip
> (see [Results](#results)).

## How it works

1. Connects to Binance's public trade stream (no API key needed).
2. Keeps a sliding window of the 1,000 most recent trades.
3. Computes features over that window and writes a row to `data/features.csv`
   about once per second. Reconnects automatically if the connection drops.
4. Labels each row: is the price higher or lower 10 seconds later?
5. Splits the data by time, then compares baselines with a small neural network.

### Features

| Column | Meaning |
| --- | --- |
| `time_ms` | Time of the latest trade (Unix milliseconds) |
| `last_price` | Price of the latest trade |
| `vwap` | Volume-weighted average price over the window |
| `price_vs_vwap` | How far the latest price is above/below VWAP, as a fraction |
| `imbalance` | `(buy volume - sell volume) / total volume`, from -1 (all selling) to +1 (all buying) |
| `volume_rate` | BTC traded per second across the window |

Buy and sell volume use Binance's own flag for which side started each trade,
not a guess from price movement.

## Setup

Requires Python and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/rohans-git1107/market-data-service.git
cd market-data-service
uv sync
```

## Usage

Run these from the project's top folder.

Start logging features (stop with Ctrl+C):

```bash
uv run python -m market_data_service.stream
```

Label a copy of the data, then train and compare models:

```bash
cp data/features.csv data/snapshot.csv
uv run python -m market_data_service.label_data data/snapshot.csv data/labelled.csv
uv run python -m market_data_service.train data/snapshot.csv
```

Run the tests:

```bash
uv run python -m pytest
```

The `data/` folder is not tracked by git, so run the logger to generate your
own data.

## Method

- **Label:** 1 if the price is higher 10 seconds later, 0 if lower. Rows in the
  first 5 minutes after each logger start, rows with no price 10 seconds ahead
  (e.g. across a gap), and ties are dropped. Labels never look across a gap.
- **Split:** by time, not randomly. The earliest 60% of the data trains the
  models, the next 20% validates, and the latest 20% tests, with 5-minute gaps
  between them. This matters because neighbouring rows share most of their
  1,000-trade window, so a random split would leak the test period into
  training.
- **Models:** always predict the more common training label; logistic
  regression; a small PyTorch network (two hidden layers of 32 units). Inputs
  are standardised using training data only, and `volume_rate` is log-scaled.
  The network keeps the epoch with the best validation loss.

## Results

About 41,500 logged rows (roughly one day), 38,592 after labelling. Accuracy at
predicting price direction 10 seconds ahead:

| Model | Validation | Test |
| --- | --- | --- |
| Majority class | 0.515 | 0.485 |
| Logistic regression | 0.527 | 0.520 |
| Neural network | 0.516 | 0.529 |

**Interpretation:** none of the models reliably beats a coin flip (0.500).
The network's validation loss stayed at about 0.693, which is what random
guessing gives, so it learned little. The majority baseline moves between
0.515 and 0.485 only because the price drifted differently in the validation
and test periods. Neighbouring rows overlap heavily, so the effective test size
is much smaller than the row count, and differences of one to three points are
within noise. This is one day of one market, so it says little about other
periods.

## Project layout

```
src/market_data_service/
    features.py     # feature calculation (pure functions, unit tested)
    stream.py       # Binance websocket client and CSV logging
    label_data.py   # labels rows with future price direction
    train.py        # time split, baselines and neural network
tests/
    test_features.py
    test_label_data.py
```

## Notes and limitations

- The first few minutes after each start are a warm-up period: the window
  holds few trades, so features are unreliable. These rows are dropped when
  labelling. A reconnect also starts a fresh window.
- Rows are logged when a trade arrives after the interval has passed, so
  spacing is roughly 1 to 3 seconds, not exactly 1.
- Short-horizon crypto price prediction is very noisy. The aim of this project
  is a clean pipeline and honest evaluation, not a profitable strategy.

## Roadmap

- [x] Live trade stream with sliding window
- [x] Feature calculation with unit tests
- [x] CSV logging with automatic reconnect
- [x] Label rows (price up or down after 10 seconds)
- [x] Baselines: majority class and logistic regression
- [x] Small PyTorch neural network with a time-based train/validation/test split
- [ ] Collect several more days of data
- [ ] Add features over shorter windows (last 20 to 50 trades, last 10 seconds)
- [ ] Add an uncertainty estimate (e.g. bootstrap over blocks of time)