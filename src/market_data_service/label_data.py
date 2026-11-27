"""Turn logged features into a labelled dataset.

For each row, the label is 1 if the price is higher `horizon_ms` later,
0 if it is lower. Rows are dropped when:
  - they fall in a warm-up period after the logger (re)started,
  - there is no row close enough to "now + horizon" (e.g. a gap in the data),
  - the price is exactly unchanged (a tie).

Run with:
    python -m market_data_service.label_data data/snapshot.csv data/labelled.csv
"""

import sys

import numpy as np
import pandas as pd

HORIZON_MS = 10_000   # predict 10 seconds ahead
MAX_GAP_MS = 10_000   # a gap bigger than this means the logger was restarted
WARMUP_MS = 300_000   # ignore the first 5 minutes after each (re)start
SLACK_MS = 5_000      # the "future" row may be at most 5s later than the target


def label_data(
    df,
    horizon_ms=HORIZON_MS,
    max_gap_ms=MAX_GAP_MS,
    warmup_ms=WARMUP_MS,
    slack_ms=SLACK_MS,
):
    """Return a copy of df with `future_return` and `label` columns added."""
    df = df.sort_values("time_ms").reset_index(drop=True)

    # A big jump in time means the logger stopped and restarted.
    # Each unbroken stretch of data gets its own segment number.
    df["segment"] = (df["time_ms"].diff() > max_gap_ms).cumsum()

    # Drop the unreliable warm-up rows at the start of every segment.
    segment_start = df.groupby("segment")["time_ms"].transform("min")
    df = df[df["time_ms"] - segment_start >= warmup_ms]

    labelled = []
    for _, seg in df.groupby("segment"):
        t = seg["time_ms"].to_numpy()
        price = seg["last_price"].to_numpy()

        # For each row, find the first row at least `horizon_ms` later.
        idx = np.searchsorted(t, t + horizon_ms)
        in_range = idx < len(t)
        idx = np.minimum(idx, len(t) - 1)
        close_enough = (t[idx] - (t + horizon_ms)) <= slack_ms

        seg = seg.assign(
            future_return=(price[idx] - price) / price,
            valid=in_range & close_enough,
        )
        labelled.append(seg)

    if not labelled:
        return df.assign(future_return=[], label=[]).drop(columns=["segment"])

    out = pd.concat(labelled)
    out = out[out["valid"] & (out["future_return"] != 0)].copy()
    out["label"] = (out["future_return"] > 0).astype(int)
    return out.drop(columns=["valid"]).reset_index(drop=True)


def main(in_path, out_path):
    raw = pd.read_csv(in_path)
    out = label_data(raw)
    out.to_csv(out_path, index=False)

    n_up = int(out["label"].sum())
    majority = max(n_up, len(out) - n_up) / len(out) if len(out) else float("nan")
    print(f"Read {len(raw)} rows, kept {len(out)} labelled rows")
    print(f"Unbroken segments of data: {raw['time_ms'].diff().gt(MAX_GAP_MS).sum() + 1}")
    print(f"Price up: {n_up}, down: {len(out) - n_up}")
    print(f"Always guessing the majority class scores: {majority:.3f}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python -m market_data_service.label_data IN.csv OUT.csv")
    main(sys.argv[1], sys.argv[2])