import pandas as pd

from market_data_service.label_data import label_data

# Settings that make hand-calculation easy: no warm-up, 10 s horizon.
KWARGS = dict(horizon_ms=10_000, max_gap_ms=10_000, warmup_ms=0, slack_ms=5_000)


def make_df(times_and_prices):
    return pd.DataFrame(
        {"time_ms": [t for t, _ in times_and_prices],
         "last_price": [p for _, p in times_and_prices]}
    )


def test_labels_match_hand_calculation():
    df = make_df([(0, 100), (5000, 101), (10000, 99), (15000, 102), (20000, 98)])
    out = label_data(df, **KWARGS)

    # t=0:     10 s later price is 99  vs 100 -> down (0)
    # t=5000:  10 s later price is 102 vs 101 -> up   (1)
    # t=10000: 10 s later price is 98  vs 99  -> down (0)
    # t=15000 and t=20000 have no row 10 s later -> dropped
    assert list(out["time_ms"]) == [0, 5000, 10000]
    assert list(out["label"]) == [0, 1, 0]


def test_ties_are_dropped():
    df = make_df([(0, 100), (10000, 100), (20000, 105)])
    out = label_data(df, **KWARGS)
    # t=0 -> 100 vs 100 is a tie (dropped); t=10000 -> 105 vs 100 is up
    assert list(out["time_ms"]) == [10000]
    assert list(out["label"]) == [1]


def test_never_labels_across_a_gap_in_the_data():
    df = make_df([(0, 100), (5000, 101), (10000, 102),
                  (100000, 50), (105000, 51), (110000, 52)])
    out = label_data(df, **KWARGS)
    # Only t=0 and t=100000 have a valid row 10 s later inside the same segment.
    assert list(out["time_ms"]) == [0, 100000]
    assert list(out["label"]) == [1, 1]


def test_warmup_rows_are_dropped():
    df = make_df([(0, 100), (5000, 101), (10000, 102), (15000, 103), (20000, 104)])
    out = label_data(df, **{**KWARGS, "warmup_ms": 6000})
    # Rows at t=0 and t=5000 are inside the 6 s warm-up and are removed.
    assert 0 not in list(out["time_ms"])
    assert 5000 not in list(out["time_ms"])