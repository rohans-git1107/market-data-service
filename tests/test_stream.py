
import pytest

from market_data_service.features import compute_features


def test_not_enough_trades_returns_none():
    assert compute_features([]) is None
    assert compute_features([(100.0, 1.0, 0, True)]) is None


def test_features_match_hand_calculation():
    # Trade 1: price 100, qty 1, buyer-started
    # Trade 2: price 102, qty 3, seller-started, 1 second later
    trades = [
        (100.0, 1.0, 0, True),
        (102.0, 3.0, 1000, False),
    ]
    f = compute_features(trades)

    # VWAP = (100*1 + 102*3) / 4 = 406 / 4 = 101.5
    assert f["vwap"] == pytest.approx(101.5)
    # Imbalance = (buy 1 - sell 3) / 4 = -0.5
    assert f["imbalance"] == pytest.approx(-0.5)
    # Volume rate = 4 units over 1 second = 4.0
    assert f["volume_rate"] == pytest.approx(4.0)
    # Last price 102 vs VWAP 101.5 -> 0.5 / 101.5
    assert f["price_vs_vwap"] == pytest.approx(0.5 / 101.5)
    assert f["last_price"] == 102.0


def test_all_buying_gives_imbalance_of_one():
    trades = [(100.0, 2.0, 0, True), (101.0, 3.0, 500, True)]
    assert compute_features(trades)["imbalance"] == pytest.approx(1.0)
