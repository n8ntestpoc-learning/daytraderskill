import numpy as np
import pandas as pd
import pytest

import core
from core import Config


def make_session(day="2026-10-19", seed=0, drift=0.0, n=78, base=360.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(f"{day} 09:30", periods=n, freq="5min", tz=core.NY)
    close = base * np.exp(np.cumsum(rng.normal(drift, 0.0008, n)))
    open_ = np.r_[base, close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.0008, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.0008, n))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 1000}, index=idx)


def sessions(k=20):
    days = pd.bdate_range("2026-09-01", periods=k)
    return {d.date(): make_session(str(d.date()), seed=i) for i, d in enumerate(days)}


def test_opening_range_and_orb_trigger():
    s = make_session()
    hi, lo = core.opening_range(s)
    assert hi == s.iloc[:3]["high"].max() and lo == s.iloc[:3]["low"].min()
    s2 = s.copy()
    s2.iloc[3, s2.columns.get_loc("close")] = hi * 1.01
    st = core.orb_state(s2.iloc[:5])
    assert st["state"] in ("triggered", "expired") and st["trigger_price"] == pytest.approx(hi * 1.01)


def test_forming():
    assert core.orb_state(make_session().iloc[:2])["state"] == "forming"


def test_plan_long_and_short():
    p = core.make_plan("X", "long", 364.18, 362.18, 354.47, "or-mid", 1.0)
    assert p.stop == 358.32 and p.risk_per_share == 5.86 and p.target == 370.04  # matches the repo's TSLA example
    q = core.make_plan("X", "short", 100, 102, 99, "or-mid", 2.0)
    assert q.stop == 100.5 and q.target == 99.0
    with pytest.raises(ValueError):
        core.make_plan("X", "long", 350, 362, 354, "or-mid")


def test_budget_matches_repo_example():
    cfg = Config(account=5000, max_loss_per_trade=50)
    p = core.make_plan("TSLA", "long", 364.18, 362.18, 354.47)
    o = core.size_position(cfg, p)
    assert o["shares"] == 8 and o["position_value"] == pytest.approx(2913.44) and o["loss_at_stop"] == pytest.approx(46.88)


def test_risk_pass_and_blocks():
    cfg = Config(account=5000, max_loss_per_trade=50)
    p = core.make_plan("TSLA", "long", 364.18, 362.18, 354.47)
    o = core.size_position(cfg, p)
    t = pd.Timestamp("2026-10-19 10:20", tz=core.NY).to_pydatetime()
    assert core.risk_verdict(core.risk_check(cfg, o, p, 0, 0.0, t)) == "PASS"
    assert core.risk_verdict(core.risk_check(cfg, o, p, 3, 0.0, t)) == "BLOCK"      # trades/day
    assert core.risk_verdict(core.risk_check(cfg, o, p, 0, -140.0, t)) == "BLOCK"   # daily loss
    late = pd.Timestamp("2026-10-19 13:00", tz=core.NY).to_pydatetime()
    assert core.risk_verdict(core.risk_check(cfg, o, p, 0, 0.0, late)) == "BLOCK"   # window


def test_simulation_reproducible_and_sums():
    ses = sessions()
    p = core.make_plan("X", "long", 360, 362, 356, "or-mid", 1.0)
    a = core.simulate(ses, p, "10:15", 2000, seed=19)
    b = core.simulate(ses, p, "10:15", 2000, seed=19)
    assert a == b
    assert a["target_first"] + a["stop_first"] + a["open_at_close"] == pytest.approx(1.0)


def test_replay_history():
    r = core.replay_history(sessions(), "long", "or-mid", 1.0)
    assert r["fired"] == r["target"] + r["stop"] + r["open"]


def test_alerts_time_and_tags():
    p = core.make_plan("X", "long", 364.18, 362.18, 354.47)
    al = core.alerts_from_plan(p)
    assert [a["op"] for a in al] == ["<=", ">=", "time"]
    late = pd.Timestamp("2026-10-19 15:56", tz=core.NY).to_pydatetime()
    assert [a["kind"] for a in core.check_alerts([al[2]], late)] == ["flat"]
    assert "earnings" in core.tag_headline("Tesla Q3 earnings preview") and "deal" in core.tag_headline("Merger talks")


def test_rsi_bounds():
    s = pd.Series(np.linspace(100, 50, 60))
    assert core.rsi(s).iloc[-1] < 10
