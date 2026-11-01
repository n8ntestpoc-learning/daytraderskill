"""Core logic for the Day Trader Streamlit app.

Inspired by the 11-skill pipeline in Jakeschincariol/day-trader-skill:
scanner -> news -> entry -> exit -> timemachine -> budget -> risk -> broker,
with alerts / analyst / congress tracker alongside.

Data: free, unofficial public endpoints (Yahoo Finance chart + RSS, Nasdaq Trader
symbol lists, SEC EDGAR, House Clerk). Educational software, not financial advice.
"""
from __future__ import annotations

import io
import json
import math
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from datetime import datetime, time as dtime
from typing import Optional

import numpy as np
import pandas as pd
import requests

NY = "America/New_York"
UA = "Mozilla/5.0 (day-trader-app; educational)"
OR_MINUTES = 15  # opening range length
BAR_MIN = 5


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    account: float = 5000.0
    max_loss_per_trade: float = 50.0
    daily_loss_limit: float = 150.0
    max_trades_per_day: int = 3
    min_reward_risk: float = 1.0
    entry_window_start: str = "09:45"
    entry_window_end: str = "11:30"
    flat_by: str = "15:55"
    max_position_pct: float = 60.0  # % of account in one position
    paper: bool = True
    sec_user_agent: str = "Your Name you@example.com"


def _hm(s: str) -> dtime:
    h, m = s.split(":")
    return dtime(int(h), int(m))


# --------------------------------------------------------------------------- #
# Market data (Yahoo Finance chart endpoint)
# --------------------------------------------------------------------------- #
def _get(url, **kw):
    kw.setdefault("headers", {"User-Agent": UA})
    kw.setdefault("timeout", 15)
    return requests.get(url, **kw)


def fetch_bars(symbol: str, interval: str = "5m", rng: str = "60d") -> pd.DataFrame:
    """OHLCV bars indexed by New York time. Empty frame on failure."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    try:
        r = _get(url, params={"interval": interval, "range": rng, "includePrePost": "false"})
        r.raise_for_status()
        res = r.json()["chart"]["result"][0]
        q = res["indicators"]["quote"][0]
        idx = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(NY)
        df = pd.DataFrame(
            {k: q[k] for k in ("open", "high", "low", "close", "volume")}, index=idx
        ).dropna(subset=["open", "high", "low", "close"])
        df.index.name = "ts"
        return df
    except Exception:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


def split_sessions(df: pd.DataFrame) -> dict:
    """{date: bars of that regular session}."""
    if df.empty:
        return {}
    out = {}
    for d, g in df.groupby(df.index.date):
        g = g.between_time("09:30", "15:59")
        if len(g) >= 20:
            out[d] = g
    return out


# --------------------------------------------------------------------------- #
# Chart Scanner
# --------------------------------------------------------------------------- #
def opening_range(session: pd.DataFrame, minutes: int = OR_MINUTES):
    """(high, low) of the first `minutes` of the session, or None if still forming."""
    n = minutes // BAR_MIN
    if len(session) < n:
        return None
    first = session.iloc[:n]
    return float(first["high"].max()), float(first["low"].min())


def orb_state(session: pd.DataFrame, direction: str = "long"):
    """Return dict(state, trigger_ts, trigger_price, or_high, or_low, last, ...).

    state: forming | waiting | triggered | expired (price came back inside the range
    after triggering) . Trigger = first 5-minute close beyond the range after it formed.
    """
    n = OR_MINUTES // BAR_MIN
    orr = opening_range(session)
    if orr is None:
        return {"state": "forming"}
    hi, lo = orr
    after = session.iloc[n:]
    last = float(session["close"].iloc[-1])
    res = {"or_high": hi, "or_low": lo, "or_mid": (hi + lo) / 2, "last": last, "direction": direction}
    cond = (after["close"] > hi) if direction == "long" else (after["close"] < lo)
    hits = after[cond]
    if hits.empty:
        res["state"] = "waiting"
        return res
    t = hits.index[0]
    px = float(hits["close"].iloc[0])
    res.update(trigger_ts=t, trigger_price=px)
    inside = lo <= last <= hi
    res["state"] = "expired" if inside else "triggered"
    res["pct_beyond"] = ((last / hi - 1) if direction == "long" else (lo / last - 1)) * 100
    return res


def scan_symbol_orb(symbol: str, direction: str = "long") -> Optional[dict]:
    df = fetch_bars(symbol, "5m", "1d")
    ses = split_sessions(df) if len(df) else {}
    if not ses:
        # intraday partial sessions can be < 20 bars; fall back to raw
        if df.empty:
            return None
        ses = {df.index[-1].date(): df.between_time("09:30", "15:59")}
    day = ses[max(ses)]
    s = orb_state(day, direction)
    if s["state"] in ("forming", "waiting"):
        return None
    return {"symbol": symbol, **{k: s[k] for k in ("or_high", "or_low", "last", "trigger_price", "pct_beyond", "state")},
            "trigger_time": s["trigger_ts"].strftime("%H:%M")}


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def scan_symbol_daily(symbol: str) -> Optional[dict]:
    """Pre-market style daily setups: oversold bounce and tight flag near highs."""
    df = fetch_bars(symbol, "1d", "6mo")
    if len(df) < 60:
        return None
    c = df["close"]
    r = float(rsi(c).iloc[-1])
    sma50 = float(c.rolling(50).mean().iloc[-1])
    last = float(c.iloc[-1])
    out = []
    if r < 30 and c.iloc[-1] > c.iloc[-2]:
        out.append("oversold bounce")
    hi20 = float(df["high"].iloc[-21:-1].max())
    rng5 = float((df["high"].iloc[-5:] - df["low"].iloc[-5:]).mean())
    rng20 = float((df["high"].iloc[-20:] - df["low"].iloc[-20:]).mean())
    if last > sma50 and last >= 0.97 * hi20 and rng5 < 0.75 * rng20:
        out.append("tight flag")
    if not out:
        return None
    avg_vol = float(df["volume"].iloc[-20:].mean())
    return {"symbol": symbol, "setup": ", ".join(out), "last": round(last, 2), "rsi14": round(r, 1),
            "vs_sma50_%": round((last / sma50 - 1) * 100, 1), "avg_vol_20d": int(avg_vol)}


def run_scan(symbols, fn, workers: int = 12, **kw):
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(lambda s: _safe(fn, s, **kw), symbols):
            if r:
                rows.append(r)
    return pd.DataFrame(rows)


def _safe(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except Exception:
        return None


def nasdaq_universe(limit: Optional[int] = None) -> list[str]:
    """US-listed symbols from Nasdaq Trader (no ETFs, no test issues, no odd tickers)."""
    syms = []
    for url, col, etf_col in (
        ("https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt", "Symbol", "ETF"),
        ("https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt", "ACT Symbol", "ETF"),
    ):
        r = _get(url)
        r.raise_for_status()
        lines = [l for l in r.text.splitlines() if "|" in l and not l.startswith("File Creation")]
        df = pd.read_csv(io.StringIO("\n".join(lines)), sep="|", dtype=str)
        df = df[(df.get("Test Issue", "N") == "N") & (df[etf_col] == "N")]
        syms += [s for s in df[col].dropna() if re.fullmatch(r"[A-Z]{1,5}", s)]
    syms = sorted(set(syms))
    return syms[:limit] if limit else syms


# --------------------------------------------------------------------------- #
# News Monitor
# --------------------------------------------------------------------------- #
_NEWS_TAGS = {
    "earnings": r"earnings|eps|revenue|guidance|quarter|q[1-4]\b",
    "deal": r"merger|acquir|acquisition|buyout|takeover|deal",
    "analyst": r"upgrade|downgrade|price target|initiat",
    "legal/reg": r"sec |lawsuit|probe|investigation|fda|antitrust|recall",
    "product": r"launch|unveil|approval|partnership|contract",
    "offering": r"offering|dilution|secondary|convertible",
}


def tag_headline(title: str) -> list[str]:
    t = title.lower()
    return [k for k, rx in _NEWS_TAGS.items() if re.search(rx, t)]


def fetch_news(symbols, hours: int = 24) -> pd.DataFrame:
    rows = []
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)
    for s in symbols:
        try:
            r = _get("https://feeds.finance.yahoo.com/rss/2.0/headline", params={"s": s, "region": "US", "lang": "en-US"})
            root = ET.fromstring(r.content)
        except Exception:
            continue
        for it in root.iter("item"):
            title = (it.findtext("title") or "").strip()
            link = it.findtext("link") or ""
            try:
                ts = pd.to_datetime(it.findtext("pubDate"), utc=True)
            except Exception:
                continue
            if ts < cutoff:
                continue
            rows.append({"symbol": s, "time_ET": ts.tz_convert(NY).strftime("%a %H:%M"), "headline": title,
                         "tags": ", ".join(tag_headline(title)), "link": link, "_ts": ts})
    df = pd.DataFrame(rows)
    return df.sort_values("_ts", ascending=False).drop(columns="_ts") if not df.empty else df


def today_move(symbol: str) -> Optional[float]:
    df = fetch_bars(symbol, "1d", "5d")
    if len(df) < 2:
        return None
    return float(df["close"].iloc[-1] / df["close"].iloc[-2] - 1) * 100


# --------------------------------------------------------------------------- #
# Exit Planner
# --------------------------------------------------------------------------- #
def _cents(x: float) -> float:
    from decimal import Decimal, ROUND_HALF_EVEN
    return float(Decimal(str(round(x, 6))).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN))


@dataclass
class Plan:
    symbol: str
    direction: str
    entry: float
    stop: float
    target: float
    risk_per_share: float
    reward_risk: float
    flat_by: str
    stop_rule: str = ""
    target_rule: str = ""


def make_plan(symbol, direction, entry, or_high, or_low, stop_rule="or-mid", target_r=1.0, flat_by="15:55") -> Plan:
    mid = (or_high + or_low) / 2
    if direction == "long":
        stop = {"or-mid": mid, "or-low": or_low}[stop_rule]
        risk = entry - stop
        target = entry + target_r * risk
    else:
        stop = {"or-mid": mid, "or-low": or_high}[stop_rule]  # or-low for shorts = top of range
        risk = stop - entry
        target = entry - target_r * risk
    if risk <= 0:
        raise ValueError("Entry is not beyond the stop: no valid risk.")
    # Round the stop first (half-even on the cent), then derive risk and target from the
    # rounded prices so the numbers on the ticket agree with each other.
    entry, stop = _cents(entry), _cents(stop)
    risk = abs(entry - stop)
    target = _cents(entry + target_r * risk) if direction == "long" else _cents(entry - target_r * risk)
    return Plan(symbol, direction, entry, stop, target,
                round(risk, 2), round(target_r, 2), flat_by, stop_rule, f"{target_r}R")


# --------------------------------------------------------------------------- #
# Time Machine: bootstrap simulation + historical replay
# --------------------------------------------------------------------------- #
def _slot_table(sessions: dict):
    """For each session, bar candles relative to previous close, aligned by slot index.
    Returns arrays hi, lo, cl each shaped (n_sessions, n_slots) of ratios to prior close."""
    his, los, cls = [], [], []
    for d in sorted(sessions):
        g = sessions[d]
        if len(g) < 78:  # skip half days / partial sessions
            continue
        g = g.iloc[:78]
        prev = g["close"].shift(1)
        prev.iloc[0] = g["open"].iloc[0]
        his.append((g["high"] / prev).to_numpy())
        los.append((g["low"] / prev).to_numpy())
        cls.append((g["close"] / prev).to_numpy())
    if not his:
        raise ValueError("Not enough full sessions to simulate.")
    return np.array(his), np.array(los), np.array(cls)


def _slot_of(ts_or_str, hm: Optional[str] = None) -> int:
    """5-minute slot index within the 09:30 session."""
    if hm is None:
        hm = ts_or_str.strftime("%H:%M")
    h, m = map(int, hm.split(":"))
    return ((h * 60 + m) - (9 * 60 + 30)) // BAR_MIN


def simulate(sessions: dict, plan: Plan, entry_hm: str, n_sims: int = 10000, seed: int = 19) -> dict:
    """Replay the rest of the session n_sims times, each bar drawn from the stock's own
    candle at that time of day. Stop is checked before target inside a bar (conservative)."""
    HI, LO, CL = _slot_table(sessions)
    n_days, n_slots = HI.shape
    start = _slot_of(None, entry_hm) + 1
    end = min(_slot_of(None, plan.flat_by) + 1, n_slots)
    rng = np.random.default_rng(seed)
    p = np.full(n_sims, plan.entry)
    state = np.zeros(n_sims, dtype=int)  # 0 open, 1 target, 2 stop
    long = plan.direction == "long"
    for s in range(start, end):
        d = rng.integers(0, n_days, n_sims)
        hi, lo, cl = p * HI[d, s], p * LO[d, s], p * CL[d, s]
        live = state == 0
        hit_stop = live & ((lo <= plan.stop) if long else (hi >= plan.stop))
        hit_tgt = live & ((hi >= plan.target) if long else (lo <= plan.target))
        state[hit_stop] = 2
        state[hit_tgt & ~hit_stop] = 1
        p = np.where(live, cl, p)
    pnl_open = (p - plan.entry) if long else (plan.entry - p)
    pnl = np.where(state == 1, abs(plan.target - plan.entry), np.where(state == 2, -plan.risk_per_share, pnl_open))
    return {
        "n": n_sims,
        "target_first": float((state == 1).mean()),
        "stop_first": float((state == 2).mean()),
        "open_at_close": float((state == 0).mean()),
        "expected_R": float(pnl.mean() / plan.risk_per_share),
    }


def replay_history(sessions: dict, direction: str, stop_rule: str, target_r: float, flat_by: str = "15:55") -> dict:
    """Run the same ORB rule over each past session."""
    n = OR_MINUTES // BAR_MIN
    rows = []
    for d in sorted(sessions):
        g = sessions[d]
        if len(g) < 78:
            continue
        orr = opening_range(g)
        hi, lo = orr
        after = g.iloc[n:]
        cond = (after["close"] > hi) if direction == "long" else (after["close"] < lo)
        if not cond.any():
            continue
        t = cond.idxmax()
        entry = float(after.loc[t, "close"])
        try:
            pl = make_plan("", direction, entry, hi, lo, stop_rule, target_r, flat_by)
        except ValueError:
            continue
        rest = g.loc[g.index > t].between_time("09:30", flat_by)
        outcome, exit_px = "open", float(rest["close"].iloc[-1]) if len(rest) else entry
        for _, b in rest.iterrows():
            if direction == "long":
                if b["low"] <= pl.stop: outcome, exit_px = "stop", pl.stop; break
                if b["high"] >= pl.target: outcome, exit_px = "target", pl.target; break
            else:
                if b["high"] >= pl.stop: outcome, exit_px = "stop", pl.stop; break
                if b["low"] <= pl.target: outcome, exit_px = "target", pl.target; break
        pnl = (exit_px - entry) if direction == "long" else (entry - exit_px)
        rows.append({"date": d, "trigger": t.strftime("%H:%M"), "entry": round(entry, 2), "outcome": outcome,
                     "R": round(pnl / pl.risk_per_share, 2)})
    df = pd.DataFrame(rows)
    if df.empty:
        return {"fired": 0, "table": df}
    vc = df["outcome"].value_counts()
    return {"fired": len(df), "target": int(vc.get("target", 0)), "stop": int(vc.get("stop", 0)),
            "open": int(vc.get("open", 0)), "avg_R": float(df["R"].mean()), "table": df}


def verdict(sim: dict, hist: dict) -> str:
    edge = sim["expected_R"]
    ratio = sim["target_first"] / max(sim["stop_first"], 1e-9)
    if abs(edge) < 0.05 and 0.8 < ratio < 1.25:
        return "Close to a coin flip. No visible edge from the stock's own recent bars."
    if edge >= 0.15 and hist.get("fired", 0) >= 10 and hist.get("avg_R", 0) > 0:
        return "Positive on both the simulation and the real sessions. Still a small sample; size small."
    if edge < -0.05:
        return "Negative expectancy on these bars. The setup is not paying for its risk."
    return "Mixed or thin evidence. Treat as unproven."


# --------------------------------------------------------------------------- #
# Budget Manager
# --------------------------------------------------------------------------- #
def size_position(cfg: Config, plan: Plan) -> dict:
    by_loss = math.floor(cfg.max_loss_per_trade / plan.risk_per_share)
    by_cash = math.floor(cfg.account / plan.entry)
    by_pct = math.floor(cfg.account * cfg.max_position_pct / 100 / plan.entry)
    shares = max(0, min(by_loss, by_cash, by_pct))
    return {
        "symbol": plan.symbol, "side": "buy" if plan.direction == "long" else "sell_short",
        "shares": shares, "entry": plan.entry, "stop": plan.stop, "target": plan.target,
        "position_value": round(shares * plan.entry, 2),
        "loss_at_stop": round(shares * plan.risk_per_share, 2),
        "gain_at_target": round(shares * abs(plan.target - plan.entry), 2),
        "limited_by": min((("max loss", by_loss), ("cash", by_cash), ("position %", by_pct)), key=lambda x: x[1])[0],
    }


# --------------------------------------------------------------------------- #
# Risk Manager
# --------------------------------------------------------------------------- #
def risk_check(cfg: Config, order: dict, plan: Plan, trades_today: int, realized_today: float,
               now_et: Optional[datetime] = None) -> list[dict]:
    now_et = now_et or pd.Timestamp.now(tz=NY).to_pydatetime()
    t = now_et.time()
    rr = abs(plan.target - plan.entry) / plan.risk_per_share
    checks = [
        ("Shares > 0", order["shares"] > 0, f"{order['shares']} shares"),
        ("Max loss per trade", order["loss_at_stop"] <= cfg.max_loss_per_trade + 1e-9,
         f"${order['loss_at_stop']:.2f} vs limit ${cfg.max_loss_per_trade:.2f}"),
        ("Daily loss limit", (-realized_today + order["loss_at_stop"]) <= cfg.daily_loss_limit + 1e-9,
         f"realized ${realized_today:+.2f}, this trade risks ${order['loss_at_stop']:.2f}, limit ${cfg.daily_loss_limit:.2f}"),
        ("Trades per day", trades_today < cfg.max_trades_per_day, f"{trades_today} done, limit {cfg.max_trades_per_day}"),
        ("Reward : risk", rr >= cfg.min_reward_risk - 1e-9, f"{rr:.2f} vs min {cfg.min_reward_risk:.2f}"),
        ("Entry window", _hm(cfg.entry_window_start) <= t <= _hm(cfg.entry_window_end),
         f"{t.strftime('%H:%M')} ET, window {cfg.entry_window_start}-{cfg.entry_window_end}"),
        ("Position size", order["position_value"] <= cfg.account * cfg.max_position_pct / 100 + 1e-9,
         f"${order['position_value']:,.2f} vs cap ${cfg.account * cfg.max_position_pct / 100:,.2f}"),
        ("Cash", order["position_value"] <= cfg.account + 1e-9,
         f"${order['position_value']:,.2f} vs account ${cfg.account:,.2f}"),
    ]
    return [{"check": n, "result": "PASS" if ok else "BLOCK", "detail": d} for n, ok, d in checks]


def risk_verdict(checks) -> str:
    return "PASS" if all(c["result"] == "PASS" for c in checks) else "BLOCK"


# --------------------------------------------------------------------------- #
# Price Alerts
# --------------------------------------------------------------------------- #
def alerts_from_plan(plan: Plan) -> list[dict]:
    stop_op = "<=" if plan.direction == "long" else ">="
    tgt_op = ">=" if plan.direction == "long" else "<="
    return [
        {"symbol": plan.symbol, "kind": "stop", "op": stop_op, "level": plan.stop, "fired": False},
        {"symbol": plan.symbol, "kind": "target", "op": tgt_op, "level": plan.target, "fired": False},
        {"symbol": plan.symbol, "kind": "flat", "op": "time", "level": plan.flat_by, "fired": False},
    ]


def check_alerts(alerts: list[dict], now_et: Optional[datetime] = None):
    """Mutates `fired`; returns list of newly-fired alerts."""
    now_et = now_et or pd.Timestamp.now(tz=NY).to_pydatetime()
    prices = {}
    new = []
    for a in alerts:
        if a["fired"]:
            continue
        if a["op"] == "time":
            hit = now_et.time() >= _hm(a["level"])
        else:
            if a["symbol"] not in prices:
                b = fetch_bars(a["symbol"], "1m", "1d")
                prices[a["symbol"]] = float(b["close"].iloc[-1]) if len(b) else None
            px = prices[a["symbol"]]
            hit = px is not None and ((px <= a["level"]) if a["op"] == "<=" else (px >= a["level"]))
        if hit:
            a["fired"] = True
            new.append(a)
    return new


# --------------------------------------------------------------------------- #
# Earnings Analyst (SEC EDGAR XBRL)
# --------------------------------------------------------------------------- #
def _sec_headers(ua):
    return {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}


def sec_cik(ticker: str, ua: str) -> Optional[str]:
    r = _get("https://www.sec.gov/files/company_tickers.json", headers=_sec_headers(ua))
    r.raise_for_status()
    for v in r.json().values():
        if v["ticker"].upper() == ticker.upper():
            return str(v["cik_str"]).zfill(10)
    return None


def sec_facts(ticker: str, ua: str) -> pd.DataFrame:
    """Quarterly revenue, net income, diluted EPS with YoY change."""
    cik = sec_cik(ticker, ua)
    if not cik:
        raise ValueError(f"{ticker}: not found at SEC")
    r = _get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", headers=_sec_headers(ua))
    r.raise_for_status()
    gaap = r.json()["facts"].get("us-gaap", {})
    spec = {
        "Revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
        "Net income": ["NetIncomeLoss"],
        "Diluted EPS": ["EarningsPerShareDiluted"],
    }
    frames = {}
    for label, tags in spec.items():
        for tag in tags:
            units = gaap.get(tag, {}).get("units", {})
            vals = units.get("USD") or units.get("USD/shares") or []
            q = [v for v in vals if v.get("fp", "").startswith("Q") and v.get("form") in ("10-Q", "10-K")
                 and v.get("start") and 80 <= (pd.Timestamp(v["end"]) - pd.Timestamp(v["start"])).days <= 100]
            if q:
                s = pd.DataFrame(q).drop_duplicates("end", keep="last").set_index("end")["val"]
                frames[label] = s
                break
    df = pd.DataFrame(frames).sort_index().tail(12)
    for c in list(df.columns):
        df[f"{c} YoY %"] = (df[c] / df[c].shift(4) - 1) * 100
    df.index.name = "quarter end"
    return df


# --------------------------------------------------------------------------- #
# Congress tracker (House Clerk disclosures index)
# --------------------------------------------------------------------------- #
def house_ptr_index(year: int, member: str = "") -> pd.DataFrame:
    r = _get(f"https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.zip", timeout=60)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    xml_name = next(n for n in z.namelist() if n.lower().endswith(".xml"))
    root = ET.fromstring(z.read(xml_name))
    rows = []
    for m in root.iter("Member"):
        if (m.findtext("FilingType") or "").strip() != "P":
            continue
        last, first = (m.findtext("Last") or "").strip(), (m.findtext("First") or "").strip()
        name = f"{first} {last}"
        if member and member.lower() not in name.lower():
            continue
        doc = (m.findtext("DocID") or "").strip()
        rows.append({"member": name, "state_district": (m.findtext("StateDst") or "").strip(),
                     "filed": (m.findtext("FilingDate") or "").strip(), "doc_id": doc,
                     "pdf": f"https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{year}/{doc}.pdf"})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["_d"] = pd.to_datetime(df["filed"], errors="coerce")
        df = df.sort_values("_d", ascending=False).drop(columns="_d")
    return df
