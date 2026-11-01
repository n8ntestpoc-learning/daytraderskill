"""Day Trader desk: a Streamlit app built around the 11-skill pipeline from
github.com/Jakeschincariol/day-trader-skill.

scanner -> news -> entry -> exit -> time machine -> budget -> risk -> broker (paper)
alerts / analyst / congress tracker alongside.

Run:  streamlit run app.py
Educational software. Not financial advice. Paper trading only.
"""
import json
from datetime import datetime

import pandas as pd
import streamlit as st

import core
from core import Config

st.set_page_config(page_title="Day Trader Desk", page_icon="📈", layout="wide")

DEFAULT_WATCH = "TSLA, NVDA, AMD, AAPL, MSFT, META, AMZN, GOOGL, PLTR, COIN, SOFI, SMCI, MARA, NFLX, AVGO, SHOP"

# ----------------------------------------------------------------------------- state
S = st.session_state
S.setdefault("cfg", Config())
S.setdefault("watch", DEFAULT_WATCH)
S.setdefault("plan", None)
S.setdefault("entry_hm", None)
S.setdefault("sim", None)
S.setdefault("order", None)
S.setdefault("risk", None)
S.setdefault("alerts", [])
S.setdefault("blotter", [])  # paper tickets
S.setdefault("trades_today", 0)
S.setdefault("realized_today", 0.0)
cfg: Config = S.cfg


def watchlist():
    return [s.strip().upper() for s in S.watch.replace("\n", ",").split(",") if s.strip()]


def now_et():
    return pd.Timestamp.now(tz=core.NY)


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("📈 Day Trader Desk")
    st.caption("Paper trading only. Nothing here can place a real order.")
    S.watch = st.text_area("Watchlist", S.watch, height=110)
    st.subheader("Your limits")
    cfg.account = st.number_input("Account ($)", 500.0, 10_000_000.0, cfg.account, 500.0)
    cfg.max_loss_per_trade = st.number_input("Max loss per trade ($)", 1.0, 100_000.0, cfg.max_loss_per_trade, 5.0)
    cfg.daily_loss_limit = st.number_input("Daily loss limit ($)", 1.0, 100_000.0, cfg.daily_loss_limit, 10.0)
    cfg.max_trades_per_day = st.number_input("Max trades per day", 1, 50, cfg.max_trades_per_day)
    cfg.min_reward_risk = st.number_input("Min reward : risk", 0.1, 10.0, cfg.min_reward_risk, 0.1)
    cfg.max_position_pct = st.number_input("Max position (% of account)", 5.0, 100.0, cfg.max_position_pct, 5.0)
    c1, c2 = st.columns(2)
    cfg.entry_window_start = c1.text_input("Entry from (ET)", cfg.entry_window_start)
    cfg.entry_window_end = c2.text_input("Entry until (ET)", cfg.entry_window_end)
    cfg.flat_by = st.text_input("Flat by (ET)", cfg.flat_by)
    cfg.sec_user_agent = st.text_input("SEC user agent (name email)", cfg.sec_user_agent,
                                       help="SEC EDGAR requires you to identify yourself.")
    st.divider()
    st.metric("Trades today", S.trades_today)
    S.realized_today = st.number_input("Realized P&L today ($)", -100000.0, 100000.0, float(S.realized_today), 10.0)
    st.caption(f"New York time: {now_et().strftime('%a %H:%M:%S')}")
    if st.button("Reset day"):
        for k in ("plan", "entry_hm", "sim", "order", "risk"):
            S[k] = None
        S.alerts, S.blotter, S.trades_today, S.realized_today = [], [], 0, 0.0
        st.rerun()

st.title("Day Trader Desk")
with st.expander("Not financial advice. Read this once.", expanded=False):
    st.markdown(
        "- This reads free, unofficial public data (Yahoo Finance, Nasdaq Trader, SEC EDGAR, House Clerk). It can lag, change or rate-limit you.\n"
        "- Most people who day trade lose money. The Time Machine is a simulation, not a prediction.\n"
        "- A stop is a plan, not a guarantee: fast markets and gaps fill past the price.\n"
        "- The Broker tab only records a **paper ticket** inside this app after Risk passes and you confirm."
    )

tabs = st.tabs(["1 Scanner", "2 News", "3 Entry", "4 Exit", "5 Time Machine", "6 Budget", "7 Risk",
                "8 Broker", "9 Alerts", "10 Analyst", "11 Congress"])
(t_scan, t_news, t_entry, t_exit, t_tm, t_budget, t_risk, t_broker, t_alerts, t_analyst, t_cong) = tabs


# ----------------------------------------------------------------------------- 1 scanner
with t_scan:
    st.subheader("Chart Scanner")
    st.caption("Opening-range breakouts (first 15 min), plus pre-market daily setups: oversold bounce and tight flag.")
    c1, c2, c3 = st.columns([2, 1, 1])
    mode = c1.radio("Scan", ["Opening-range breakout (intraday)", "Daily setups (oversold / flag)"], horizontal=True)
    universe = c2.selectbox("Universe", ["Watchlist", "Nasdaq Trader (first N)"])
    n_univ = c3.number_input("N symbols", 50, 6000, 300, 50, disabled=universe == "Watchlist")
    direction = "long"
    if mode.startswith("Opening"):
        direction = st.radio("Direction", ["long", "short"], horizontal=True)
    if st.button("Run scan", type="primary"):
        try:
            syms = watchlist() if universe == "Watchlist" else core.nasdaq_universe(int(n_univ))
        except Exception as e:
            st.error(f"Could not load the symbol list: {e}")
            syms = []
        if syms:
            with st.spinner(f"Scanning {len(syms)} charts..."):
                if mode.startswith("Opening"):
                    df = core.run_scan(syms, core.scan_symbol_orb, direction=direction)
                else:
                    df = core.run_scan(syms, core.scan_symbol_daily)
            st.session_state["scan_df"] = df
            st.session_state["scan_n"] = len(syms)
    df = S.get("scan_df")
    if df is not None:
        st.write(f"{len(df)} hits of {S.get('scan_n')} charts scanned.")
        if len(df):
            sort_col = "pct_beyond" if "pct_beyond" in df.columns else "rsi14"
            st.dataframe(df.sort_values(sort_col, ascending=False), use_container_width=True, hide_index=True)
        else:
            st.info("Nothing qualifies right now. The opening-range scan only returns charts after 09:45 ET on a trading day.")


# ----------------------------------------------------------------------------- 2 news
with t_news:
    st.subheader("News Monitor")
    c1, c2 = st.columns([3, 1])
    news_syms = c1.text_input("Tickers", ", ".join(watchlist()[:6]))
    hours = c2.number_input("Last N hours", 1, 168, 24)
    if st.button("Fetch headlines"):
        syms = [s.strip().upper() for s in news_syms.split(",") if s.strip()]
        with st.spinner("Reading headlines..."):
            ndf = core.fetch_news(syms, hours)
            moves = {s: core.today_move(s) for s in syms}
        cols = st.columns(min(len(syms), 6) or 1)
        for i, s in enumerate(syms[:6]):
            mv = moves.get(s)
            cols[i].metric(s, "n/a" if mv is None else f"{mv:+.2f}%")
        if ndf.empty:
            st.info("No headlines in that window.")
        else:
            st.dataframe(ndf, use_container_width=True, hide_index=True,
                         column_config={"link": st.column_config.LinkColumn("link", display_text="open")})
            st.caption("Tags are keyword matches, not sentiment. Earnings this week? A day trade is flat before the close, so it is not exposed to the report.")


# ----------------------------------------------------------------------------- 3 entry
with t_entry:
    st.subheader("Entry Timer")
    st.caption("Checks one symbol's opening range: waiting, triggered, or expired (back inside the range).")
    c1, c2, c3 = st.columns(3)
    e_sym = c1.text_input("Symbol", "TSLA").upper()
    e_dir = c2.radio("Direction ", ["long", "short"], horizontal=True)
    if c3.button("Check now", type="primary"):
        bars = core.fetch_bars(e_sym, "5m", "1d")
        ses = core.split_sessions(bars)
        if not len(bars):
            st.error("No data returned (market closed, bad symbol, or Yahoo rate limit).")
        else:
            day = ses[max(ses)] if ses else bars.between_time("09:30", "15:59")
            S.entry_state = {"symbol": e_sym, **core.orb_state(day, e_dir)}
            S.entry_bars = day
    es = S.get("entry_state")
    if es:
        stt = es["state"]
        badge = {"forming": "🟡 forming", "waiting": "🟡 waiting", "triggered": "🟢 triggered", "expired": "🔴 expired"}[stt]
        st.markdown(f"### {es['symbol']}: {badge}")
        if stt == "forming":
            st.info("The 15-minute opening range is still forming.")
        else:
            m = st.columns(4)
            m[0].metric("Range high", f"{es['or_high']:.2f}")
            m[1].metric("Range low", f"{es['or_low']:.2f}")
            m[2].metric("Last", f"{es['last']:.2f}")
            if "trigger_price" in es:
                m[3].metric("Trigger", f"{es['trigger_price']:.2f}", es["trigger_ts"].strftime("%H:%M"))
            st.line_chart(S.entry_bars["close"], height=220)
            if stt == "triggered":
                if st.button("Send to Exit planner"):
                    S.entry_for_plan = es
                    S.entry_hm = es["trigger_ts"].strftime("%H:%M")
                    st.success("Sent. Open the Exit tab.")


# ----------------------------------------------------------------------------- 4 exit
with t_exit:
    st.subheader("Exit Planner")
    st.caption("Set the stop and target before you enter. Flat by the close either way.")
    es = S.get("entry_for_plan")
    if not es:
        st.info("Send a triggered setup from the Entry tab first.")
    else:
        c1, c2, c3 = st.columns(3)
        stop_rule = c1.selectbox("Stop", ["or-mid", "or-low"], help="or-mid = midpoint of the opening range; or-low = far side of the range")
        target_r = c2.number_input("Target (multiples of risk)", 0.25, 10.0, 1.0, 0.25)
        entry_px = c3.number_input("Entry price", value=float(es["trigger_price"]), step=0.01)
        try:
            plan = core.make_plan(es["symbol"], es["direction"], entry_px, es["or_high"], es["or_low"],
                                  stop_rule, target_r, cfg.flat_by)
            S.plan = plan
            m = st.columns(5)
            m[0].metric("Entry", f"{plan.entry:.2f}")
            m[1].metric("Stop", f"{plan.stop:.2f}")
            m[2].metric("Target", f"{plan.target:.2f}")
            m[3].metric("Risk / share", f"{plan.risk_per_share:.2f}")
            m[4].metric("Flat by", plan.flat_by)
        except ValueError as e:
            S.plan = None
            st.error(str(e))


# ----------------------------------------------------------------------------- 5 time machine
with t_tm:
    st.subheader("Time Machine")
    st.caption("Replays the rest of the session thousands of times using the stock's own last ~60 days of 5-minute bars, "
               "plus every past day the same rule fired. A reality check, not a prediction.")
    plan = S.plan
    if not plan:
        st.info("Build a plan in the Exit tab first.")
    else:
        c1, c2, c3 = st.columns(3)
        n_sims = c1.select_slider("Simulations", [1000, 5000, 10000, 20000], 10000)
        seed = c2.number_input("Seed", 0, 10**6, 19)
        entry_hm = c3.text_input("Entry time (HH:MM ET)", S.entry_hm or "10:15")
        if st.button("Run Time Machine", type="primary"):
            with st.spinner("Pulling 60 days of 5-minute bars and simulating..."):
                bars = core.fetch_bars(plan.symbol, "5m", "60d")
                ses = core.split_sessions(bars)
                try:
                    today = now_et().date()
                    hist = {d: g for d, g in ses.items() if d != today}
                    sim = core.simulate(hist, plan, entry_hm, int(n_sims), int(seed))
                    rep = core.replay_history(hist, plan.direction, plan.stop_rule, plan.reward_risk, plan.flat_by)
                    S.sim = (sim, rep, len(hist))
                except Exception as e:
                    st.error(f"Could not run: {e}")
        if S.sim:
            sim, rep, ndays = S.sim
            st.markdown(f"**{plan.symbol}** · {sim['n']:,} simulated sessions drawn from {ndays} real sessions")
            m = st.columns(4)
            m[0].metric("Target first", f"{sim['target_first']*100:.1f}%")
            m[1].metric("Stop first", f"{sim['stop_first']*100:.1f}%")
            m[2].metric("Still open at close", f"{sim['open_at_close']*100:.1f}%")
            m[3].metric("Expected R", f"{sim['expected_R']:+.2f}")
            if rep["fired"]:
                st.write(f"Same rule on the real sessions: fired **{rep['fired']}** times, "
                         f"**{rep['target']}** hit target, **{rep['stop']}** hit stop, **{rep['open']}** neither. "
                         f"Average {rep['avg_R']:+.2f}R.")
                with st.expander("Every past trigger"):
                    st.dataframe(rep["table"], use_container_width=True, hide_index=True)
            else:
                st.write("The rule never fired in the history window.")
            st.info(core.verdict(sim, rep))
            st.caption("Built from the last 60 days only. It cannot know about an earnings report, a halt or a headline.")


# ----------------------------------------------------------------------------- 6 budget
with t_budget:
    st.subheader("Budget Manager")
    if not S.plan:
        st.info("Build a plan in the Exit tab first.")
    else:
        order = core.size_position(cfg, S.plan)
        S.order = order
        m = st.columns(4)
        m[0].metric("Shares", order["shares"], help=f"Limited by {order['limited_by']}")
        m[1].metric("Position value", f"${order['position_value']:,.2f}")
        m[2].metric("Loss at stop", f"-${order['loss_at_stop']:,.2f}")
        m[3].metric("Gain at target", f"+${order['gain_at_target']:,.2f}")
        st.caption(f"Sized by: {order['limited_by']}. One bracket order: {order['side']} {order['shares']} {order['symbol']}, "
                   f"stop {order['stop']}, take-profit {order['target']}.")
        st.json(order, expanded=False)


# ----------------------------------------------------------------------------- 7 risk
with t_risk:
    st.subheader("Risk Manager")
    if not (S.plan and S.order):
        st.info("Needs a plan and a sized order.")
    else:
        checks = core.risk_check(cfg, S.order, S.plan, S.trades_today, S.realized_today, now_et().to_pydatetime())
        v = core.risk_verdict(checks)
        S.risk = v
        (st.success if v == "PASS" else st.error)(f"{v}")
        st.dataframe(pd.DataFrame(checks), use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------------- 8 broker
with t_broker:
    st.subheader("Order Executor (paper ticket)")
    if not (S.plan and S.order):
        st.info("Needs a plan and a sized order.")
    elif S.risk != "PASS":
        st.error("Risk has not passed this exact order. No ticket can be created.")
    else:
        o = S.order
        st.markdown(
            f"**Ticket:** {o['side'].replace('_', ' ').upper()} **{o['shares']} {o['symbol']}** at about {o['entry']:.2f}  \n"
            f"Stop-loss {o['stop']:.2f} · Take-profit {o['target']:.2f} · Flat by {S.plan.flat_by} ET · Paper account")
        confirm = st.text_input(f"Type yes to record this exact paper order for {o['shares']} {o['symbol']}")
        if st.button("Record paper ticket", disabled=confirm.strip().lower() != "yes"):
            S.blotter.append({"time": now_et().strftime("%H:%M:%S"), **o, "paper": True})
            S.trades_today += 1
            S.alerts += core.alerts_from_plan(S.plan)
            st.success("Paper ticket recorded and alerts armed. To trade for real, wire your own broker's API here yourself.")
    if S.blotter:
        st.markdown("**Paper blotter**")
        st.dataframe(pd.DataFrame(S.blotter), use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------------- 9 alerts
with t_alerts:
    st.subheader("Price Alerts")
    c1, c2 = st.columns(2)
    if c1.button("Add from plan", disabled=not S.plan):
        S.alerts += core.alerts_from_plan(S.plan)
    if c2.button("Check now", type="primary", disabled=not S.alerts):
        fired = core.check_alerts(S.alerts, now_et().to_pydatetime())
        for a in fired:
            st.toast(f"{a['symbol']} {a['kind']} hit ({a['op']} {a['level']})", icon="🔔")
        if not fired:
            st.info("Nothing new has hit.")
    with st.expander("Custom alert"):
        cc = st.columns(4)
        a_sym = cc[0].text_input("Symbol ", "TSLA").upper()
        a_op = cc[1].selectbox("When price", [">=", "<="])
        a_lvl = cc[2].number_input("Level", 0.0, 1e6, 0.0, 0.01)
        if cc[3].button("Add alert") and a_lvl > 0:
            S.alerts.append({"symbol": a_sym, "kind": "price", "op": a_op, "level": a_lvl, "fired": False})
    if S.alerts:
        st.dataframe(pd.DataFrame(S.alerts), use_container_width=True, hide_index=True)
    st.caption("Alerts are checked when you press the button or the app reruns. For unattended ringing, run this on a schedule.")


# ----------------------------------------------------------------------------- 10 analyst
with t_analyst:
    st.subheader("Earnings Analyst")
    st.caption("Quarterly revenue, net income and diluted EPS from SEC EDGAR XBRL facts, with year-over-year change.")
    a_t = st.text_input("Ticker", "TSLA").upper()
    if st.button("Read filings"):
        if "you@example.com" in cfg.sec_user_agent:
            st.warning("Put your name and email in the sidebar. SEC EDGAR asks callers to identify themselves.")
        else:
            try:
                with st.spinner("Reading SEC EDGAR..."):
                    f = core.sec_facts(a_t, cfg.sec_user_agent)
                if f.empty:
                    st.info("No quarterly facts found.")
                else:
                    last = f.iloc[-1]
                    m = st.columns(3)
                    for i, k in enumerate(["Revenue", "Net income", "Diluted EPS"]):
                        if k in f.columns:
                            yoy = last.get(f"{k} YoY %")
                            m[i].metric(k, f"{last[k]:,.2f}" if k == "Diluted EPS" else f"${last[k]/1e6:,.0f}M",
                                        None if pd.isna(yoy) else f"{yoy:+.1f}% YoY")
                    st.dataframe(f.round(2), use_container_width=True)
                    if "Revenue" in f.columns:
                        st.bar_chart(f["Revenue"])
            except Exception as e:
                st.error(f"SEC request failed: {e}")


# ----------------------------------------------------------------------------- 11 congress
with t_cong:
    st.subheader("Congress Tracker")
    st.caption("Periodic Transaction Reports filed with the House Clerk. Disclosures arrive up to 45 days after the trade.")
    c1, c2 = st.columns(2)
    yr = c1.number_input("Year", 2015, now_et().year, now_et().year)
    who = c2.text_input("Member name contains", "pelosi")
    if st.button("Load index"):
        try:
            with st.spinner("Downloading the House Clerk index..."):
                p = core.house_ptr_index(int(yr), who)
            st.write(f"{len(p)} periodic transaction reports.")
            if len(p):
                st.dataframe(p, use_container_width=True, hide_index=True,
                             column_config={"pdf": st.column_config.LinkColumn("report", display_text="open PDF")})
        except Exception as e:
            st.error(f"Could not load the index: {e}")
