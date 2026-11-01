# Day Trader Desk (Streamlit)

A Streamlit app built around the 11-skill pipeline in
[Jakeschincariol/day-trader-skill](https://github.com/Jakeschincariol/day-trader-skill).
One tab per skill, each reading what the last one produced:

scanner → news → entry → exit → time machine → budget → risk → broker (paper)
alerts · analyst · congress tracker alongside.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Put your name and email in the sidebar (SEC EDGAR asks callers to identify themselves).

## Data (free, unofficial, no API key)

Yahoo Finance chart + RSS, Nasdaq Trader symbol lists, SEC EDGAR XBRL, House Clerk disclosures.
These can lag, change or rate-limit you. The opening-range scan only returns results after 09:45 ET on a trading day.

## Safety

- Paper only. The Broker tab records a ticket inside the app, and only after Risk returns PASS and you type "yes".
- The Time Machine is a simulation from the stock's last ~60 days of 5-minute bars, not a prediction.
- Not financial advice. Most people who day trade lose money.

## Tests

```bash
pytest -q test_core.py
```

Offline, on synthetic bars. They reproduce the repo's TSLA example (stop 358.32, target 370.04, 8 shares, $46.88 at risk).
