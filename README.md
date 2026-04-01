# Kalshi Information-Alpha Bot

An async Python bot that implements an **information-alpha** trading strategy on [Kalshi](https://kalshi.com/) prediction markets. It continuously ingests real-time data from a wide range of public sources, matches each signal to open Kalshi markets, and automatically places (or simulates) trades when a statistically significant edge is found.

---

## How It Works

```
Data Sources                        Matching Engine                 Output
──────────────────────────────      ───────────────────────────     ──────────────────────
Federal Register (6 agencies) ──┐
RSS feeds (AP, Reuters, BBC…) ──┤──► Keyword Matcher     ─────────► TEXT opportunities
SEC EDGAR 8-K filings         ──┘     (topic × market title)

NOAA/NWS + OWM city weather   ──┐
Binance crypto prices          ──┤──► Numeric Matcher     ─────────► DATA opportunities
Frankfurter forex rates        ──┤     (live value vs. strike)
BLS economic releases          ──┤
FRED interest rates            ──┤
EIA energy prices              ──┘

Polymarket (real-money)        ──┐
PredictIt (real-money)         ──┤──► Divergence Matcher  ─────────► EXT opportunities
Metaculus (reputation)         ──┤     (external p vs. Kalshi mid)
Manifold (play-money)          ──┘
```

Every 60 seconds (configurable), the bot runs a full **fetch → match → score → execute** cycle:

1. All sources are fetched **concurrently** via `aiohttp`.
2. Text sources are keyword-matched against market titles (stopword-stripped, stemmed Jaccard).
3. Numeric sources compare live values to market strikes using a calibrated probability model.
4. External forecast platforms are matched to Kalshi via stemmed Jaccard similarity; material divergences become signals.
5. Opportunities are **scored**, **filtered** through quality gates, and executed as dry-run or live trades.

---

## Data Sources

### Text Sources (keyword matching)
| Source | Coverage |
|---|---|
| AP News Top / Politics | US politics, breaking news |
| Reuters | Global economics, markets |
| BBC News | International breaking news |
| NPR | US policy, politics |
| Politico | Congress, administration |
| The Hill | US political news |
| ESPN NBA | Player prop markets |
| ESPN Top Stories | General sports |
| Billboard | Song chart markets |
| Federal Register — EPA | Environmental regulation |
| Federal Register — FDA | Drug approvals, food safety |
| Federal Register — FTC | Antitrust, consumer protection |
| Federal Register — CFTC | Derivatives, crypto regulation |
| Federal Register — Treasury | Fiscal policy, sanctions |
| Federal Register — HHS | Healthcare, public health |
| SEC EDGAR 8-K filings | Corporate events |
| NWS Alerts | Severe weather alerts |

### Numeric Sources (live value vs. strike price)
| Source | Data | Kalshi Markets |
|---|---|---|
| NOAA/NWS | Daily high temp — 9 cities (LAX, DEN, CHI, NY, MIA, AUS, DAL, BOS, HOU) | `KXHIGH*` |
| OpenWeatherMap | Independent high temp cross-validation (same 9 cities) | `KXHIGH*` |
| Binance | BTC, ETH, SOL, XRP, DOGE, ADA, AVAX, LINK (USD) | `KXBTCD`, `KXBTC15M`, `KXETH15M`, `KXSOL15M`, `KXXRP15M`, `KXDOGE*`, `KXADA*`, `KXAVAX*`, `KXLINK*` |
| Frankfurter (ECB) | EUR/USD, USD/JPY | `KXEURUSD`, `KXUSDJPY` |
| BLS | CPI-U, Nonfarm Payrolls, Unemployment Rate | `KXCPI`, `KXNFP`, `KXUNRATE` |
| FRED (St. Louis Fed) | Fed funds rate, 2Y/10Y Treasury yields | `KXFED`, `KXFFR`, `KXDGS2`, `KXDGS10` |
| EIA | WTI crude oil, natural gas | `KXWTI`, `KXOIL`, `KXNATGAS`, `KXNG` |
| CME FedWatch | Probability-weighted next Fed meeting outcome | `KXFED` |

### External Forecast Sources (divergence matching)
| Source | Signal type | Minimum divergence |
|---|---|---|
| Polymarket | Real-money global prediction market | 20 pp (configurable) |
| PredictIt | Real-money US political prediction market | 15 pp (configurable) |
| Metaculus | Reputation-tracked crowd forecasting | 20 pp (configurable) |
| Manifold | Play-money prediction market | 25 pp (configurable) |

---

## Signal Quality & Filters

### Temperature markets (NOAA/OWM consensus)
- Both NOAA and OWM must agree on temperature direction before a trade is placed.
- When today's observed station readings exceed the forecast, source switches to `noaa_observed` with a much tighter uncertainty model (σ ≈ 0.5°F vs 4°F forecast), enabling high-confidence late-day trades.
- Minimum temperature edge configurable via `NUMERIC_MIN_TEMP_EDGE` (default 4°F for forecast, bypassed for observed).
- Forecast-only trades are blocked within `SAME_DAY_CUTOFF_HOURS` of market close (default 2h).

### Divergence matching (Polymarket / PredictIt / Metaculus / Manifold)
- Market titles and external questions are compared using **stemmed Jaccard similarity** (suffix-stripped tokens — "elections" → "elect", "confirmed" → "confirm").
- Entertainment/sports/esports markets (`KXMVE*`, `KXNBA*`, `KXNHL*`, etc.) are excluded from external matching to prevent flooding.
- Markets with no `last_price` use bid/ask midpoint for divergence calculation.
- Manifold signals require either Polymarket/PredictIt corroboration or divergence below `MANI_MAX_SOLO_DIVERGENCE` (default 50%).

### Trade execution quality gates
Each opportunity passes through, in order:
1. `score ≥ TRADE_MIN_SCORE` (default 0.5)
2. Live orderbook present
3. Minimum temperature edge (temperature markets only)
4. Market-vs-model disagreement ≥ `NUMERIC_MIN_DISAGREEMENT` (default 10 pp)
5. Same-day cutoff (temperature markets)
6. Kelly criterion ≥ 1 contract
7. Per-ticker cooldown (`TRADE_TICKER_COOLDOWN_MINUTES`, default 30 min)
8. Circuit breaker not active

### Circuit breakers
- **Consecutive-loss breaker**: trips after `CIRCUIT_BREAKER_CONSECUTIVE_LOSSES` (default 3) consecutive settled losses in a market category, pausing that category for `CIRCUIT_BREAKER_PAUSE_HOURS` (default 24h).
- **Open-trade cap**: trips if `CIRCUIT_BREAKER_MAX_OPEN` (default 5) unsettled trades exist for a category, preventing runaway exposure.

---

## Trade Execution & Dry Run

By default the bot runs in **dry-run mode** (`TRADE_DRY_RUN=true`). Every intended trade is persisted to `opportunity_log.db` as if it had been placed — same sizing, same price — but no order is sent to Kalshi. Set `TRADE_DRY_RUN=false` to enable live order placement.

### Kelly sizing
```
raw_kelly  =  (P(win) − cost/100) / (1 − cost/100)
contracts  =  floor(KELLY_FRACTION × raw_kelly × MAX_POSITION_CENTS / cost)
contracts  =  min(contracts, TRADE_MAX_CONTRACTS)
```
`KELLY_FRACTION` defaults to 0.25 (quarter-Kelly). `MAX_POSITION_CENTS` defaults to $5.00 per trade.

### Dry-run ledger
A live overview file (`dry_run_overview.md`) is updated every cycle showing open positions, unrealized P&L, and cumulative performance — useful for evaluating signal quality before going live.

---

## Output Format

### Text Opportunity
```
────────────────────────────────────────────────────────────────
  [TEXT #1  score=0.72]  tariff  |  KXTRUMP-TARIFF-YES
  Market:   Will Trump impose new tariffs in 2026?
  Price:    55¢  bid=53  ask=57  vol=1,200  |  closes in 14.2 days
  Article:  White House signals new steel tariff package
  URL:      https://...
```

### Data Opportunity
```
────────────────────────────────────────────────────────────────
  [DATA #2  score=0.88]  temp_high_lax  |  KXHIGHLAX-26MAR09-T74
  Market:   Will the high temp in LA be >74° on Mar 9, 2026?
  Live:     76.3°F  (as of 2026-03-09T18:00Z, source: noaa_observed)
  Strike:   OVER 74.0  →  implied YES  (edge 2.3)
  Price:    68¢  bid=66  ask=70  vol=340
```

### External Forecast Opportunity
```
────────────────────────────────────────────────────────────────
  [EXT #3  score=0.81  src=Polymarket]  divergence=28%  |  KXTRUMP-EXEC-ORDER
  Market:   Will Trump sign the executive order?
  Kalshi:   55¢  →  Polymarket: 83%  →  BUY YES
  Match:    0.62  |  liq=$42,000
  External: Will Trump sign the executive order on immigration?
```

---

## Project Structure

```
kalshi_bot/
├── auth.py                Kalshi V2 RSA-PSS authentication
├── markets.py             Async Kalshi API — two-pronged fetch strategy:
│                            series_ticker targeted fetch (numeric markets)
│                            + throttled general pagination (political markets)
├── market_parser.py       Ticker/title → ParsedMarket (direction + strike)
├── matcher.py             Keyword matching → Opportunity dataclass
├── numeric_matcher.py     Numeric matching → NumericOpportunity dataclass
├── polymarket_matcher.py  External forecast divergence matching (Polymarket,
│                            PredictIt, Metaculus, Manifold) with stemmed Jaccard
├── scoring.py             Composite score for all opportunity types
├── trade_executor.py      Kelly sizing, quality gates, dry-run/live execution,
│                            circuit breakers, filter statistics
├── dry_run_ledger.py      Dry-run position tracking + P&L overview file
├── opportunity_log.py     SQLite log of surfaced opportunities (opportunity_log.db)
├── win_rate_tracker.py    Per-category win rate analysis
├── portfolio.py           Live position fetcher and summariser
├── state.py               SQLite deduplication for text sources (state.db)
├── data.py                Shared DataPoint dataclass
└── news/
    ├── federal_register.py  Federal Register fetcher (6 agencies)
    ├── rss.py               Generic RSS/Atom feed fetcher (10+ feeds)
    ├── noaa.py              NOAA/NWS forecast + observed-max fetcher (9 cities)
    ├── owm.py               OpenWeatherMap cross-validation fetcher
    ├── binance.py           Binance spot price fetcher (8 crypto assets)
    ├── frankfurter.py       ECB/Frankfurter forex rate fetcher
    ├── bls.py               BLS economic release fetcher
    ├── fred.py              FRED interest rate fetcher
    ├── eia.py               EIA energy price fetcher
    ├── cme_fedwatch.py      CME FedWatch next-meeting probability fetcher
    ├── polymarket.py        Polymarket binary market fetcher
    ├── predictit.py         PredictIt binary market fetcher
    ├── metaculus.py         Metaculus community forecast fetcher
    ├── manifold.py          Manifold market fetcher
    ├── edgar.py             SEC EDGAR 8-K filing fetcher
    └── nws_alerts.py        NWS severe weather alert fetcher

run.py                   Entry point
state.db                 Text deduplication database (auto-created)
opportunity_log.db       Trade and opportunity history (auto-created)
dry_run_overview.md      Live dry-run P&L overview (auto-updated each cycle)
market_discovery.py      Legacy reference file — do not modify
```

---

## Setup

### Prerequisites
- Python 3.14
- A Kalshi account with API credentials (Key ID + RSA private key)
- Optional: `OWM_API_KEY` for OpenWeatherMap cross-validation (free at openweathermap.org)
- Optional: `BLS_API_KEY` for higher BLS API rate limits

### Install
```bash
python3.14 -m venv venv
source venv/bin/activate
pip install aiohttp cryptography python-dotenv
```

### Configure
Create a `.env` file in the project root:
```env
# Kalshi credentials (required)
KALSHI_KEY_ID=your-key-id-here
KALSHI_PRIVATE_KEY_STR="-----BEGIN RSA PRIVATE KEY-----\n...\n-----END RSA PRIVATE KEY-----"
KALSHI_ENVIRONMENT=demo           # or "production"

# Polling
POLL_INTERVAL=60                  # seconds between poll cycles

# Optional API keys
OWM_API_KEY=                      # OpenWeatherMap (free tier sufficient)
BLS_API_KEY=                      # BLS.gov (500 req/day vs. 25 without)

# Trade execution
TRADE_DRY_RUN=true                # false to place live orders
KELLY_FRACTION=0.25               # quarter-Kelly (conservative default)
MAX_POSITION_CENTS=500            # max dollars per trade ($5.00)
TRADE_MAX_CONTRACTS=50            # hard per-trade cap
TRADE_MIN_SCORE=0.5               # minimum composite score to trade
TRADE_TICKER_COOLDOWN_MINUTES=30  # min minutes between trades on same ticker
```

### Run
```bash
venv/bin/python run.py
```

Use `caffeinate` to prevent the machine from sleeping during an overnight run:
```bash
caffeinate -i venv/bin/python run.py
```

---

## Configuration Reference

### Core
| Env Var | Default | Description |
|---|---|---|
| `KALSHI_KEY_ID` | — | Kalshi API key ID |
| `KALSHI_PRIVATE_KEY_STR` | — | PEM-encoded RSA private key |
| `KALSHI_ENVIRONMENT` | `demo` | `demo` or `production` |
| `POLL_INTERVAL` | `60` | Seconds between poll cycles |

### Market fetching
| Env Var | Default | Description |
|---|---|---|
| `MARKET_REFRESH_INTERVAL` | `300` | Seconds between full market cache refreshes |
| `MARKET_MAX_DAYS_OUT` | `30` | Drop markets closing more than N days out (0 = off) |
| `MARKET_MIN_MINUTES_TO_CLOSE` | `30` | Drop markets closing in less than N minutes (0 = off) |
| `LIQUIDITY_MAX_SPREAD` | `10` | Drop markets with bid-ask spread wider than N cents (0 = off) |
| `LIQUIDITY_MIN_VOLUME` | `0` | Drop markets below this 24h volume floor (0 = off) |

### Trade execution
| Env Var | Default | Description |
|---|---|---|
| `TRADE_DRY_RUN` | `true` | Set `false` to place live orders |
| `KELLY_FRACTION` | `0.25` | Fractional Kelly multiplier |
| `MAX_POSITION_CENTS` | `500` | Max cost per trade in cents ($5.00) |
| `TRADE_MAX_CONTRACTS` | `50` | Hard per-trade contract cap |
| `TRADE_MIN_SCORE` | `0.5` | Minimum composite score to attempt a trade |
| `TRADE_TICKER_COOLDOWN_MINUTES` | `30` | Minimum gap between trades on same ticker |

### Signal quality
| Env Var | Default | Description |
|---|---|---|
| `NUMERIC_MIN_DISAGREEMENT` | `0.10` | Min model-vs-market probability gap to trade |
| `NUMERIC_MIN_TEMP_EDGE` | `4.0` | Min °F edge to trade forecast-only temperature signals |
| `SAME_DAY_CUTOFF_HOURS` | `2.0` | Block temperature trades within N hours of market close |

### Circuit breaker
| Env Var | Default | Description |
|---|---|---|
| `CIRCUIT_BREAKER_CONSECUTIVE_LOSSES` | `3` | Consecutive losses that trip a category pause |
| `CIRCUIT_BREAKER_PAUSE_HOURS` | `24` | Hours to pause a category after the breaker trips |
| `CIRCUIT_BREAKER_MAX_OPEN` | `5` | Max open trades per category before tripping |

### External forecast thresholds
| Env Var | Default | Description |
|---|---|---|
| `POLY_MIN_DIVERGENCE` | `0.20` | Min Polymarket vs. Kalshi probability gap |
| `POLY_MIN_LIQUIDITY` | `5000` | Min Polymarket liquidity (USD) |
| `POLY_MIN_MATCH_SCORE` | `0.20` | Min Jaccard similarity to match questions |
| `PDIT_MIN_DIVERGENCE` | `0.15` | Min PredictIt vs. Kalshi probability gap |
| `PDIT_MIN_VOLUME` | `500` | Min PredictIt day volume (USD) |
| `META_MIN_DIVERGENCE` | `0.20` | Min Metaculus vs. Kalshi probability gap |
| `META_MIN_FORECASTERS` | `20` | Min Metaculus community participants |
| `MANI_MIN_DIVERGENCE` | `0.25` | Min Manifold vs. Kalshi probability gap |
| `MANI_MIN_LIQUIDITY` | `500` | Min Manifold liquidity (mana) |

---

## Architecture Notes

- **Market fetching uses a two-pronged strategy** to bypass Kalshi's pagination ordering, which places 10,000+ sports markets before any weather/crypto/political markets. A targeted `series_ticker=` fetch directly retrieves all known numeric series; a throttled general pagination (0.25s/page) collects political/text-matchable markets without hitting rate limits.
- **All HTTP is async** via `aiohttp`; a shared `ClientSession` with a 30-connection pool is reused across all cycles.
- **Deduplication** for text sources is SQLite-backed (`state.db`). Trade and opportunity history are persisted to `opportunity_log.db`, which also stores circuit breaker state and price snapshots for P&L tracking.
- **Temperature matching** uses two independent weather models (NOAA/NWS + OWM). Both must agree on direction before a trade is placed. Late in the trading day, observed station max temperatures provide a hard lower bound, tightening the uncertainty model from σ ≈ 4°F to σ ≈ 0.5°F.
- **External forecast matching** uses suffix-stripped stemmed Jaccard similarity so "elections"/"elected"/"electing" all map to the same stem "elect", and "confirmed"/"confirmation" both map to "confirm" — dramatically reducing false-negative matches.
- **Authentication** uses RSA-PSS signatures (Kalshi V2 API): `timestamp + method + path` signed with the private key.
- **Adding a new news source**: create a module in `kalshi_bot/news/` returning `list[DataPoint]` (numeric) or `list[dict]` (text), then add it to the task list in `main.py`'s `_poll` coroutine.
- **Adding a new numeric market series**: add the ticker prefix → metric mapping in `market_parser.py → TICKER_TO_METRIC` and `_NUMERIC_PATTERN_PREFIXES`, add the series prefix to `markets.py → NUMERIC_SERIES`, and add the edge scale in `scoring.py → METRIC_EDGE_SCALES`.
- **Adding a new city to weather tracking**: add a row to `noaa.py → CITIES` and the ticker prefix mapping in `market_parser.py → TICKER_TO_METRIC`. OWM picks up new cities automatically (imports `CITIES` from `noaa.py`).

---

## Limitations & Known Issues

- Kalshi's `status=open` pagination returns sports/entertainment markets first; numeric and political markets only appear after 10,000+ entries. The series_ticker targeted fetch works around this, but political market coverage depends on throttled general pagination staying within rate limits.
- AP News and Reuters RSS feeds may be unreachable depending on network/DNS.
- Politico's feed returns HTTP 403 in some environments.
- CoinGecko (legacy) free tier: ~30 requests/minute. The bot uses Binance by default (no key required, no rate limit issues).
- BLS free tier: 25 queries/day. Use `BLS_API_KEY` for production.
- Jaccard matching is keyword-overlap-based and can produce false positives. The stemmer reduces but does not eliminate mismatches between differently phrased questions.
- `market_discovery.py` in the project root is a legacy reference file — do not modify or import it.
