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

NOAA/NWS forecast + METAR     ──┐
NWS HRRR hourly forecasts     ──┤
NWS climatological (CLI)      ──┤──► Numeric Matcher     ─────────► DATA opportunities
Open-Meteo model forecasts    ──┤     (live value vs. strike)
OpenWeatherMap                ──┤
Binance crypto prices         ──┤
Frankfurter forex rates       ──┤
BLS / FRED / EIA economics    ──┘

Polymarket (real-money)       ──┐
Metaculus (reputation)        ──┤──► Divergence Matcher  ─────────► EXT opportunities
Manifold (play-money)         ──┘     (external p vs. Kalshi mid)

METAR observed daily max/min  ──────► Band-Pass Arb      ─────────► BAND_ARB signals
(airport ASOS, 5-8 min ahead)          (definitively-NO bands)

Box Office Mojo estimates     ──────► Box Office Matcher ─────────► DATA opportunities
```

Every 60 seconds (configurable), the bot runs a full **fetch → match → score → execute** cycle:

1. All sources are fetched **concurrently** via `aiohttp`.
2. Text sources are keyword-matched against market titles (stopword-stripped, stemmed Jaccard).
3. Numeric sources compare live values to market strikes using a calibrated probability model.
4. External forecast platforms are matched to Kalshi via stemmed Jaccard similarity; material divergences become signals.
5. METAR observed data triggers **band-pass arbitrage** on temperature partition markets.
6. Opportunities are **scored**, **filtered** through quality gates, and executed as dry-run or live trades.
7. A lightweight **fast loop** (default every 10s) runs band-arb checks between full cycles for near-threshold cities.

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
| NOAA/NWS day-1 forecast | Daily high/low temp — 20 cities | `KXHIGH*`, `KXLOWT*` |
| NOAA/NWS day-2+ forecast | Extended forecast, higher edge threshold | `KXHIGH*`, `KXLOWT*` |
| NOAA observed (ASOS) | Observed daily max/min — running intraday ground truth | `KXHIGH*`, `KXLOWT*` |
| METAR (airport ASOS) | Real-time airport observations, 5-8 min ahead of NOAA | `KXHIGH*` |
| NWS HRRR hourly | High-resolution rapid-refresh model forecasts | `KXHIGH*`, `KXLOWT*` |
| NWS climatological (CLI) | Official daily high/low from NWS CLI products | `KXHIGH*`, `KXLOWT*` |
| Open-Meteo | Independent model forecasts (WMO ensemble) | `KXHIGH*`, `KXLOWT*` |
| OpenWeatherMap | Independent high/low temp cross-validation | `KXHIGH*`, `KXLOWT*` |
| Box Office Mojo | Weekend domestic gross estimates | `KXBO*` |
| Binance | BTC, ETH, SOL, XRP, DOGE, ADA, AVAX, LINK (USD) | `KXBTCD`, `KXBTC15M`, `KXETH15M`, `KXSOL15M`, `KXXRP15M`, `KXDOGE*`, `KXADA*`, `KXAVAX*`, `KXLINK*` |
| Frankfurter (ECB) | EUR/USD, USD/JPY, GBP/USD | `KXEURUSD`, `KXUSDJPY`, `KXGBPUSD` |
| BLS | CPI-U, Nonfarm Payrolls, Unemployment Rate, PPI | `KXCPI`, `KXNFP`, `KXUNRATE`, `KXPPI` |
| FRED (St. Louis Fed) | Fed funds rate, 2Y/10Y Treasury yields, PCE, GDP | `KXFED`, `KXFFR`, `KXDGS2`, `KXDGS10`, `KXPCE`, `KXGDP` |
| EIA | WTI crude oil, natural gas | `KXWTI`, `KXOIL`, `KXNATGAS`, `KXNG` |
| ISM | Manufacturing and Services PMI | `KXISM`, `KXISMMFG`, `KXISMSVC` |
| CME FedWatch | Probability-weighted next Fed meeting outcome | `KXFED` |

#### Temperature cities covered (20 cities, high and low)
| City | High metric | Low metric | Station |
|---|---|---|---|
| Los Angeles | `temp_high_lax` | `temp_low_lax` | KLAX |
| New York | `temp_high_ny` | `temp_low_ny` | KNYC |
| Chicago | `temp_high_chi` | `temp_low_chi` | KMDW |
| Denver | `temp_high_den` | `temp_low_den` | KDEN |
| Miami | `temp_high_mia` | `temp_low_mia` | KMIA |
| Austin | `temp_high_aus` | `temp_low_aus` | KAUS |
| Boston | `temp_high_bos` | `temp_low_bos` | KBOS |
| Houston | `temp_high_hou` | `temp_low_hou` | KHOU |
| Dallas (Love Field) | `temp_high_dal` | — | KDAL |
| Dallas/Fort Worth | `temp_high_dfw` | `temp_low_dfw` | KDFW |
| San Francisco | `temp_high_sfo` | `temp_low_sfo` | KSFO |
| Seattle | `temp_high_sea` | `temp_low_sea` | KSEA |
| Phoenix | `temp_high_phx` | `temp_low_phx` | KPHX |
| Philadelphia | `temp_high_phl` | `temp_low_phl` | KPHL |
| Atlanta | `temp_high_atl` | `temp_low_atl` | KATL |
| Minneapolis | `temp_high_msp` | `temp_low_msp` | KMSP |
| Washington DC | `temp_high_dca` | `temp_low_dca` | KDCA |
| Las Vegas | `temp_high_las` | `temp_low_las` | KLAS |
| Oklahoma City | `temp_high_okc` | `temp_low_okc` | KOKC |
| San Antonio | `temp_high_sat` | `temp_low_sat` | KSAT |
| New Orleans | `temp_high_msy` | `temp_low_msy` | KMSY |

### External Forecast Sources (divergence matching)
| Source | Signal type | Minimum divergence |
|---|---|---|
| Polymarket | Real-money global prediction market | 20 pp (configurable) |
| Metaculus | Reputation-tracked crowd forecasting | 20 pp (configurable) |
| Manifold | Play-money prediction market | 25 pp (configurable) |

---

## Signal Quality & Filters

### Temperature markets (multi-source consensus)
- Multiple independent sources (NOAA forecast, Open-Meteo, NWS HRRR, OWM) must agree on direction before a trade is placed.
- When observed station readings exceed the forecast, source switches to `noaa_observed` or `metar` with a much tighter uncertainty model (σ ≈ 0.5°F vs 4°F for forecasts), enabling high-confidence late-day trades.
- Per-source, per-direction minimum edge thresholds (see configuration reference). `open_meteo` under-signals require 12°F raw edge to net ~7°F true edge after cold bias correction; `weatherapi` under-signals are disabled entirely (−8°F cold bias).
- Forecast-only trades are blocked within `SAME_DAY_CUTOFF_HOURS` of market close (default 2h).
- **Daily high gate**: `noaa_observed` YES signals on high-temp markets are blocked before 13:00 local (afternoon high not yet established).
- **Daily low gate (morning gate)**: `noaa_observed` YES signals on low-temp markets are blocked before 05:00 local. The running daily minimum resets at local midnight and equals the current temperature — not the overnight trough, which typically occurs at 4–6 AM.

### Band-pass / strike arbitrage
The bot exploits a **5–8 minute information advantage** METAR airport observations have over NOAA's aggregated feed. Every temperature partition market (`KXHIGH*`) forms part of a collectively exhaustive set — when the observed daily maximum definitively passes through a band, that band resolves NO with near-certainty.

- **"between" markets**: NO signal fires when `METAR ≥ strike_hi + 0.5°F`. The +0.5°F buffer ensures NWS integer rounding will place the official daily high above the band ceiling.
- **"under" markets**: NO signal fires when `METAR ≥ strike − 0.5°F`. Same rounding guarantee in the other direction.
- METAR and NOAA observed are cross-checked; signals are suppressed when divergence exceeds `BAND_ARB_MAX_SOURCE_DIVERGENCE_F` (default 4°F) — the primary guard against sensor failures.
- When NOAA has no data yet (METAR is ahead), the market price provides soft confirmation: signals are suppressed if NO ask exceeds `BAND_ARB_NOAA_NONE_MAX_NO_ASK` (default 40¢).
- A fast inner loop (default every 10s) re-checks near-threshold cities between full poll cycles without re-fetching the entire market list.

### Divergence matching (Polymarket / Metaculus / Manifold)
- Market titles and external questions are compared using **stemmed Jaccard similarity** (suffix-stripped tokens — "elections" → "elect", "confirmed" → "confirm").
- Entertainment/sports/esports markets (`KXMVE*`, `KXNBA*`, `KXNHL*`, etc.) are excluded from external matching to prevent flooding.
- Markets with no `last_price` use bid/ask midpoint for divergence calculation.
- Manifold signals require either Polymarket/Metaculus corroboration or divergence below `MANI_MAX_SOLO_DIVERGENCE` (default 50%).

### Trade execution quality gates
Each opportunity passes through, in order:
1. `score ≥ TRADE_MIN_SCORE` (default 0.5)
2. Live orderbook present
3. Minimum temperature edge (temperature markets only, per-source)
4. Market-vs-model disagreement ≥ `NUMERIC_MIN_DISAGREEMENT` (default 10 pp)
5. Same-day cutoff (temperature markets)
6. Kelly criterion ≥ 1 contract
7. Per-ticker cooldown (`TRADE_TICKER_COOLDOWN_MINUTES`, default 30 min)
8. Aggregate exposure cap (`MAX_TOTAL_EXPOSURE_CENTS`)
9. Circuit breaker not active

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

### Exit management
Open positions are monitored every cycle and exited when any of the following triggers:

| Trigger | Description |
|---|---|
| **Profit-take** | Exit when current value ≥ entry cost × (1 + `EXIT_PROFIT_TAKE`). Default 20%. Per-source overrides available. |
| **Stop-loss** | Exit when current value ≤ entry cost × `EXIT_STOP_LOSS`. Default 70% of cost remaining. Per-source overrides available. |
| **Trailing stop** | Exit if the position has ever been up by `EXIT_TRAILING_ACTIVATE` and has since drawn back by `EXIT_TRAILING_DRAWDOWN`. |
| **Counter-signal** | Exit early if a strong opposing signal (≥ `COUNTER_SIGNAL_MIN_EDGE`) appears from ≥ `COUNTER_SIGNAL_MIN_SOURCES` independent sources. |
| **Settlement** | Position expires at Kalshi settlement; outcome recorded. |

High-confidence sources (`noaa_observed`, `metar`, `band_arb`, `nws_climo`, `nws_alert`) are held to settlement by default — profit-take is disabled so the full edge is captured.

### Capital recycling
When a new trade is blocked by `MAX_TOTAL_EXPOSURE_CENTS`, the bot greedily force-exits the most-settled open positions to free capital. Eligible positions are from high-confidence sources with current NO value ≥ `CAPITAL_RECYCLE_MIN_NO_VALUE` (default 97¢ — essentially at settlement). This ensures capital isn't idle in near-resolved positions when new signals appear.

### Dry-run ledger
A live overview file (`dry_run_overview.md`) is updated every cycle showing open positions, unrealized P&L, and cumulative performance — useful for evaluating signal quality before going live. Historical trades with outcomes are stored in `opportunity_log.db`.

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
│                            Metaculus, Manifold) with stemmed Jaccard
├── box_office_matcher.py  Box office gross estimate vs. Kalshi strike matching
├── strike_arb.py          Band-pass arbitrage: METAR vs. KXHIGH partition markets
│                            Emits BandArbSignal for definitively-NO bands
├── scoring.py             Composite score for all opportunity types
├── trade_executor.py      Kelly sizing, quality gates, dry-run/live execution,
│                            circuit breakers, capital recycling, filter statistics
├── exit_manager.py        Profit-take, stop-loss, trailing stop, counter-signal
│                            exit logic; force-exit for capital recycling
├── dry_run_ledger.py      Dry-run position tracking, P&L overview file,
│                            recyclable_trades() for capital recycling
├── opportunity_log.py     SQLite log of surfaced opportunities + raw_forecasts
│                            table for per-source accuracy backtesting
├── win_rate_tracker.py    Per-category win rate analysis
├── portfolio.py           Live position fetcher and summariser
├── state.py               SQLite deduplication for text sources (state.db)
├── data.py                Shared DataPoint dataclass
└── news/
    ├── federal_register.py  Federal Register fetcher (6 agencies)
    ├── rss.py               Generic RSS/Atom feed fetcher (10+ feeds)
    ├── noaa.py              NOAA/NWS forecast + METAR observed fetcher
    │                          (20 cities, high + low; midnight→5 AM window
    │                           for observed min to exclude daytime contamination)
    ├── nws_climo.py         NWS climatological (CLI) product parser
    │                          (official daily high/low from NWS CLI text)
    ├── open_meteo.py        Open-Meteo WMO ensemble forecast fetcher
    │                          (high + low, all 20 cities)
    ├── owm.py               OpenWeatherMap cross-validation fetcher
    ├── binance.py           Binance spot price fetcher (8 crypto assets)
    ├── frankfurter.py       ECB/Frankfurter forex rate fetcher
    ├── bls.py               BLS economic release fetcher
    ├── fred.py              FRED interest rate fetcher
    ├── eia.py               EIA energy price fetcher
    ├── cme_fedwatch.py      CME FedWatch next-meeting probability fetcher
    ├── polymarket.py        Polymarket binary market fetcher
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
FAST_LOOP_INTERVAL=10             # seconds between fast band-arb checks

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
MAX_TOTAL_EXPOSURE_CENTS=5000     # aggregate open exposure cap ($50.00)
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
| `POLL_INTERVAL` | `60` | Seconds between full poll cycles |
| `FAST_LOOP_INTERVAL` | `10` | Seconds between fast band-arb checks |

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
| `MAX_TOTAL_EXPOSURE_CENTS` | `5000` | Aggregate open exposure cap in cents ($50.00) |

### Temperature signal quality
| Env Var | Default | Description |
|---|---|---|
| `TEMP_FORECAST_MIN_EDGE` | `5.0` | Min °F edge for day-1 forecast signals (noaa, nws_hourly, open_meteo) |
| `TEMP_DAY2_MIN_EDGE` | `9.0` | Min °F edge for day-2+ forecast signals (higher MAE) |
| `TEMP_EDGE_OVER_{SOURCE}` | varies | Per-source override for "over" edge threshold |
| `TEMP_EDGE_UNDER_{SOURCE}` | varies | Per-source override for "under" edge threshold |
| `TEMP_OBSERVED_MIN_EDGE_OVER` | `0.5` | Min °F edge for observed YES (over) signals |
| `TEMP_OBSERVED_MIN_EDGE_BETWEEN` | `0.2` | Min °F edge for observed YES (between) signals |
| `TEMP_OBSERVED_MIN_EDGE_UNDER` | `2.0` | Min °F edge for observed YES (under) signals |
| `NOAA_OBS_YES_MIN_LOCAL_HOUR` | `13` | Earliest local hour to allow high-temp observed YES trades |
| `NOAA_OBS_LOW_PAST_LOCAL_HOUR` | `5` | Earliest local hour to allow low-temp observed YES trades |
| `SAME_DAY_CUTOFF_HOURS` | `2.0` | Block temperature trades within N hours of market close |

### Band-pass arbitrage
| Env Var | Default | Description |
|---|---|---|
| `BAND_ARB_EXECUTION_ENABLED` | `true` | Enable/disable band-arb trade execution |
| `BAND_ARB_MIN_NO_ASK` | `1` | Min NO ask in cents — below this, market already priced it |
| `BAND_ARB_MAX_NO_ASK` | `99` | Max NO ask in cents (0 = no cap) |
| `BAND_ARB_NOAA_NONE_MAX_NO_ASK` | `40` | Max NO ask when NOAA absent (market soft-confirmation cap) |
| `BAND_ARB_MAX_SOURCE_DIVERGENCE_F` | `4.0` | Max METAR vs NOAA divergence before suppressing signal |
| `WATCH_THRESHOLD_F` | `2.0` | °F from a band ceiling to add city to fast-loop watchlist |

### Exit management
| Env Var | Default | Description |
|---|---|---|
| `EXIT_PROFIT_TAKE` | `0.20` | Exit when gain ≥ this fraction of entry cost (20%) |
| `EXIT_STOP_LOSS` | `0.70` | Exit when remaining value ≤ this fraction of entry cost (70%) |
| `EXIT_TRAILING_ACTIVATE` | `0.15` | Trailing stop activates after this gain fraction |
| `EXIT_TRAILING_DRAWDOWN` | `0.10` | Trailing stop fires on this drawback from peak |
| `COUNTER_SIGNAL_MIN_EDGE` | `6.0` | Min °F edge for a counter-direction signal to force exit |
| `COUNTER_SIGNAL_MIN_SOURCES` | `2` | Min independent sources required for counter-signal exit |
| `CAPITAL_RECYCLE_MIN_NO_VALUE` | `97` | Min current NO value (¢) to be eligible for force-exit recycling |
| `CAPITAL_RECYCLE_SOURCES` | `band_arb,metar,noaa_observed` | Sources eligible for capital recycling |

### Signal quality (general)
| Env Var | Default | Description |
|---|---|---|
| `NUMERIC_MIN_DISAGREEMENT` | `0.10` | Min model-vs-market probability gap to trade |
| `NUMERIC_MIN_TEMP_EDGE` | `0` | Global min °F edge across all temperature sources (0 = use per-source thresholds) |

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
| `META_MIN_DIVERGENCE` | `0.20` | Min Metaculus vs. Kalshi probability gap |
| `META_MIN_FORECASTERS` | `20` | Min Metaculus community participants |
| `MANI_MIN_DIVERGENCE` | `0.25` | Min Manifold vs. Kalshi probability gap |
| `MANI_MIN_LIQUIDITY` | `500` | Min Manifold liquidity (mana) |

---

## Architecture Notes

- **Market fetching uses a two-pronged strategy** to bypass Kalshi's pagination ordering, which places 10,000+ sports markets before any weather/crypto/political markets. A targeted `series_ticker=` fetch directly retrieves all known numeric series; a throttled general pagination (0.25s/page) collects political/text-matchable markets without hitting rate limits.
- **All HTTP is async** via `aiohttp`; a shared `ClientSession` with a 30-connection pool is reused across all cycles.
- **Deduplication** for text sources is SQLite-backed (`state.db`). Trade and opportunity history are persisted to `opportunity_log.db`, which also stores circuit breaker state, price snapshots for P&L tracking, and a `raw_forecasts` table capturing every weather source's forecast per cycle for post-hoc accuracy analysis.
- **Temperature matching** uses multiple independent weather sources. Sources are weighted by historical accuracy, with `noaa_observed` and `metar` carrying the highest confidence (direct observation, same ASOS station Kalshi uses for settlement). METAR data arrives 5–8 minutes ahead of NOAA's aggregated feed — the core edge for band-pass arbitrage.
- **NWS rounding**: Kalshi temperature markets settle against NWS CLI integer daily highs/lows (rounded to nearest degree). All threshold comparisons include ±0.5°F buffers to guarantee the official rounded value crosses the boundary.
- **Band-pass arb fast loop**: A secondary async loop polls METAR every 10s (configurable) for cities within `WATCH_THRESHOLD_F` of a band ceiling. This allows intraday signals without waiting for the next full poll cycle.
- **Low-temperature signals**: `noaa_observed` returns the running minimum since local midnight. At midnight this equals the current temperature, not the overnight low. The query window is capped at midnight→5 AM local, and a morning gate blocks trades before 05:00 local when the overnight trough hasn't yet been established.
- **Capital recycling**: When the aggregate exposure cap would block a new trade, the bot greedily force-exits the most-settled open positions from high-confidence sources (those with NO value ≥ 97¢). This keeps capital deployed in new signals rather than idle in near-resolved positions.
- **External forecast matching** uses suffix-stripped stemmed Jaccard similarity so "elections"/"elected"/"electing" all map to the same stem "elect", and "confirmed"/"confirmation" both map to "confirm" — dramatically reducing false-negative matches.
- **Authentication** uses RSA-PSS signatures (Kalshi V2 API): `timestamp + method + path` signed with the private key.
- **Adding a new news source**: create a module in `kalshi_bot/news/` returning `list[DataPoint]` (numeric) or `list[dict]` (text), then add it to the task list in `main.py`'s `_poll` coroutine.
- **Adding a new numeric market series**: add the ticker prefix → metric mapping in `market_parser.py → TICKER_TO_METRIC` and `_NUMERIC_PATTERN_PREFIXES`, add the series prefix to `markets.py → NUMERIC_SERIES`, and add the edge scale in `scoring.py → METRIC_EDGE_SCALES`.
- **Adding a new temperature city**: add entries to `noaa.py → CITIES` (and `LOW_CITIES` for low-temp), `KALSHI_STATION_IDS`, and `_CITY_SIGMA_F`; add the ticker prefix mapping in `market_parser.py → TICKER_TO_METRIC`; add the NWS 3-letter code to `nws_climo.py → CLIMO_LOCATIONS`; add the timezone string to `news/open_meteo.py → _CITY_TZ_STRINGS`.

---

## Limitations & Known Issues

- Kalshi's `status=open` pagination returns sports/entertainment markets first; numeric and political markets only appear after 10,000+ entries. The series_ticker targeted fetch works around this, but political market coverage depends on throttled general pagination staying within rate limits.
- AP News and Reuters RSS feeds may be unreachable depending on network/DNS.
- Politico's feed returns HTTP 403 in some environments.
- Binance is used for crypto pricing by default (no key required, no rate limit issues). CoinGecko (legacy) free tier is limited to ~30 requests/minute and is no longer used.
- BLS free tier: 25 queries/day. Use `BLS_API_KEY` for production.
- Jaccard matching is keyword-overlap-based and can produce false positives. The stemmer reduces but does not eliminate mismatches between differently phrased questions.
- `market_discovery.py` in the project root is a legacy reference file — do not modify or import it.
- Per-city temperature sigma values for new cities (launched 2026-04) are initial estimates based on climate type and should be calibrated after 30 days of live data.
