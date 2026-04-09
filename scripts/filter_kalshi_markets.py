"""Filter kalshi_markets.jsonl to resolved KX* markets with domain labels.

Reads data/kalshi_markets.jsonl (22,186 markets, mix of KX* and MET-* tickers),
keeps only real resolved Kalshi (KX*) markets, assigns a kalshi_category domain
label based on ticker prefix, and writes data/kalshi_resolved_markets.jsonl.

Output fields:
    ticker, title, result, close_time, rules_primary, category, kalshi_category
"""

import json
from pathlib import Path
from collections import Counter

INPUT  = Path(__file__).parent.parent / "data" / "kalshi_markets.jsonl"
OUTPUT = Path(__file__).parent.parent / "data" / "kalshi_resolved_markets.jsonl"

# ---------------------------------------------------------------------------
# Ticker-prefix → kalshi_category mapping
# Categories match the bot's _SOURCE_GROUPS domains.
# ---------------------------------------------------------------------------

_POLITICS = {
    "KXTRUMPS", "KXTRUMPM", "KXTRUMPA", "KXTRUTHS", "KXPRESME",
    "KXTXPRIM", "KXILPRIM", "KXNCPRIM", "KXMSPRIM", "KXSENATE",
    "KXCABOUT", "KXCONGRE", "KXVOTESH", "KXVOTEMU",
    "KXGOVTSH", "KXDHSFUN", "KXACAHOU", "KXGOVTFU",
    "KXSCOTUS", "KXNEXTIR", "KXDENMAR", "KXJPNPM-", "KXJAPANH",
    "KXHOCHUL", "KXNYCMDE", "KXEPSTEI", "KXNEXTDH", "KXMULLIN",
    "KXGORTON", "KXDUTCHC", "KXBARRME", "KXWHVISI", "KXMOVPOR",
    "KXMOVTX1", "KXTXSEND", "KXHORMUZ", "KXVANCEM", "KXSECPRE",
    "KXSTARME", "KXMAMDAN", "KXNYCMDE", "KXIL9D-2",
}

_SPORTS = {
    "KXNCAABM", "KXNCAAMB", "KXNCAAWB", "KXNCAAHO", "KXNCAABB",
    "KXNEXTNF", "KXNEXTTE", "KXMICHCO", "KXNYGCOA", "KXTENNCO",
    "KXDPWORL", "KXLIVTOP", "KXLIVR1L",
    "KXBRASIL", "KXLALIGA", "KXWCROUN", "KXWCGROU",
    "KXUCL-26", "KXUCLFIN", "KXUCLRO4",
    "KXWBCMVP", "KXATPSET", "KXATPCHA", "KXALCARA",
    "KXCS2MAP", "KXCS2GAM", "KXCS2TOT",
    "KXLOLGAM", "KXVALORA",
    "KXTATUMR", "KXKNUEPP",
    "KXMARMAD", "KXWMARMA",
    "KXFIDECA", "KXBBLGAM", "KXNCPRIM",
    "KXFIGHTM",
}

_ECONOMICS = {
    "KXNASDAQ", "KXINXU-2", "KXINX-26",
    "KXGOLDD-", "KXBRENTD", "KXSILVER", "KXCOPPER",
    "KXUSTYLD", "KXTNOTED",
    "KXAAAGAS", "KXSPRLVL",
    "KXEARNIN", "KXMENTIO",
    "KXIPO-25",
}

_CRYPTO = {
    "KXBNBD-2", "KXBNB-26", "KXBNB15M",
    "KXHYPED-", "KXHYPE-2", "KXHYPE15",
    "KXSHIBAD", "KXSHIBA-",
}

_ENTERTAINMENT = {
    "KXOSCARN", "KXOSCARS", "KXOSCARA", "KXOSCARP", "KXOSCARC",
    "KXOSCARD", "KXOSCARM", "KXOSCARG", "KXOSCARE", "KXOSCARI",
    "KXOSCARI", "KXOSCARV",
    "KXTOPSONG", "KXRANKLI", "KXTOPSON",
    "KX20SONG", "KX10SONG", "KX1ALBUM",
    "KXSPOTIF", "KXFEATUR", "KXALBUMR", "KXSONGRE", "KXWCUPSO",
    "KXLOVEIS", "KXTRAITO", "KXLOVEOV", "KXBACHEL",
    "KXTPUSAH", "KXMRBEAS", "KXMEDIAR",
    "KXSBSETL", "KXPERFOR",
    "KXAUCTIO", "KXTVSEAS", "KXNEXTMA",
    "KXNCAABM",  # "what will announcers say" — text market
}


def _assign_category(ticker: str) -> str:
    for length in (8, 7, 6):
        pfx = ticker[:length]
        if pfx in _POLITICS:
            return "politics"
        if pfx in _SPORTS:
            return "sports"
        if pfx in _ECONOMICS:
            return "economics"
        if pfx in _CRYPTO:
            return "crypto"
        if pfx in _ENTERTAINMENT:
            return "entertainment"

    # Fallback: try contains-match on well-known patterns
    t = ticker.upper()
    if any(kw in t for kw in ("TRUMP", "CONGRESS", "SENATE", "PRIM", "VOTE", "SCOTUS",
                               "POTUS", "PRESME", "VANCEM", "HOCHUL", "MAMDAN")):
        return "politics"
    if any(kw in t for kw in ("NBA", "NFL", "NHL", "MLB", "NCAA", "GOLF", "TENNIS",
                               "SOCCER", "FIGHT", "UFC", "CS2", "LOL", "VALO")):
        return "sports"
    if any(kw in t for kw in ("NASDAQ", "INXU", "EARN", "GOLD", "BRENT", "SILVER",
                               "COPPER", "YIELD", "GAS", "SPR", "IPO")):
        return "economics"
    if any(kw in t for kw in ("BNB", "HYPE", "SHIB", "BTCD", "ETHU", "DOGE", "XRP",
                               "SOL", "ADA", "AVAX")):
        return "crypto"
    if any(kw in t for kw in ("OSCAR", "SONG", "ALBUM", "SPOT", "LOVE", "BEAST",
                               "TRAITOR", "BACHEL", "MEDIA", "PERFOR", "SUPER")):
        return "entertainment"

    return "other"


def main() -> None:
    lines = INPUT.read_text().strip().split("\n")
    print(f"Loaded {len(lines)} markets from {INPUT.name}")

    kept: list[dict] = []
    skipped_non_kx = 0
    skipped_no_result = 0
    cat_counts: Counter = Counter()

    for line in lines:
        d = json.loads(line)
        ticker = d.get("ticker", "")

        if not ticker.startswith("KX"):
            skipped_non_kx += 1
            continue

        result = d.get("result", "")
        if result not in ("yes", "no"):
            skipped_no_result += 1
            continue

        kalshi_cat = _assign_category(ticker)
        cat_counts[kalshi_cat] += 1

        kept.append({
            "ticker":          ticker,
            "title":           d.get("title", ""),
            "result":          result,
            "close_time":      d.get("close_time", ""),
            "open_time":       d.get("open_time", ""),
            "resolution_time": d.get("resolution_time", ""),
            "rules_primary":   d.get("rules_primary", ""),
            "category":        d.get("category", ""),
            "kalshi_category": kalshi_cat,
        })

    print(f"Skipped {skipped_non_kx} non-KX markets (MET-*, etc.)")
    print(f"Skipped {skipped_no_result} unresolved KX markets")
    print(f"Kept {len(kept)} resolved KX* markets\n")
    print("Category distribution:")
    for cat, n in cat_counts.most_common():
        pct = 100 * n / len(kept)
        print(f"  {cat:<15} {n:>5} ({pct:.1f}%)")

    OUTPUT.write_text("\n".join(json.dumps(d) for d in kept) + "\n")
    print(f"\nWrote {len(kept)} markets to {OUTPUT.name}")


if __name__ == "__main__":
    main()
