"""Board membership, target, and canonical labels.

These are the SOURCE OF TRUTH for what Pythia P0 evaluates. The 20-symbol
macro board is taken verbatim from raptor-intel's MacroChangePanel; when
that board grows or shrinks, this list must be updated in the same PR.
"""

from __future__ import annotations

# The target we forecast. QQQ = Nasdaq-100 ETF.
TARGET: str = "QQQ"

# The 19 non-target macro-board symbols (equities, commodities, metals).
# Verified against raptor-intel/frontend/src/components/MacroChangePanel.tsx.
MACRO_COVARIATES: tuple[str, ...] = (
    "SPY",
    "DIA",
    "IWM",  # broad indices
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOG",
    "AMZN",
    "META",
    "TSLA",  # mag-7
    "GLD",
    "SLV",
    "GDX",  # metals
    "USO",
    "UGA",
    "UNG",
    "DBE",  # energy
    "CORN",
    "WEAT",  # ag
)

# The full board (target + covariates) = 20 symbols.
BOARD_SYMBOLS: tuple[str, ...] = (TARGET,) + MACRO_COVARIATES

# --- Rates / vol proxies (task calls for VIX + rates) ---
# NOTE for helen: raptor's staging.quote_raw does NOT ship ^VIX or TNX today
# (verified 2026-07-11). Pythia's data assembler will look for them by symbol
# and simply omit whichever isn't present, logging a warning. When raptor
# starts ingesting them, they'll materialise here without a code change.
VIX_SYMBOLS: tuple[str, ...] = ("^VIX", "VIX", "VIXY")  # any one is fine
RATE_SYMBOLS: tuple[str, ...] = ("^TNX", "TNX", "IEF", "TLT")  # any one


# --- Postgres source ---
DEFAULT_DB_DSN: str = "postgresql://hollywood@postgres.hollywood.svc.cluster.local:5432/raptor"
QUOTE_TABLE: str = "staging.quote_raw"
