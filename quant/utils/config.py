"""Configuration loader."""

import math
import yaml
from datetime import datetime
from pathlib import Path


def _number(section: dict, key: str, *, minimum=None, maximum=None) -> float:
    raw = section.get(key)
    if isinstance(raw, bool):
        raise ValueError(f"Configuration value {key} must be numeric")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Configuration value {key} must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"Configuration value {key} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"Configuration value {key} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"Configuration value {key} must be <= {maximum}")
    return value


def static_sector_map(config: dict):
    """The universe's sector table from config, as a Series, or None if absent.

    When present it is the single sector classification for signals, sector
    caps and attribution on both the backtest and the live path.
    """
    import pandas as pd

    universe = config.get("universe", {}) if isinstance(config, dict) else {}
    table = universe.get("sectors") if isinstance(universe, dict) else None
    if not table:
        return None
    return pd.Series(
        {str(symbol).strip().upper(): str(sector).strip() for symbol, sector in table.items()},
        dtype="object",
    )


# The parts EarningsStrategy can score (earnings.weights keys) and every key
# the earnings: section may carry: EarningsStrategy's own and the trend
# machinery's, which EarningsStrategy reads from this section.  A key not
# listed here is a typo the strategy would ignore, so it is rejected.
EARNINGS_PARTS = ("sue", "eps_growth", "days_to_next", "momentum")
EARNINGS_KEYS = frozenset({
    # EarningsStrategy
    "weights", "momentum_lookback", "momentum_skip", "sue_lookback",
    "next_known_within_days", "min_parts", "sizing", "require_uptrend",
    "fixed_gross", "earnings_limit", "universe_file", "universe_lead_days",
    "min_coverage",
    # TrendStrategy, read from earnings: for this book
    "positions", "rank_buffer", "skip_days", "lookback_long", "lookback_mid",
    "lookback_short", "blend", "vol_window", "sma_long", "sma_short",
    "trailing_stop", "trailing_window", "benchmark_vol_window",
    "benchmark_vol_high", "breadth_min", "regime_caps", "deleverage_tolerance",
    "reentry_cooldown_sessions",
})


def _whole(section: dict, key: str, default: int, *, minimum: int) -> int:
    raw = section.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < minimum:
        raise ValueError(f"earnings.{key} must be a whole number >= {minimum}")
    return raw


def _validate_earnings(earnings, max_leverage: float) -> None:
    """The earnings: section, checked like trend: plus what the book relies on.

    Nothing here was validated before: a misspelt part was skipped, and a
    min_parts above the number of parts left every name unrankable, which
    the live path turned into an empty target.
    """
    if not isinstance(earnings, dict):
        raise ValueError("earnings must be a mapping")
    unknown = sorted(str(key) for key in set(earnings) - EARNINGS_KEYS)
    if unknown:
        raise ValueError(f"earnings has unknown key(s) {unknown}")

    weights = earnings.get("weights", {part: 1.0 for part in EARNINGS_PARTS})
    if not isinstance(weights, dict) or not weights:
        raise ValueError("earnings.weights must be a non-empty mapping")
    unknown_parts = sorted(str(name) for name in set(weights) - set(EARNINGS_PARTS))
    if unknown_parts:
        raise ValueError(
            f"earnings.weights names unknown part(s) {unknown_parts}; "
            f"known: {list(EARNINGS_PARTS)}"
        )
    positive = 0
    for name in weights:
        weight = _number(weights, name, minimum=0)
        positive += weight > 0
    if not positive:
        raise ValueError("earnings.weights needs at least one positive weight")

    min_parts = _whole(earnings, "min_parts", 2, minimum=1)
    if min_parts > positive:
        raise ValueError(
            f"earnings.min_parts {min_parts} exceeds the {positive} part(s) with a "
            "positive weight: no name could be ranked"
        )
    positions = _whole(earnings, "positions", 10, minimum=1)
    rank_buffer = _whole(earnings, "rank_buffer", 15, minimum=1)
    if rank_buffer < positions:
        raise ValueError("earnings.rank_buffer must be at least earnings.positions")
    _whole(earnings, "reentry_cooldown_sessions", 5, minimum=0)

    fixed = earnings.get("fixed_gross")
    if fixed not in (None, "", False):
        gross = _number(earnings, "fixed_gross", minimum=0)
        if gross <= 0 or gross > max_leverage:
            raise ValueError(
                f"earnings.fixed_gross must be in (0, leverage.max_leverage={max_leverage:g}]"
            )
    if "trailing_stop" in earnings:
        stop = _number(earnings, "trailing_stop", minimum=0)
        if stop >= 1:
            raise ValueError("earnings.trailing_stop must be below 1 (0 turns it off)")
    if "min_coverage" in earnings:
        _number(earnings, "min_coverage", minimum=0, maximum=1)
    sizing = earnings.get("sizing", "inverse_vol")
    if sizing not in ("equal", "inverse_vol"):
        raise ValueError("earnings.sizing must be 'equal' or 'inverse_vol'")
    caps = earnings.get("regime_caps")
    if caps is not None:
        if not isinstance(caps, dict) or not caps:
            raise ValueError("earnings.regime_caps must be a non-empty mapping")
        for lights, cap in caps.items():
            # YAML gives int keys, a JSON override string ones; the strategy int()s them.
            if isinstance(lights, bool) or not str(lights).isdigit():
                raise ValueError("earnings.regime_caps keys must be light counts 0, 1, 2")
            if cap is not None and (
                isinstance(cap, bool) or not isinstance(cap, (int, float))
                or not math.isfinite(float(cap)) or float(cap) < 0
            ):
                raise ValueError(f"earnings.regime_caps[{lights}] must be null or >= 0")


def validate_config(config: dict) -> dict:
    """Validate risk-critical configuration before any data or orders run."""
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    required = {"universe", "data", "signals", "portfolio", "risk", "backtest"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Configuration is missing sections: {sorted(missing)}")

    universe = config["universe"]
    symbols = universe.get("symbols") if isinstance(universe, dict) else None
    benchmark = universe.get("benchmark") if isinstance(universe, dict) else None
    if (
        not isinstance(symbols, list)
        or not symbols
        or any(not isinstance(symbol, str) or not symbol.strip() for symbol in symbols)
    ):
        raise ValueError("universe.symbols must be a non-empty list of ticker strings")
    normalized = [symbol.strip().upper() for symbol in symbols]
    if len(normalized) != len(set(normalized)):
        raise ValueError("universe.symbols contains duplicate tickers")
    if not isinstance(benchmark, str) or not benchmark.strip():
        raise ValueError("universe.benchmark must be a ticker string")
    if benchmark.strip().upper() in set(normalized):
        raise ValueError("universe.benchmark must not also appear in universe.symbols")
    sectors = universe.get("sectors")
    if sectors is not None:
        if not isinstance(sectors, dict) or any(
            not isinstance(sector, str) or not sector.strip() for sector in sectors.values()
        ):
            raise ValueError("universe.sectors must map each ticker to a non-empty sector name")
        table = {str(symbol).strip().upper() for symbol in sectors}
        uncovered = sorted(set(normalized) - table)
        if uncovered:
            raise ValueError(f"universe.sectors is missing entries for: {uncovered}")

    data = config["data"]
    _number(data, "lookback_years", minimum=1)
    if not isinstance(data.get("frequency"), str) or not data["frequency"].strip():
        raise ValueError("data.frequency must be a non-empty string")

    signals = config["signals"]
    windows = signals.get("momentum_windows")
    if (
        not isinstance(windows, list)
        or not windows
        or any(isinstance(value, bool) or int(value) <= 0 for value in windows)
    ):
        raise ValueError("signals.momentum_windows must contain positive integers")
    factor_weights = signals.get("factor_weights")
    if not isinstance(factor_weights, dict) or not factor_weights:
        raise ValueError("signals.factor_weights must be a non-empty mapping")
    from quant.signals.factors import KNOWN_FACTORS, unknown_factor_names

    unknown = unknown_factor_names(factor_weights)
    if unknown:
        # A misspelt factor used to be skipped silently and the others
        # renormalized, so the book ran without it.
        raise ValueError(
            f"signals.factor_weights names unknown factor(s) {unknown}; "
            f"known: {list(KNOWN_FACTORS)}"
        )
    parsed_weights = []
    for name, raw in factor_weights.items():
        if not isinstance(name, str) or isinstance(raw, bool):
            raise ValueError("signals.factor_weights contains an invalid entry")
        try:
            weight = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Factor weight {name} must be numeric") from exc
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"Factor weight {name} must be finite and non-negative")
        parsed_weights.append(weight)
    if not any(weight > 0 for weight in parsed_weights):
        raise ValueError("At least one factor weight must be positive")

    portfolio = config["portfolio"]
    max_positions = portfolio.get("max_positions")
    if (
        isinstance(max_positions, bool)
        or not isinstance(max_positions, int)
        or max_positions <= 0
        or max_positions > len(symbols)
    ):
        raise ValueError("portfolio.max_positions must fit the configured universe")
    max_weight = _number(portfolio, "max_position_weight", minimum=0, maximum=1)
    min_weight = _number(portfolio, "min_position_weight", minimum=0, maximum=1)
    if max_weight <= 0 or min_weight > max_weight:
        raise ValueError("Portfolio position-weight bounds are invalid")
    if _number(portfolio, "target_volatility", minimum=0) <= 0:
        raise ValueError("portfolio.target_volatility must be positive")
    _number(portfolio, "rebalance_frequency_days", minimum=1)
    _number(portfolio, "transaction_cost_bps", minimum=0)
    _number(portfolio, "max_turnover_per_rebalance", minimum=0, maximum=1)
    _number(portfolio, "alpha_scale", minimum=0)
    if _number(portfolio, "risk_aversion", minimum=0) <= 0:
        raise ValueError("portfolio.risk_aversion must be positive")
    # Minimum trade size (optional; defaults live in quant.execution.broker).
    if "min_trade_notional_usd" in portfolio:
        _number(portfolio, "min_trade_notional_usd", minimum=0)
    if "min_trade_fraction" in portfolio:
        _number(portfolio, "min_trade_fraction", minimum=0, maximum=1)
    # Optional: how target weights are built from the scores.  "proportional"
    # (the default) is what the live books run; "mvo" is the research-only
    # mean-variance path.
    allocation = portfolio.get("allocation", "proportional")
    if (
        not isinstance(allocation, str)
        or allocation.strip().lower() not in {"proportional", "mvo"}
    ):
        raise ValueError("portfolio.allocation must be 'proportional' or 'mvo'")
    # Optional: width of the optimizer's smooth turnover penalty.  Zero would
    # bring back the non-differentiable L1 term that stopped SLSQP converging.
    if "turnover_penalty_epsilon" in portfolio:
        if _number(portfolio, "turnover_penalty_epsilon", minimum=0) <= 0:
            raise ValueError("portfolio.turnover_penalty_epsilon must be positive")

    risk = config["risk"]
    for key in ("max_drawdown_limit", "max_sector_weight"):
        if _number(risk, key, minimum=0, maximum=1) <= 0:
            raise ValueError(f"risk.{key} must be positive")
    # A fixed stop of 0 means "no fixed stop": the backtest engine and the
    # live runner both skip the check then (the earnings book runs that way).
    _number(risk, "stop_loss_pct", minimum=0, maximum=1)
    if "reentry_cooldown_sessions" in risk:
        _number(risk, "reentry_cooldown_sessions", minimum=0)

    leverage = config.get("leverage", {})
    if leverage:
        max_leverage = _number(leverage, "max_leverage", minimum=0)
        if max_leverage <= 0:
            raise ValueError("leverage.max_leverage must be positive")
        _number(leverage, "margin_annual_rate", minimum=0)
        thresholds = leverage.get("regime_thresholds", {})
        low = _number(thresholds, "low", minimum=0)
        high = _number(thresholds, "high", minimum=0)
        if low >= high:
            raise ValueError("leverage regime threshold low must be below high")
        for name, cap in leverage.get("regime_leverage_caps", {}).items():
            if not math.isfinite(float(cap)) or float(cap) < 0 or float(cap) > max_leverage:
                raise ValueError(f"Leverage cap {name} is outside [0, max_leverage]")

    trend = config.get("trend") or {}
    if "reentry_cooldown_sessions" in trend:
        # The trend book's other keys have defaults in TrendStrategy; this one
        # is checked because a cooldown that is not a whole number of sessions,
        # or a negative one, would quietly change what a scheduled rebalance
        # may buy.
        cooldown = trend["reentry_cooldown_sessions"]
        if isinstance(cooldown, bool) or not isinstance(cooldown, int) or cooldown < 0:
            raise ValueError(
                "trend.reentry_cooldown_sessions must be a whole number of sessions, zero or more"
            )

    if config.get("earnings") is not None:
        max_leverage = float(leverage.get("max_leverage", 1.0)) if leverage else 1.0
        _validate_earnings(config["earnings"], max_leverage)

    backtest = config["backtest"]
    if _number(backtest, "initial_capital", minimum=0) <= 0:
        raise ValueError("backtest.initial_capital must be positive")
    _number(backtest, "slippage_bps", minimum=0)
    _number(backtest, "market_impact_coeff", minimum=0)
    risk_free = _number(backtest, "risk_free_rate")
    if risk_free <= -1:
        raise ValueError("backtest.risk_free_rate must be greater than -1")
    start = datetime.fromisoformat(str(backtest["start_date"]))
    end_raw = backtest.get("end_date")
    if end_raw is not None and datetime.fromisoformat(str(end_raw)) < start:
        raise ValueError("backtest.end_date must not precede start_date")

    return config


def load_config(path: str = None) -> dict:
    """Load configuration from YAML file."""
    repo_root = Path(__file__).resolve().parents[2]
    if path is None:
        path = repo_root / "config.yaml"
    else:
        path = Path(path)
        if not path.is_absolute() and not path.exists():
            candidate = repo_root / path
            if candidate.exists():
                path = candidate

    with open(path) as f:
        return validate_config(yaml.safe_load(f))
