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

    risk = config["risk"]
    for key in ("max_drawdown_limit", "max_sector_weight", "stop_loss_pct"):
        if _number(risk, key, minimum=0, maximum=1) <= 0:
            raise ValueError(f"risk.{key} must be positive")

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
    if path is None:
        path = Path(__file__).resolve().parents[2] / "config.yaml"
    else:
        path = Path(path)

    with open(path) as f:
        return validate_config(yaml.safe_load(f))
