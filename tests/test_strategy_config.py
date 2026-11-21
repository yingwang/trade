"""Config validation for factor names and the earnings: section (review
2026-11-21, F9), checked against every config this repository runs."""

import copy
import json
import re
from pathlib import Path

import pytest
import yaml

from quant.signals.factors import KNOWN_FACTORS, PRICE_FACTORS, SignalGenerator
from quant.utils.config import EARNINGS_PARTS, load_config, validate_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted(ROOT.glob("config*.yaml"))


def _deep_update(base: dict, overrides: dict) -> dict:
    """research/earnings_book.py's and run.py's merge."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _documented_overrides():
    """Every ``--override '{...}'`` example under research/, plus run.py's."""
    found = []
    for path in sorted((ROOT / "research").glob("*.py")):
        for match in re.finditer(r"--override '(\{.*?\})'", path.read_text()):
            found.append((path.name, match.group(1)))
    run_py = (ROOT / "run.py").read_text()
    found += [("run.py", m.group(1)) for m in re.finditer(r"# '(\{\"trend\".*?\})'", run_py)]
    return found


# Which config each documented example runs against.
OVERRIDE_BASE = {"earnings_book.py": "config_earnings.yaml", "run.py": "config_trend.yaml"}

# The 2026-10-22 variants (docs/research/2026-10-22-alpha-audit.md, section 5)
# as overrides of config_earnings.yaml.
AUDIT_VARIANTS = [
    {"earnings": {"sizing": "inverse_vol"}},
    {"earnings": {"regime_caps": {"0": None, "1": 0.6, "2": 0.0}}},
    {"earnings": {"trailing_stop": 0.2}},
    {"earnings": {"fixed_gross": None}, "portfolio": {"target_volatility": 0.25},
     "leverage": {"max_leverage": 1.5}},
    {"earnings": {"require_uptrend": True}},
    {"earnings": {"positions": 15}},
    {"earnings": {"positions": 40}},
    {"earnings": {"weights": {"days_to_next": 1.0}}},
    {"earnings": {"weights": {"days_to_next": 1.0}, "next_known_within_days": 30}},
    {"earnings": {"weights": {"days_to_next": 1.0}, "next_known_within_days": 14}},
    {"earnings": {"fixed_gross": 1.3}, "leverage": {"max_leverage": 1.3}},
    {"earnings": {"fixed_gross": 1.5}, "leverage": {"max_leverage": 1.5}},
]


@pytest.mark.parametrize("path", CONFIGS, ids=[p.name for p in CONFIGS])
def test_every_repository_config_loads(path):
    load_config(path)


def test_documented_override_examples_exist_and_validate():
    examples = _documented_overrides()
    assert {name for name, _ in examples} >= {"earnings_book.py", "run.py"}
    for name, text in examples:
        assert name in OVERRIDE_BASE, f"map {name}'s --override example to its config"
        config = load_config(ROOT / OVERRIDE_BASE[name])
        validate_config(_deep_update(config, json.loads(text)))


@pytest.mark.parametrize("override", AUDIT_VARIANTS, ids=lambda o: json.dumps(o)[:60])
def test_audit_variants_validate(override):
    config = load_config(ROOT / "config_earnings.yaml")
    validate_config(_deep_update(config, copy.deepcopy(override)))


def _raw(name):
    return yaml.safe_load((ROOT / name).read_text())


def test_unknown_factor_name_is_rejected_on_both_paths(config):
    raw = _raw("config.yaml")
    raw["signals"]["factor_weights"]["high_proximty"] = raw["signals"]["factor_weights"].pop(
        "high_proximity"
    )
    with pytest.raises(ValueError, match="high_proximty"):
        validate_config(raw)
    bad = copy.deepcopy(config)
    bad["signals"]["factor_weights"]["momentun"] = 0.3
    with pytest.raises(ValueError, match="momentun"):
        SignalGenerator(bad)


def test_factor_registry_matches_the_builders(config, synthetic_prices):
    """Every price factor name is one SignalGenerator can build."""
    cfg = copy.deepcopy(config)
    cfg["signals"]["factor_weights"] = {name: 1.0 for name in PRICE_FACTORS}
    gen = SignalGenerator(cfg)
    gen.generate(synthetic_prices, synthetic_prices.pct_change(fill_method=None).dropna(how="all"))
    assert set(PRICE_FACTORS) <= set(gen.last_factors_)
    assert set(KNOWN_FACTORS) - set(PRICE_FACTORS) == {"value", "quality"}


@pytest.mark.parametrize("change, message", [
    ({"weights": {"sue": 1.0, "eps_grwoth": 1.0, "momentum": 1.0}}, "eps_grwoth"),
    ({"min_part": 2}, "unknown key"),
    ({"min_parts": 4}, "min_parts 4 exceeds the 3 part"),
    ({"min_parts": 0}, "min_parts"),
    ({"positions": 0}, "positions"),
    ({"rank_buffer": 20}, "rank_buffer"),
    ({"fixed_gross": 1.3}, "fixed_gross"),
    ({"fixed_gross": -0.5}, "fixed_gross"),
    ({"reentry_cooldown_sessions": -1}, "reentry_cooldown_sessions"),
    ({"reentry_cooldown_sessions": 2.5}, "reentry_cooldown_sessions"),
    ({"trailing_stop": 1.0}, "trailing_stop"),
    ({"min_coverage": 1.5}, "min_coverage"),
    ({"sizing": "equal_weight"}, "sizing"),
])
def test_bad_earnings_values_are_rejected(change, message):
    raw = _raw("config_earnings.yaml")
    raw["earnings"].update(change)
    with pytest.raises(ValueError, match=message):
        validate_config(raw)


def test_validator_parts_are_the_strategy_parts(config):
    """EARNINGS_PARTS is the set EarningsStrategy.parts_on can build."""
    from tests.test_earnings_strategy import _config, _events, _prices, _snapshots
    from quant.earnings_strategy import EarningsStrategy

    cfg = _config(config)
    cfg["earnings"]["weights"] = {part: 1.0 for part in EARNINGS_PARTS}
    strategy = EarningsStrategy(cfg)
    prices = _prices()
    strategy._snapshots = _snapshots(prices.index)
    strategy._events = _events(prices.index)
    pre = strategy.precompute(prices)
    parts = strategy.parts_on(pre, prices.index[-1])
    assert set(parts.columns) == set(EARNINGS_PARTS)
