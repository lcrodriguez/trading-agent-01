from typing import Type
from .base import BaseStrategy

STRATEGY_REGISTRY: dict[str, Type[BaseStrategy]] = {}


def register_strategy(cls: Type[BaseStrategy]) -> Type[BaseStrategy]:
    STRATEGY_REGISTRY[cls.name] = cls
    return cls


def get_strategy(name: str) -> Type[BaseStrategy]:
    if name not in STRATEGY_REGISTRY:
        raise KeyError(f"Unknown strategy '{name}'. Available: {list(STRATEGY_REGISTRY)}")
    return STRATEGY_REGISTRY[name]


def list_strategies() -> list[dict]:
    return [
        {"name": cls.name, "description": cls.description, "params": cls.param_specs()}
        for cls in STRATEGY_REGISTRY.values()
    ]


# Auto-import strategies to trigger registration
from . import buy_and_hold, sma_crossover, renko_sma, renko_sma_dca, ha_sma_dca, ha_sma  # noqa: E402, F401
