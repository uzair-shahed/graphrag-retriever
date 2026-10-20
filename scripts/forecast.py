"""Model spend forecasting based on planned cache misses and unit cost.

Last modified: 2026-10-20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def measure_unit_cost_from_cache(cache_dir: Path) -> float:
    """Measure the average unit cost per API call from existing cache recordings.

    Args:
        cache_dir: Directory containing gateway record cache JSON files.

    Returns:
        Average cost in USD per call, or default estimate 0.001 if no cost data found.
    """
    if not cache_dir.exists():
        return 0.001

    costs: list[float] = []
    for file in cache_dir.glob("*.json"):
        try:
            with open(file, encoding="utf-8") as f:
                data = json.load(f)
            cost = data.get("cost")
            if cost is None:
                cost = (data.get("response") or {}).get("usage", {}).get("cost")
            if cost is not None and float(cost) > 0:
                costs.append(float(cost))
        except (json.JSONDecodeError, OSError):
            continue

    if not costs:
        return 0.001
    return sum(costs) / len(costs)


def calculate_forecast(planned_calls: int, unit_cost: float) -> dict[str, Any]:
    """Calculate projected cost for planned API calls.

    Args:
        planned_calls: Number of planned cache misses / API calls.
        unit_cost: Cost in USD per call.

    Returns:
        Dictionary with planned calls, unit cost, and total forecast in USD.
    """
    total = planned_calls * unit_cost
    return {
        "planned_calls": planned_calls,
        "unit_cost_usd": unit_cost,
        "forecast_usd": round(total, 6),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Forecast model API spend based on cache misses")
    parser.add_argument("--planned-calls", type=int, default=10, help="Planned cache misses / API calls")
    parser.add_argument("--unit-cost", type=float, default=None, help="Explicit unit cost in USD per call")
    parser.add_argument("--cache-dir", default="runs/cache", help="Path to cache directory")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    args = parser.parse_args(argv)

    if args.unit_cost is not None:
        unit_cost = args.unit_cost
    else:
        unit_cost = measure_unit_cost_from_cache(Path(args.cache_dir))

    forecast = calculate_forecast(args.planned_calls, unit_cost)

    if args.json:
        print(json.dumps(forecast, indent=2))
    else:
        print(
            f"Forecast: {forecast['planned_calls']} planned calls × "
            f"${forecast['unit_cost_usd']:.6f} = ${forecast['forecast_usd']:.6f} USD"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
