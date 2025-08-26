
import json
import os
import sys

from ufc_predictor.train import train


def _load_overrides(path: str) -> dict:
    if path.lower().endswith((".yml", ".yaml")):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise SystemExit("PyYAML es requerido para archivos YAML") from exc
        with open(path) as f:
            return yaml.safe_load(f)
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    grid_path = sys.argv[1] if len(sys.argv) > 1 else None
    overrides = _load_overrides(grid_path) if grid_path else None

    model_names = os.getenv("MODEL_NAMES")
    nombres = [m.strip() for m in model_names.split(",") if m.strip()] if model_names else None
    fast = os.getenv("FAST_MODE", "").lower() in ("1", "true", "yes")
    extended = os.getenv("EXTENDED_SEARCH", "").lower() in ("1", "true", "yes")
    train(
        "Winner",
        model_names=nombres,
        fast_mode=fast,
        extended_search=extended,
        grid_overrides=overrides,
    )
