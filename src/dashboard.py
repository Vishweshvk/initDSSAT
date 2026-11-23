"""Small dependency-free SVG charts for the offline evaluation dashboard."""

from html import escape
import json

import pandas as pd

COLORS = {"DSSAT": "#356859", "ML-only": "#d59b32", "Hybrid": "#7257a8",
          "HGB residual": "#7257a8", "Extra Trees": "#2c82a0", "Ridge": "#b45c50"}
METRICS_LABELS = {"dssat_only": "DSSAT", "ml_only": "ML-only", "hybrid": "Hybrid"}
PREDICTION_COLUMNS = {"DSSAT": "dssat_prediction_kg_ha", "ML-only": "ml_only_prediction_kg_ha",
                      "HGB residual": "hybrid_prediction_kg_ha",
                      "Extra Trees": "extra_trees_prediction_kg_ha",
                      "Ridge": "ridge_prediction_kg_ha"}


def _header(title: str, width: int = 760, height: int = 330) -> list[str]:
    return [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}" xmlns="http://www.w3.org/2000/svg">',
            '<rect width="100%" height="100%" rx="12" fill="#fff"/>',
            f'<text x="28" y="32" font-family="Arial,sans-serif" font-size="17" font-weight="700" fill="#173a2b">{escape(title)}</text>']


def _legend(items: list[str], y: int = 302) -> list[str]:
    parts, x = [], 32
    for label in items:
        color = COLORS[label]
        parts.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{color}"/><text x="{x+12}" y="{y+4}" font-family="Arial,sans-serif" font-size="12" fill="#41534a">{escape(label)}</text>')
        x += 122
    return parts


def mae_chart(metrics: dict) -> str:
    labels = [("DSSAT", "dssat_only"), ("ML-only", "ml_only"), ("HGB residual", "hybrid"),
              ("Extra Trees", "extra_trees"), ("Ridge", "ridge")]
    values = [float(metrics["pooled_oof"][key]["mae_kg_ha"]) for _, key in labels]
    maximum = max(values) or 1.0
    svg = _header("Pooled out-of-fold mean absolute error (kg/ha)")
    for index, ((label, _), value) in enumerate(zip(labels, values)):
        y = 66 + index * 46
        width = 500 * value / maximum
        svg.append(f'<text x="30" y="{y+20}" font-family="Arial,sans-serif" font-size="13" fill="#22352a">{label}</text>')
        svg.append(f'<rect x="108" y="{y}" width="{width:.2f}" height="32" rx="6" fill="{COLORS[label]}"/>')
        svg.append(f'<text x="{min(620, 118+width):.2f}" y="{y+21}" font-family="Arial,sans-serif" font-size="13" fill="#22352a">{value:,.1f}</text>')
    svg.append('</svg>')
    return "".join(svg)


def scatter_chart(predictions: pd.DataFrame) -> str:
    cols = ["observed_yield_kg_ha", *PREDICTION_COLUMNS.values()]
    values = predictions[cols].to_numpy(dtype=float).ravel()
    low, high = float(values.min()), float(values.max())
    spread = max(high - low, 1.0)
    low -= spread * 0.05
    high += spread * 0.05
    left, top, size = 74, 58, 236
    coordinate = lambda value: left + (value - low) / (high - low) * size
    y_coordinate = lambda value: top + size - (value - low) / (high - low) * size
    svg = _header("Observed vs predicted yield (kg/ha; shared axes)")
    svg.append(f'<line x1="{left}" y1="{y_coordinate(low):.1f}" x2="{left+size}" y2="{y_coordinate(low):.1f}" stroke="#7b887e"/><line x1="{left}" y1="{top}" x2="{left}" y2="{top+size}" stroke="#7b887e"/>')
    svg.append(f'<line x1="{coordinate(low):.1f}" y1="{y_coordinate(low):.1f}" x2="{coordinate(high):.1f}" y2="{y_coordinate(high):.1f}" stroke="#aab5aa" stroke-dasharray="6,5"/>')
    svg.append(f'<text x="{left+size/2-22:.1f}" y="{top+size+28}" font-family="Arial,sans-serif" font-size="12" fill="#41534a">Observed</text>')
    svg.append(f'<text x="20" y="{top+size/2:.1f}" transform="rotate(-90 20 {top+size/2:.1f})" font-family="Arial,sans-serif" font-size="12" fill="#41534a">Predicted</text>')
    for label, column in PREDICTION_COLUMNS.items():
        for observed, predicted in zip(predictions["observed_yield_kg_ha"], predictions[column]):
            svg.append(f'<circle cx="{coordinate(float(observed)):.2f}" cy="{y_coordinate(float(predicted)):.2f}" r="4" fill="{COLORS[label]}" fill-opacity="0.72"/>')
    svg += _legend(list(PREDICTION_COLUMNS))
    svg.append('</svg>')
    return "".join(svg)


def error_by_experiment_chart(predictions: pd.DataFrame) -> str:
    errors = predictions.copy()
    for label, column in PREDICTION_COLUMNS.items():
        errors[f"{label}_error"] = errors[column] - errors["observed_yield_kg_ha"]
    groups = []
    for experiment, group in errors.groupby("experiment_id", sort=True):
        groups.append((str(experiment), {label: float(group[f"{label}_error"].mean()) for label in PREDICTION_COLUMNS}))
    max_abs = max((abs(value) for _, vals in groups for value in vals.values()), default=1.0) or 1.0
    center, span = 390, 300
    svg = _header("Mean signed prediction error by held-out experiment (kg/ha)")
    svg.append(f'<line x1="{center}" y1="56" x2="{center}" y2="{260}" stroke="#8b978d" stroke-dasharray="4,4"/>')
    for index, (experiment, vals) in enumerate(groups):
        y0 = 68 + index * 64
        svg.append(f'<text x="28" y="{y0+22}" font-family="Arial,sans-serif" font-size="12" fill="#22352a">{escape(experiment)}</text>')
        for model_index, label in enumerate(PREDICTION_COLUMNS):
            value = vals[label]
            bar_width = abs(value) / max_abs * span
            x = center if value >= 0 else center - bar_width
            y = y0 + model_index * 15
            svg.append(f'<rect x="{x:.2f}" y="{y}" width="{bar_width:.2f}" height="10" rx="3" fill="{COLORS[label]}"/>')
    svg += _legend(list(PREDICTION_COLUMNS), y=302)
    svg.append('</svg>')
    return "".join(svg)


def dashboard_charts(config) -> dict | None:
    base = config.project_data_dir / "evaluation"
    metrics_path, predictions_path = base / "metrics.json", base / "oof_predictions.csv"
    metadata_path = base / "metadata.json"
    if not metrics_path.is_file() or not predictions_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        predictions = pd.read_csv(predictions_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if (predictions.empty or metadata.get("models", {}).get("residual", {}).get("estimator")
            != "HistGradientBoostingRegressor"):
        return None
    return {"mae": mae_chart(metrics), "scatter": scatter_chart(predictions),
            "errors": error_by_experiment_chart(predictions)}
