"""Single-file HTML report with embedded matplotlib figures.

Every piece of interpolated text goes through html.escape (via _e); no
data-derived string is placed into the HTML unescaped. Figures are PNGs
embedded as base64 data URIs. Each chart is followed by a table view of
the same numbers.
"""
from __future__ import annotations

import base64
import html
import io
from typing import Dict, List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .metrics import NA  # noqa: E402

ROLE_LINE = (
    "Synthetic portfolio demonstration, implemented with AI coding agents; independent review pending. "
    "No client data or client work."
)
# Categorical slots 1-4 of the validated reference palette (light mode),
# assigned to methods in fixed order.
METHOD_COLORS = {
    "naive_last_week": "#2a78d6",
    "seasonal_naive": "#eb6834",
    "moving_average_4": "#1baf7a",
    "gradient_boosting": "#eda100",
}
METHOD_STYLES = {
    "naive_last_week": ("-", "o"),
    "seasonal_naive": ("--", "s"),
    "moving_average_4": (":", "^"),
    "gradient_boosting": ("-.", "D"),
}
INK = "#0b0b0b"
INK_2 = "#52514e"
SURFACE = "#fcfcfb"


def _e(value: object) -> str:
    return html.escape(str(value), quote=True)


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "None"
    if value == NA:
        return NA
    if isinstance(value, (bool, np.bool_)):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor=SURFACE, metadata={"Software": None})
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)


def per_store_bar_png(metrics: Mapping, key: str, title: str, ylabel: str) -> str:
    overall = metrics["groups"]["test_overall"]["strata"]
    stores = [s for s in overall if s.startswith("store_")]
    methods = metrics["methods"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), facecolor=SURFACE)
    width = 0.8 / len(methods)
    x = np.arange(len(stores))
    for i, m in enumerate(methods):
        vals = [overall[s]["own_coverage"][m]["metrics"][key] for s in stores]
        heights = [np.nan if v == NA else float(v) for v in vals]
        ax.bar(x + (i - (len(methods) - 1) / 2) * width, heights, width * 0.92,
               color=METHOD_COLORS.get(m, INK_2), label=m, edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(x, [s.replace("store_", "Store ") for s in stores])
    ax.set_ylabel(ylabel, color=INK_2)
    ax.set_title(title, color=INK, loc="left", fontsize=11)
    _style(ax)
    ax.legend(frameon=False, fontsize=8, ncols=2, loc="upper left", bbox_to_anchor=(0, -0.12))
    fig.tight_layout()
    return _png(fig)


def series_line_png(forecasts: pd.DataFrame, partition: str, store: str, sku: str, methods: Sequence[str]) -> str:
    sub = forecasts[(forecasts["partition"] == partition) & (forecasts["store"] == store) & (forecasts["sku"] == sku)]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), facecolor=SURFACE)
    act = sub[sub["method"] == methods[0]].sort_values("target_week")
    ax.plot(act["target_week"], act["actual"], color=INK, linewidth=2, marker="o", markersize=5, label="actual (observed)")
    for m in methods:
        s = sub[sub["method"] == m].sort_values("target_week")
        ls, mk = METHOD_STYLES.get(m, ("-", "o"))
        ax.plot(s["target_week"], s["prediction"], color=METHOD_COLORS.get(m, INK_2), linestyle=ls,
                marker=mk, markersize=4, linewidth=1.6, label=m)
    ax.set_xlabel("target week", color=INK_2)
    ax.set_ylabel("units", color=INK_2)
    ax.set_title(f"Store {store}, {sku}: actual vs one-week-ahead predictions (test weeks)", color=INK, loc="left", fontsize=11)
    _style(ax)
    ax.legend(frameon=False, fontsize=8, ncols=3, loc="upper left", bbox_to_anchor=(0, -0.16))
    fig.tight_layout()
    return _png(fig)


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    head = "".join(f"<th>{_e(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{_e(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def stratum_rows(metrics: Mapping) -> List[List[object]]:
    rows = []
    for gname, group in metrics["groups"].items():
        for sname, rep in group["strata"].items():
            for m in metrics["methods"]:
                own = rep["own_coverage"][m]
                mm = own["metrics"]
                rows.append([gname, sname, m, own["eligible_rows"], own["scored_rows"], own["unavailable_rows"],
                             fmt(own["coverage"]), fmt(mm["MAE"]), fmt(mm["WAPE"]), fmt(mm["signed_bias"])])
    return rows


def common_support_rows(metrics: Mapping) -> List[List[object]]:
    rows = []
    for gname, group in metrics["groups"].items():
        for sname, rep in group["strata"].items():
            cs = rep["common_support"]
            for m in metrics["methods"]:
                mm = cs["metrics"][m]
                rows.append([gname, sname, m, cs["eligible_rows"], cs["common_rows"],
                             fmt(mm["MAE"]), fmt(mm["WAPE"]), fmt(mm["signed_bias"])])
    return rows


def markdown_results_table(metrics: Mapping) -> str:
    """Own-coverage results table (markdown) for README."""
    lines = [
        "| partition | stratum | method | eligible | scored | unavailable | MAE | WAPE % | signed bias |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in stratum_rows(metrics):
        g, s, m, e, sc, u, _cov, mae, wape, bias = r
        lines.append(f"| {g} | {s} | {m} | {e} | {sc} | {u} | {mae} | {wape} | {bias} |")
    return "\n".join(lines) + "\n"


def render_report(
    metrics: Mapping,
    selection_log: Mapping,
    forecasts: pd.DataFrame,
    meta: Mapping[str, object],
) -> str:
    methods = metrics["methods"]
    shift_skus = list(metrics["shifted_skus"])
    rep_sku = shift_skus[0]  # declared rule: first planted-shift SKU (the +60% step), chosen before scoring
    figs = [
        ("Test MAE by store (own coverage, row-weighted within store)",
         per_store_bar_png(metrics, "MAE", "Test MAE by store, lower is better", "MAE (units)")),
        ("Test WAPE by store (own coverage)",
         per_store_bar_png(metrics, "WAPE", "Test WAPE by store, lower is better", "WAPE (%)")),
        (f"Store A, {rep_sku} (planted +shift SKU), test weeks",
         series_line_png(forecasts, "test_AB", "A", rep_sku, methods)),
        (f"Store C (unseen during fitting), {rep_sku}, test weeks",
         series_line_png(forecasts, "test_C", "C", rep_sku, methods)),
    ]
    sel = selection_log.get("selected")
    grid_rows = [
        [c["index"], ", ".join(f"{k}={v}" for k, v in c["config"].items()), c["status"],
         fmt(c.get("validation_MAE", NA)) if c["status"] == "succeeded" else NA, c.get("reason", "")]
        for c in selection_log.get("configurations", [])
    ]
    base_rows = [
        [name, fmt(b["metrics"]["N"]), fmt(b["metrics"]["MAE"]), fmt(b["metrics"]["WAPE"]), fmt(b["metrics"]["signed_bias"])]
        for name, b in selection_log.get("baselines_on_eligible_key_set", {}).items()
    ]
    macro = metrics["groups"]["test_overall"].get("macro_store_average", {})
    macro_rows = [[m, fmt(v["MAE"]), fmt(v["WAPE"]), fmt(v["signed_bias"]), ", ".join(v["MAE_stores_included"])] for m, v in macro.items()]
    meta_rows = [[k, v] for k, v in meta.items()]
    clip = metrics["gbm_non_negative_clipping"]

    parts = [
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        f"<title>{_e('Tabular demand forecast - step 2 test report')}</title>",
        "<style>body{font-family:system-ui,sans-serif;background:#fcfcfb;color:#0b0b0b;max-width:980px;margin:2em auto;padding:0 1em;line-height:1.45}"
        "table{border-collapse:collapse;font-size:13px;margin:.5em 0 1.5em}td,th{border:1px solid #d8d7d2;padding:3px 6px;text-align:right}"
        "td:first-child,th:first-child,td:nth-child(2),td:nth-child(3){text-align:left}.muted{color:#52514e}img{max-width:100%}</style>",
        "</head><body>",
        f"<h1>{_e('Weekly demand forecast: locked test report')}</h1>",
        f"<p><strong>{_e(ROLE_LINE)}</strong></p>",
        f"<p>{_e('Every number below is measured ' + metrics['label'] + '. No significance test, interval, or production-accuracy claim is made.')}</p>",
        f"<h2>{_e('Provenance')}</h2>",
        _table(["field", "value"], meta_rows),
        f"<h2>{_e('Model selection (validation, stores A/B, weeks 79-91)')}</h2>",
        f"<p>{_e('Status: ' + str(selection_log.get('status')) + ('; selected ' + str(sel) if sel else '; reason: ' + str(selection_log.get('reason'))))}</p>",
        f"<p>{_e('Eligible key set: ' + str(selection_log.get('eligible_key_set')))}</p>",
        f"<h3>{_e('Baselines on the same eligible key set')}</h3>",
        _table(["method", "N", "MAE", "WAPE %", "signed bias"], base_rows),
        f"<h3>{_e('All grid configurations (failures included)')}</h3>",
        _table(["index", "config", "status", "validation MAE", "failure reason"], grid_rows),
        f"<h2>{_e('Locked test, scored once (weeks 92-104)')}</h2>",
        f"<p>{_e('GBM non-negative clipping: ' + str(clip))}</p>",
    ]
    for caption, b64 in figs:
        parts.append(f"<figure><img alt=\"{_e(caption)}\" src=\"data:image/png;base64,{_e(b64)}\"><figcaption class=\"muted\">{_e(caption)}</figcaption></figure>")
    parts += [
        f"<h3>{_e('Own-coverage metrics (each method on the rows it could score)')}</h3>",
        _table(["partition", "stratum", "method", "eligible", "scored", "unavailable", "coverage", "MAE", "WAPE %", "signed bias"], stratum_rows(metrics)),
        f"<h3>{_e('Common-support metrics (rows scored by ALL four methods) - reported separately')}</h3>",
        _table(["partition", "stratum", "method", "eligible", "common rows", "MAE", "WAPE %", "signed bias"], common_support_rows(metrics)),
        f"<h3>{_e('Macro store average (unweighted mean of A, B, C; NOT the row-weighted overall)')}</h3>",
        _table(["method", "MAE", "WAPE %", "signed bias", "stores included"], macro_rows),
        "</body></html>",
    ]
    return "\n".join(parts)
