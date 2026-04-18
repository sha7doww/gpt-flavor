#!/usr/bin/env python3
"""
Generate figures for the stylometry report.

Produces under analysis/figures/:
  - cosine_heatmap.png       — bucket-to-bucket cosine similarity
  - log_odds_bars.png        — per-model top-15 characteristic n-grams
  - prompt_sensitivity.png   — top-N prompt-sensitive patterns × (model × condition)

Usage:
  python visualize.py        # read from analysis/{stats,stats_by_bucket,cosine_matrix,log_odds_top}.json
"""

import json
from pathlib import Path

from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np


ROOT = Path(__file__).parent.parent
ANA = ROOT / "analysis"
FIG = ANA / "figures"


def configure_cjk_font():
    """Set up per-glyph font fallback: Times New Roman for Latin, a Ming
    serif CJK font for Chinese. matplotlib >=3.6 picks the first font in
    the family list that has a given glyph, so the order is TNR → CJK."""
    cjk_paths = [
        # Linux arphic uming — the ORIGINAL rendering font
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf",
        # macOS (Songti = Ming serif, closest match)
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        # Linux (other fallbacks)
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        # Windows
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    latin_paths = [
        # macOS
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        # Windows
        "C:/Windows/Fonts/times.ttf",
        # Linux TNR metric-compatible
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
    ]

    cjk = next((p for p in cjk_paths if Path(p).exists()), None)
    latin = next((p for p in latin_paths if Path(p).exists()), None)

    family = []
    if latin:
        fm.fontManager.addfont(latin)
        latin_name = fm.FontProperties(fname=latin).get_name()
        family.append(latin_name)
    if cjk:
        fm.fontManager.addfont(cjk)
        cjk_name = fm.FontProperties(fname=cjk).get_name()
        family.append(cjk_name)
    family.append("DejaVu Serif")  # matplotlib-bundled final fallback

    # IMPORTANT: put the specific-font list in font.family directly (not in
    # font.serif with family="serif"). matplotlib only does per-glyph
    # fallback when font.family is a list of concrete font names; using a
    # generic category resolves to a single font and tofu appears instead.
    plt.rcParams["font.family"] = family
    plt.rcParams["axes.unicode_minus"] = False

    print(f"Font chain: {family}")
    if not cjk:
        print("WARN: no CJK font found; Chinese may render as tofu.")
    return family[0] if family else None


def plot_cosine_heatmap():
    path = ANA / "cosine_matrix.json"
    if not path.exists():
        print(f"Skip cosine heatmap: {path} missing")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    groups = data["groups"]
    mat = np.array(data["matrix"])
    n = len(groups)

    fig, ax = plt.subplots(figsize=(max(8, n * 0.6 + 2), max(6, n * 0.5 + 2)))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(groups, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=6, color="black" if mat[i, j] < 0.6 else "white")
    ax.set_title("Char n-gram TF-IDF cosine similarity between buckets\n"
                 "(model × system_prompt_condition)", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    out = FIG / "cosine_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  → {out}")


def plot_log_odds_bars():
    path = ANA / "log_odds_top.json"
    if not path.exists():
        print(f"Skip log-odds bars: {path} missing")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    groups = sorted(data.keys())
    ncols = min(len(groups), 3)
    nrows = (len(groups) + ncols - 1) // ncols
    top_k = 15

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4),
                             squeeze=False)
    for idx, g in enumerate(groups):
        ax = axes[idx // ncols][idx % ncols]
        items = data[g][:top_k]
        labels = [it["ngram"] for it in items][::-1]
        scores = [it["z"] for it in items][::-1]
        ax.barh(range(len(labels)), scores, color="#3b82f6")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(f"{g} — top {top_k} characteristic char-ngrams", fontsize=10)
        ax.set_xlabel("log-odds z-score")
        ax.grid(axis="x", alpha=0.2)
    # hide any unused axes
    for idx in range(len(groups), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.suptitle("Weighted log-odds (Monroe et al. 2008) — most characteristic char n-grams per model",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    out = FIG / "log_odds_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


def load_pattern_meta():
    """Load pattern metadata from patterns/*.yaml, keyed by name. Used for label descriptions."""
    import yaml
    meta = {}
    pdir = ROOT / "patterns"
    if not pdir.exists():
        return meta
    for path in sorted(pdir.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        for p in data.get("patterns", []):
            meta[p["name"]] = {"desc": p.get("desc", p["name"])}
    return meta


def plot_prompt_sensitivity():
    """Show which patterns are most prompt-sensitive via per-(model,condition) heatmap.
    Only covers patterns whose max-min rate across conditions > threshold."""
    path = ANA / "stats_by_bucket.json"
    if not path.exists():
        print(f"Skip prompt sensitivity: {path} missing")
        print("  → run `python analyze.py --by bucket` first")
        return
    stats = json.loads(path.read_text(encoding="utf-8"))
    buckets = sorted(stats.keys())
    pattern_names = list(stats[buckets[0]]["patterns"].keys())
    meta = load_pattern_meta()

    models = sorted({b.split("/")[0] for b in buckets})
    conditions = ["A_empty", "B_listener", "C_poster"]

    # compute per-pattern max-min range of rate across 3 conditions (per model)
    # sensitivity[pattern] = max range over all models
    sensitivity = {}
    for p in pattern_names:
        ranges = []
        for m in models:
            rates = [stats[f"{m}/{c}"]["patterns"][p]["rate"]
                     for c in conditions if f"{m}/{c}" in stats]
            if rates:
                ranges.append(max(rates) - min(rates))
        sensitivity[p] = max(ranges) if ranges else 0

    # top 10 most prompt-sensitive
    top = sorted(sensitivity.items(), key=lambda kv: kv[1], reverse=True)[:10]
    top_names = [p for p, _ in top]
    print(f"  top-10 most prompt-sensitive patterns: {[p for p, _ in top]}")

    display = [f"{p} — {meta.get(p, {}).get('desc', p)}" for p in top_names]
    mat = np.zeros((len(top_names), len(buckets)))
    for j, b in enumerate(buckets):
        for i, p in enumerate(top_names):
            mat[i, j] = stats[b]["patterns"][p]["rate"]
    _render_heatmap(
        mat, display, buckets,
        title=f"Prompt sensitivity — top 10 patterns by max-min rate across conditions",
        out_name="prompt_sensitivity.png",
    )


def _render_heatmap(mat, row_labels, col_labels, title, out_name):
    fig, ax = plt.subplots(figsize=(max(10, len(col_labels) * 0.75 + 3),
                                    max(4, len(row_labels) * 0.4 + 2)))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(row_labels, fontsize=8)
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=6, color="black" if v < 0.55 else "white")
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    out = FIG / out_name
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  → {out}")


def main():
    configure_cjk_font()
    FIG.mkdir(parents=True, exist_ok=True)
    print("Writing figures to", FIG)
    plot_cosine_heatmap()
    plot_log_odds_bars()
    plot_prompt_sensitivity()
    print("Done.")


if __name__ == "__main__":
    main()
