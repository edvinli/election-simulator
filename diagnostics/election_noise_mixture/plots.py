"""Plots for the ElectionNoise coalition-seat mixture diagnostic.

Run with an environment that has matplotlib; it only reads the .npz run artifacts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

HERE = Path(__file__).resolve().parent
RUNS = HERE / "_runs"
PLOTS = HERE / "plots"
MAJORITY = 175

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8d8b85"
GRID = "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
THRESHOLD_INK = "#e34948"

COALITIONS = {"C+S+MP": ("C", "S", "MP"), "S+V+MP": ("S", "V", "MP")}


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK2,
            "text.color": INK,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
        }
    )


def coalition_seats(npz, parties):
    p8 = [str(x) for x in npz["parties_8"]]
    cols = [p8.index(p) for p in parties]
    return npz["seats_matrix"][:, cols].sum(axis=1).astype(np.int64)


def density(x: np.ndarray, lo: int, hi: int) -> np.ndarray:
    counts = np.bincount(np.clip(x, lo, hi) - lo, minlength=hi - lo + 1)
    return counts / counts.sum()


def mark_threshold(ax, ymax, label=True):
    ax.axvline(MAJORITY, color=THRESHOLD_INK, lw=1.4, ls=(0, (4, 3)), zorder=5)
    if label:
        ax.annotate(
            "175 seats\n(majority)",
            xy=(MAJORITY, ymax),
            xytext=(4, -2),
            textcoords="offset points",
            va="top",
            ha="left",
            fontsize=8,
            color=THRESHOLD_INK,
        )


def plot_overall(prod, pre, label, parties) -> None:
    s_fin = coalition_seats(prod, parties)
    s_pre = coalition_seats(pre, parties)
    lo = int(min(s_fin.min(), s_pre.min()))
    hi = int(max(s_fin.max(), s_pre.max()))
    grid = np.arange(lo, hi + 1)
    d_pre = density(s_pre, lo, hi)
    d_fin = density(s_fin, lo, hi)

    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=200)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.fill_between(grid, d_pre, step="mid", color=SERIES[0], alpha=0.16, zorder=2)
    ax.step(grid, d_pre, where="mid", color=SERIES[0], lw=2.0, zorder=3,
            label=f"Pre-ElectionNoise  (P≥175 = {np.mean(s_pre >= MAJORITY):.2%})")
    ax.fill_between(grid, d_fin, step="mid", color=SERIES[1], alpha=0.16, zorder=2)
    ax.step(grid, d_fin, where="mid", color=SERIES[1], lw=2.0, zorder=4,
            label=f"Final (post-ElectionNoise)  (P≥175 = {np.mean(s_fin >= MAJORITY):.2%})")

    ymax = max(d_pre.max(), d_fin.max()) * 1.18
    ax.set_ylim(0, ymax)
    ax.set_xlim(lo - 0.5, hi + 0.5)
    mark_threshold(ax, ymax)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Coalition seats (349-seat Riksdag)")
    ax.set_ylabel("Share of 100,000 simulations")
    ax.set_title(f"{label} — seat distribution before and after ElectionNoise", loc="left",
                 fontweight="bold")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    fig.text(0.005, 0.005,
             "Both curves are seat distributions: identical OpinionState + Dynamics draws, "
             "identical geography and exact mandate allocator.",
             fontsize=7.2, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(PLOTS / f"overall_{label.replace('+', '_')}.png")
    plt.close(fig)


def plot_by_year(prod, label, parties) -> None:
    s = coalition_seats(prod, parties)
    ryear = prod["residual_year"]
    years = sorted(set(int(y) for y in prod["training_years"]))
    lo, hi = int(s.min()), int(s.max())
    grid = np.arange(lo, hi + 1)
    d_all = density(s, lo, hi)

    fig, axes = plt.subplots(2, 3, figsize=(11.6, 6.2), dpi=200, sharex=True, sharey=True)
    ymax = 0.0
    dens = {}
    for y in years:
        dens[y] = density(s[ryear == y], lo, hi)
        ymax = max(ymax, dens[y].max())
    ymax *= 1.22

    for ax, y, colour in zip(axes.ravel(), years, SERIES):
        sub = s[ryear == y]
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
        ax.step(grid, d_all, where="mid", color=MUTED, lw=1.2, zorder=2,
                label="All draws pooled")
        ax.fill_between(grid, dens[y], step="mid", color=colour, alpha=0.20, zorder=3)
        ax.step(grid, dens[y], where="mid", color=colour, lw=2.0, zorder=4,
                label=f"Residual year {y}")
        ax.axvline(MAJORITY, color=THRESHOLD_INK, lw=1.2, ls=(0, (4, 3)), zorder=5)
        ax.set_ylim(0, ymax)
        ax.set_xlim(lo - 0.5, hi + 0.5)
        ax.set_title(
            f"{y}   n={sub.size:,}   median {int(np.median(sub))}   "
            f"P≥175 = {np.mean(sub >= MAJORITY):.2%}",
            loc="left", fontsize=9,
        )
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))

    axes[0, 0].annotate("175", xy=(MAJORITY, ymax), xytext=(3, -2),
                        textcoords="offset points", va="top", ha="left",
                        fontsize=8, color=THRESHOLD_INK)
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(handles[:1], ["All draws pooled"], frameon=False, loc="upper left",
                      fontsize=8)
    for ax in axes[1, :]:
        ax.set_xlabel("Coalition seats")
    for ax in axes[:, 0]:
        ax.set_ylabel("Share of draws in facet")
    fig.suptitle(
        f"{label} — final seat distribution conditional on the sampled ElectionNoise residual year",
        x=0.006, ha="left", fontweight="bold", fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(PLOTS / f"by_residual_year_{label.replace('+', '_')}.png")
    plt.close(fig)


def plot_year_overlay(prod, label, parties) -> None:
    """Single-panel overlay: pooled distribution plus each year's conditional curve."""
    s = coalition_seats(prod, parties)
    ryear = prod["residual_year"]
    years = sorted(set(int(y) for y in prod["training_years"]))
    lo, hi = int(s.min()), int(s.max())
    grid = np.arange(lo, hi + 1)

    fig, ax = plt.subplots(figsize=(9.0, 4.8), dpi=200)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ymax = 0.0
    peaks = []
    for y, colour in zip(years, SERIES):
        d = density(s[ryear == y], lo, hi) / len(years)  # contribution to the pooled mixture
        ymax = max(ymax, d.max())
        ax.step(grid, d, where="mid", color=colour, lw=1.9, zorder=3)
        peaks.append((int(grid[int(np.argmax(d))]), float(d.max()), y, colour))
    # Stagger direct labels so neighbouring modes do not collide.
    peaks.sort()
    last_x = -999
    tier = 0
    for px, py, y, colour in peaks:
        tier = tier + 1 if px - last_x < 5 else 0
        last_x = px
        ax.annotate(str(y), xy=(px, py), xytext=(0, 6 + 11 * tier),
                    textcoords="offset points", ha="center", fontsize=8,
                    color=colour, fontweight="bold", zorder=6)
    d_all = density(s, lo, hi)
    ymax = max(ymax, d_all.max())
    ax.step(grid, d_all, where="mid", color=INK2, lw=2.4, zorder=4, label="Pooled mixture")
    ax.set_ylim(0, ymax * 1.16)
    ax.set_xlim(lo - 0.5, hi + 0.5)
    mark_threshold(ax, ymax * 1.16)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Coalition seats")
    ax.set_ylabel("Share of all 100,000 simulations")
    ax.set_title(
        f"{label} — the pooled distribution decomposed into its six residual-year components",
        loc="left", fontweight="bold",
    )
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    fig.text(0.005, 0.005,
             "Each thin curve is that residual year's component scaled by its 1/6 mixture weight, "
             "so the six curves sum to the pooled mixture. Years are direct-labelled at their mode.",
             fontsize=7.2, color=MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(PLOTS / f"mixture_decomposition_{label.replace('+', '_')}.png")
    plt.close(fig)


def main() -> int:
    style()
    PLOTS.mkdir(parents=True, exist_ok=True)
    prod = np.load(RUNS / "production.npz", allow_pickle=False)
    pre = np.load(RUNS / "prenoise.npz", allow_pickle=False)
    for label, parties in COALITIONS.items():
        plot_overall(prod, pre, label, parties)
        plot_by_year(prod, label, parties)
        plot_year_overlay(prod, label, parties)
    print("wrote", sorted(p.name for p in PLOTS.glob("*.png")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
