"""Discount and rate plots for the bootstrapped and smoothed curves."""

from datetime import date, timedelta
from math import log
from pathlib import Path

from .bootstrap import DiscountCurve
from .market_data import act365
from .smoothing import SmoothCurve


def _bootstrap_plot_data(curve: DiscountCurve, maturities: list[date]):
    # Only Table 2 maturities and the spot normalization enter the interpolation.
    nodes = [curve.spot, *sorted(maturities)]
    node_times = [act365(curve.spot, node) for node in nodes]
    node_discounts = [curve.discounts[node] for node in nodes]
    table_curve = DiscountCurve(curve.spot, dict(zip(nodes, node_discounts)))

    daily_times = [day / 365.0 for day in range(1, 365 * 30 + 1)]
    daily_discounts = [table_curve.discount(curve.spot + timedelta(days=day))
                       for day in range(1, len(daily_times) + 1)]
    spot_rates = [-log(discount) / time
                  for discount, time in zip(daily_discounts, daily_times)]

    # The forward rate is the negative slope of log P within each segment.
    log_discounts = [log(discount) for discount in node_discounts]
    forward_rates = [(left - right) / (end - start)
                     for left, right, start, end in zip(
                         log_discounts, log_discounts[1:], node_times, node_times[1:])]
    return node_times, node_discounts, daily_times, daily_discounts, spot_rates, forward_rates


def plot_curves(curve: DiscountCurve, maturities: list[date], output: Path) -> None:
    from matplotlib.backends.backend_pdf import FigureCanvasPdf
    from matplotlib.figure import Figure
    from matplotlib.ticker import PercentFormatter

    node_times, node_discounts, daily_times, daily_discounts, spot_rates, forward_rates = (
        _bootstrap_plot_data(curve, maturities)
    )

    fig = Figure(figsize=(7.2, 5.6), layout="constrained")
    FigureCanvasPdf(fig)
    discount_ax, rate_ax = fig.subplots(2, 1, sharex=True)
    discount_color, forward_color, marker_color = "#2058a4", "#ad4f13", "#222222"

    # Include the final Table 2 node so all crosses lie on the displayed line.
    discount_ax.plot([0.0, *daily_times, node_times[-1]],
                     [1.0, *daily_discounts, node_discounts[-1]],
                     color=discount_color, linewidth=1.7, label="Log-linear curve")
    discount_ax.scatter(node_times[1:], node_discounts[1:], marker="x", s=35,
                        color=marker_color, linewidths=1.2, zorder=3,
                        label="Table 2 discount factors")
    discount_ax.set_title("Discount curve", loc="left", fontweight="bold")
    discount_ax.set_ylabel(r"Discount factor $P(t_0,T)$")
    discount_ax.legend(loc="upper right", frameon=False)

    rate_ax.plot(daily_times, spot_rates, color=discount_color, linewidth=1.7,
                 label="Spot rate")
    rate_ax.stairs(forward_rates, [*node_times[:-1], daily_times[-1]],
                   baseline=None, color=forward_color, linestyle="--",
                   linewidth=1.6, label="Instantaneous forward rate")
    rate_ax.set_title("Continuously compounded rates", loc="left", fontweight="bold")
    rate_ax.set_ylabel("Annual rate")
    rate_ax.set_xlabel("Time to maturity (ACT/365 years)")
    rate_ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
    rate_ax.legend(loc="lower right", frameon=False)
    rate_ax.set_xticks(range(0, 31, 5))
    rate_ax.set_xlim(-0.25, node_times[-1] + 0.3)

    for ax in (discount_ax, rate_ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e2e5e9", linewidth=0.7)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)


def plot_curve_comparison(curve: DiscountCurve, maturities: list[date],
                          fitted_curves: dict[str, SmoothCurve], output: Path) -> None:
    from matplotlib.backends.backend_pdf import FigureCanvasPdf
    from matplotlib.figure import Figure
    from matplotlib.ticker import PercentFormatter

    node_times, _, daily_times, _, spot_rates, forward_rates = _bootstrap_plot_data(curve, maturities)
    fig = Figure(figsize=(7.2, 5.6), layout="constrained")
    FigureCanvasPdf(fig)
    spot_ax, forward_ax = fig.subplots(2, 1, sharex=True)

    bootstrap_style = dict(color="#424a53", linestyle="--", linewidth=1.5, label="Bootstrap")
    spot_ax.plot(daily_times, spot_rates, **bootstrap_style)
    forward_ax.stairs(forward_rates, [*node_times[:-1], daily_times[-1]],
                      baseline=None, **bootstrap_style)
    for name, color, linestyle in (("NS", "#ad4f13", ":"), ("NSS", "#2058a4", "-")):
        fitted = fitted_curves[name]
        style = dict(color=color, linestyle=linestyle, linewidth=1.8, label=name)
        spot_ax.plot(daily_times, fitted.spot_rate(daily_times), **style)
        forward_ax.plot(daily_times, fitted.forward_rate(daily_times), **style)

    spot_ax.set_title("Spot rates", loc="left", fontweight="bold")
    forward_ax.set_title("Instantaneous forward rates", loc="left", fontweight="bold")
    spot_ax.legend(loc="upper left", ncols=3, frameon=False)
    forward_ax.set_xlabel("Time to maturity (ACT/365 years)")
    forward_ax.set_xticks(range(0, 31, 5))
    forward_ax.set_xlim(-0.25, 30.25)
    for ax in (spot_ax, forward_ax):
        ax.set_ylabel("Annual rate")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e2e5e9", linewidth=0.7)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
