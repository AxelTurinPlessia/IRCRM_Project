"""Question 1(d): rates and discount factors from the Table 2 nodes."""

from datetime import date, timedelta
from math import log
from pathlib import Path

from .bootstrap import DiscountCurve
from .market_data import act365


def plot_curves(curve: DiscountCurve, maturities: list[date], output: Path) -> None:
    from matplotlib.backends.backend_pdf import FigureCanvasPdf
    from matplotlib.figure import Figure
    from matplotlib.ticker import PercentFormatter

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
