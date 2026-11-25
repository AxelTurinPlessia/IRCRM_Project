"""Run Questions 1(a) through 2(c): python3 main.py."""

import csv
from math import log
from pathlib import Path

from src.bootstrap import DiscountCurve, bootstrap, reprice_quote
from src.curve_plots import plot_curve_comparison, plot_curves
from src.market_data import SPOT_DATE, act365, annual_payment_dates, load_quotes
from src.smoothing import calibrate_curve


ROOT = Path(__file__).resolve().parent


def main() -> None:
    quotes = load_quotes(ROOT / "data" / "table1_sofr_quotes.csv")
    curve = bootstrap(quotes)
    output = ROOT / "results" / "question_1a" / "discount_curve.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "discount_factor", "source"])
        for maturity, discount in sorted(curve.discounts.items()):
            writer.writerow([maturity.isoformat(), format(discount, ".17g"),
                             curve.sources[maturity]])
    print(f"Spot date: {SPOT_DATE}")
    print(f"Bootstrapped {len(curve.discounts) - 1} future nodes plus the spot node.")
    print(f"Saved: {output.relative_to(ROOT)}")

    output = ROOT / "results" / "question_1b" / "repricing.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    max_abs_error_bp = 0.0
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["instrument", "market_quote", "repriced_quote", "quote_error_bp"])
        for quote in quotes:
            repriced = reprice_quote(curve, quote)
            # One hundredth of a quotation point is one basis point.
            error_bp = 100.0 * (repriced - quote.quote)
            max_abs_error_bp = max(max_abs_error_bp, abs(error_bp))
            displayed_error_bp = round(error_bp, 4) or 0.0
            writer.writerow([quote.instrument, f"{quote.quote:.4f}",
                             f"{repriced:.4f}", f"{displayed_error_bp:.4f}"])
    print(f"Repriced {len(quotes)} quotes. Maximum absolute error: {max_abs_error_bp:.3g} bp.")
    print(f"Saved: {output.relative_to(ROOT)}")

    # Table 2 uses the overnight maturity and annual swap dates at 1Y and quoted tenors.
    maturities = annual_payment_dates(curve.spot, 1)
    maturities.extend(quote.maturity for quote in quotes if quote.kind in ("ON", "OIS"))
    output = ROOT / "results" / "question_1c" / "zero_coupon_bonds.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    times, yields = [], []
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["maturity", "discount_factor", "time_act365", "spot_rate_percent"])
        for maturity in sorted(maturities):
            discount = curve.discounts[maturity]
            time_to_maturity = act365(curve.spot, maturity)
            spot_rate_percent = -100.0 * log(discount) / time_to_maturity
            times.append(time_to_maturity)
            yields.append(spot_rate_percent / 100.0)
            writer.writerow([maturity.isoformat(), discount, time_to_maturity, spot_rate_percent])
    print(f"Saved: {output.relative_to(ROOT)}")

    output = ROOT / "figures" / "question_1d_curves.pdf"
    plot_curves(curve, maturities, output)
    print(f"Saved: {output.relative_to(ROOT)}")

    output = ROOT / "results" / "question_2a" / "calibration.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fitted_curves = {}
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "beta_0", "beta_1", "beta_2", "beta_3",
                         "a_1", "a_2", "yield_rmse_bp"])
        for name, decay_count in (("NS", 1), ("NSS", 2)):
            fitted, rmse_bp = calibrate_curve(times, yields, decay_count)
            fitted_curves[name] = fitted
            beta_3 = fitted.betas[3] if decay_count == 2 else ""
            a_2 = fitted.decay_rates[1] if decay_count == 2 else ""
            writer.writerow([name, *fitted.betas[:3], beta_3, fitted.decay_rates[0], a_2, rmse_bp])
            print(f"{name} yield RMSE: {rmse_bp:.4f} bp")
    print(f"Saved: {output.relative_to(ROOT)}")

    repricing = {}
    for name, fitted in fitted_curves.items():
        # Evaluate the fitted curve at every reset and payment date used in 1(b).
        discounts = {date: float(fitted.discount(act365(curve.spot, date)))
                     for date in curve.discounts}
        dated_curve = DiscountCurve(curve.spot, discounts)
        repricing[name] = []
        for quote in quotes:
            repriced = reprice_quote(dated_curve, quote)
            repricing[name].append((repriced, 100.0 * (repriced - quote.quote)))
        worst_quote, (_, worst_error) = max(zip(quotes, repricing[name]),
                                             key=lambda item: abs(item[1][1]))
        print(f"{name} largest absolute quote error: {worst_quote.instrument} "
              f"({worst_error:+.4f} bp signed)")

    output = ROOT / "results" / "question_2b" / "repricing.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["instrument", "market_quote", "ns_quote", "ns_error_bp",
                         "nss_quote", "nss_error_bp"])
        for quote, ns, nss in zip(quotes, repricing["NS"], repricing["NSS"]):
            writer.writerow([quote.instrument, f"{quote.quote:.4f}",
                             *(f"{value:.4f}" for value in (*ns, *nss))])
    print(f"Saved: {output.relative_to(ROOT)}")

    output = ROOT / "figures" / "question_2c_curve_comparison.pdf"
    plot_curve_comparison(curve, maturities, fitted_curves, output)
    print(f"Saved: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
