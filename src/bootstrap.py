"""Exact bootstrap using FIN-416 slides 118--122.

SR1 and SR3 rates are both used as simple forward-rate proxies, following
the course bootstrap. SR1 actually settles on an arithmetic average:
this proxy is an approximation in addition to ignoring convexity.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import date
from math import exp, isfinite, log

from .market_data import MarketQuote, SPOT_DATE, act360, annual_payment_dates


def linear_value(target: date, left: date, right: date, a: float, b: float) -> float:
    """Interpolate using actual calendar-day distances, without extrapolation."""
    if not left <= target <= right or left == right:
        raise ValueError("Interpolation requires distinct bracketing dates")
    weight = (target - left).days / (right - left).days
    return (1.0 - weight) * a + weight * b


@dataclass
class DiscountCurve:
    spot: date
    discounts: dict[date, float] = field(default_factory=dict)
    sources: dict[date, str] = field(default_factory=dict)

    def add(self, maturity: date, discount: float, source: str) -> None:
        if maturity in self.discounts:
            raise ValueError(f"Duplicate curve node: {maturity}")
        if not isfinite(discount) or discount <= 0:
            raise ValueError(f"Invalid discount factor at {maturity}: {discount}")
        self.discounts[maturity] = discount
        self.sources[maturity] = source

    def discount(self, maturity: date) -> float:
        """Return a bootstrapped node; interpolate log discounts between nodes."""
        if maturity in self.discounts:
            return self.discounts[maturity]
        dates = sorted(self.discounts)
        index = bisect_left(dates, maturity)
        if index == 0 or index == len(dates):
            raise ValueError(f"Date outside the curve: {maturity}")
        left, right = dates[index - 1], dates[index]
        return exp(linear_value(maturity, left, right,
                                log(self.discounts[left]), log(self.discounts[right])))

    def simple_spot_rate(self, maturity: date) -> float:
        return (1.0 / self.discount(maturity) - 1.0) / act360(self.spot, maturity)


def bootstrap(quotes: list[MarketQuote], spot: date = SPOT_DATE) -> DiscountCurve:
    """Construct all reset/payment nodes, including 30 annual OIS payments."""
    curve = DiscountCurve(spot)
    curve.add(spot, 1.0, "spot normalization")

    # 1. Overnight and SR1 segment: P(end) = P(start)/(1 + alpha * rate).
    overnight = [q for q in quotes if q.kind == "ON"]
    if len(overnight) != 1:
        raise ValueError("Expected one overnight quote")
    short_quotes = overnight + sorted((q for q in quotes if q.kind == "SR1"),
                                     key=lambda q: q.maturity)
    for quote in short_quotes:
        discount = curve.discount(quote.start) / (
            1.0 + act360(quote.start, quote.maturity) * quote.rate)
        curve.add(quote.maturity, discount, quote.instrument)

    # 2. First SR3 starts inside the September SR1 period (slide 120).
    futures = sorted((q for q in quotes if q.kind == "SR3"), key=lambda q: q.start)
    if not futures:
        raise ValueError("Expected SR3 quotes")
    first_start = futures[0].start
    curve.add(first_start, curve.discount(first_start), "log-linear discount interpolation")
    for quote in futures:
        discount = curve.discount(quote.start) / (
            1.0 + act360(quote.start, quote.maturity) * quote.rate)
        curve.add(quote.maturity, discount, quote.instrument)

    # 3. First OIS payment: interpolate SIMPLE SPOT RATES, not discounts.
    swaps = sorted((q for q in quotes if q.kind == "OIS"), key=lambda q: q.maturity)
    if not swaps or swaps[0].tenor_years != 2:
        raise ValueError("Expected the quoted OIS strip to start at 2Y")
    payments = annual_payment_dates(spot, max(q.tenor_years for q in swaps))
    for quote in swaps:
        if quote.maturity != payments[quote.tenor_years - 1]:
            raise ValueError(f"OIS maturity differs from course schedule: {quote.instrument}")
    first_payment = payments[0]
    dates = sorted(curve.discounts)
    index = bisect_left(dates, first_payment)
    if index == 0 or index == len(dates):
        raise ValueError("The futures strip must bracket the first OIS payment")
    left, right = dates[index - 1], dates[index]
    rate = linear_value(first_payment, left, right,
                        curve.simple_spot_rate(left), curve.simple_spot_rate(right))
    curve.add(first_payment, 1.0 / (1.0 + act360(spot, first_payment) * rate),
              "linear simple-spot-rate interpolation")

    # 4. Missing annual par swap rates: linear interpolation (slide 121).
    par_rates = {q.maturity: q.rate for q in swaps}
    quoted_dates = sorted(par_rates)
    previous = first_payment
    annuity = act360(spot, previous) * curve.discount(previous)
    for payment in payments[1:]:
        if payment in par_rates:
            rate = par_rates[payment]
            source = next(q.instrument for q in swaps if q.maturity == payment)
        else:
            index = bisect_left(quoted_dates, payment)
            left, right = quoted_dates[index - 1], quoted_dates[index]
            rate = linear_value(payment, left, right, par_rates[left], par_rates[right])
            source = "OIS bootstrap with interpolated par swap rate"

        # K * sum(alpha_i * P_i) = 1 - P_n, solved for P_n (slide 122).
        alpha = act360(previous, payment)
        discount = (1.0 - rate * annuity) / (1.0 + alpha * rate)
        curve.add(payment, discount, source)
        annuity += alpha * discount
        previous = payment
    return curve


def reprice_quote(curve: DiscountCurve, quote: MarketQuote) -> float:
    """Recompute the Table 1 quote in its original quotation units."""
    if quote.kind == "OIS":
        if quote.tenor_years is None:
            raise ValueError(f"Missing OIS tenor for {quote.instrument}")
        previous = quote.start
        annuity = 0.0
        for payment in annual_payment_dates(quote.start, quote.tenor_years):
            annuity += act360(previous, payment) * curve.discount(payment)
            previous = payment
        rate = (curve.discount(quote.start) - curve.discount(quote.maturity)) / annuity
    else:
        rate = (curve.discount(quote.start) / curve.discount(quote.maturity) - 1.0) / (
            act360(quote.start, quote.maturity)
        )

    if quote.kind in ("SR1", "SR3"):
        return 100.0 * (1.0 - rate)
    return 100.0 * rate
