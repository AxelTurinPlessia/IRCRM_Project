"""Table 1 quotes and the date conventions of Question 1(a)."""

from __future__ import annotations

import calendar
import csv
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional


TRADE_DATE = date(2026, 8, 17)
SPOT_DATE = date(2026, 8, 19)
MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


def act360(start: date, end: date) -> float:
    """Actual calendar-day accrual fraction (end date excluded)."""
    return (end - start).days / 360.0


def act365(start: date, end: date) -> float:
    """Actual calendar days divided by 365, including in leap years."""
    return (end - start).days / 365.0


def third_wednesday(year: int, month: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(calendar.WEDNESDAY - first.weekday()) % 7 + 14)


def annual_payment_dates(spot: date, years: int) -> list[date]:
    """Reproduce the annual schedule on slide 121.

    Roll each original August 19 anniversary forward over weekends. This
    matches the supplied schedule; it is not a general US holiday calendar.
    Compute each anniversary from spot, avoiding cumulative date drift.
    """
    dates = []
    for year in range(spot.year + 1, spot.year + years + 1):
        payment = date(year, spot.month, spot.day)
        while payment.weekday() >= 5:
            payment += timedelta(days=1)
        dates.append(payment)
    return dates


@dataclass(frozen=True)
class MarketQuote:
    instrument: str
    kind: str
    quote: float
    start: date
    maturity: date
    tenor_years: Optional[int] = None

    @property
    def rate(self) -> float:
        """Decimal annual rate, converting futures prices as (100-q)/100."""
        if self.kind in ("SR1", "SR3"):
            return (100.0 - self.quote) / 100.0
        return self.quote / 100.0


def load_quotes(path: Path, spot: date = SPOT_DATE) -> list[MarketQuote]:
    """Read the existing CSV without changing the original quotes.

    SR1 Aug-26 is deliberately shortened to [spot, 1 Sep), as specified
    by the project. SR3 labels identify the reference quarter's START.
    """
    quotes = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = row["instrument"]
            kind = "ON" if name == "SOFR ON" else name.split()[0]
            maturity = date.fromisoformat(row["maturity"])
            tenor = None
            if kind in ("SR1", "SR3"):
                month_name, short_year = name.split()[1].split("-")
                year, month = 2000 + int(short_year), MONTHS.index(month_name) + 1
                if kind == "SR1":
                    start = max(spot, date(year, month, 1))
                else:
                    start = third_wednesday(year, month)
            elif kind in ("ON", "OIS"):
                start = spot
                if kind == "OIS":
                    tenor = int(name.split()[1].removesuffix("Y"))
            else:
                raise ValueError(f"Unsupported instrument: {name}")
            if maturity <= start:
                raise ValueError(f"Invalid accrual period for {name}")
            quotes.append(MarketQuote(name, kind, float(row["quote"]), start, maturity, tenor))
    return quotes
