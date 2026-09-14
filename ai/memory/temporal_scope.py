"""Does a question ask about a particular time, and does a memory fall in it?

Recall scores memories by similarity and recency, neither of which understands
"on the 3rd of March last year". Asked that, Alice retrieved her most relevant
memories — none of them from that date — and presented them under "Here is what
I have saved in memory", which reads as an answer to the question asked. Telling
someone what they said on a day they did not say it is the worst thing a memory
system can do, because it is indistinguishable from remembering.

This decides two narrow things:

* :func:`requested_period` — whether a question names a period at all, and which.
* :func:`any_item_within` — whether any recalled memory actually falls in it.

It deliberately recognises less than a full temporal parser. A question it does
not understand yields no period, and the caller behaves exactly as before; the
guard only fires where the constraint is unambiguous.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

_MONTH_ALTERNATION = "|".join(sorted(MONTHS, key=len, reverse=True))

# "on the 3rd of March", "on March 3rd", "March 3 2024"
_DAY_MONTH = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH_ALTERNATION})\b",
    re.IGNORECASE,
)
_MONTH_DAY = re.compile(
    rf"\b({_MONTH_ALTERNATION})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b",
    re.IGNORECASE,
)
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_BARE_MONTH = re.compile(rf"\b({_MONTH_ALTERNATION})\b", re.IGNORECASE)
_EXPLICIT_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_LAST_YEAR = re.compile(r"\blast\s+year\b", re.IGNORECASE)
_THIS_YEAR = re.compile(r"\bthis\s+year\b", re.IGNORECASE)
_YESTERDAY = re.compile(r"\byesterday\b", re.IGNORECASE)
_LAST_WEEK = re.compile(r"\blast\s+week\b", re.IGNORECASE)
_LAST_MONTH = re.compile(r"\blast\s+month\b", re.IGNORECASE)

# Only questions that look back over the conversation are in scope. "what is the
# weather in March" is about the world, not about what was said.
_RECOLLECTION = re.compile(
    r"\b(did i|did you|i say|you say|we (say|discuss|talk)|i tell|you tell|"
    r"told (you|me)|talk(ed)? about|discuss(ed)?|remember|said)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Period:
    """A closed date range, with the phrase that produced it."""

    start: date
    end: date
    phrase: str

    def contains(self, moment: date) -> bool:
        return self.start <= moment <= self.end

    def describe(self) -> str:
        if self.start == self.end:
            return self.start.strftime("%-d %B %Y") if hasattr(self.start, "strftime") else str(self.start)
        if (self.start.year, self.start.month) == (self.end.year, self.end.month):
            return self.start.strftime("%B %Y")
        if self.start.year == self.end.year and self.start.month == 1 and self.end.month == 12:
            return str(self.start.year)
        return f"{self.start.isoformat()} to {self.end.isoformat()}"


def _month_range(year: int, month: int) -> Tuple[date, date]:
    start = date(year, month, 1)
    end = date(year + 1, 1, 1) - timedelta(days=1) if month == 12 else date(year, month + 1, 1) - timedelta(days=1)
    return start, end


def _safe_date(year: int, month: int, day: int) -> Optional[date]:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def requested_period(question: str, *, today: Optional[date] = None) -> Optional[Period]:
    """The period a recollection question asks about, or None.

    Returns None for anything that is not clearly asking what was said during a
    specific stretch of time, so an unrecognised phrasing changes no behaviour.
    """
    text = str(question or "")
    if not text.strip() or not _RECOLLECTION.search(text):
        return None

    today = today or datetime.now().date()

    if iso := _ISO_DATE.search(text):
        year, month, day = (int(g) for g in iso.groups())
        if (moment := _safe_date(year, month, day)) is not None:
            return Period(moment, moment, iso.group(0))

    if _YESTERDAY.search(text):
        moment = today - timedelta(days=1)
        return Period(moment, moment, "yesterday")

    if _LAST_WEEK.search(text):
        return Period(today - timedelta(days=14), today - timedelta(days=7), "last week")

    if _LAST_MONTH.search(text):
        first_of_this_month = today.replace(day=1)
        end = first_of_this_month - timedelta(days=1)
        return Period(end.replace(day=1), end, "last month")

    # A year named outright wins over "last year".
    year: Optional[int] = None
    if explicit := _EXPLICIT_YEAR.search(text):
        year = int(explicit.group(0))
    elif _LAST_YEAR.search(text):
        year = today.year - 1
    elif _THIS_YEAR.search(text):
        year = today.year

    day_month = _DAY_MONTH.search(text)
    month_day = _MONTH_DAY.search(text)
    if day_month or month_day:
        if day_month:
            day, month_name = int(day_month.group(1)), day_month.group(2)
        else:
            month_name, day = month_day.group(1), int(month_day.group(2))
        month = MONTHS[month_name.lower()]
        if (moment := _safe_date(year or today.year, month, day)) is not None:
            return Period(moment, moment, (day_month or month_day).group(0))

    if bare_month := _BARE_MONTH.search(text):
        month = MONTHS[bare_month.group(1).lower()]
        start, end = _month_range(year or today.year, month)
        return Period(start, end, bare_month.group(0))

    if year is not None:
        return Period(date(year, 1, 1), date(year, 12, 31), str(year))

    return None


def _item_date(item: Dict[str, Any]) -> Optional[date]:
    raw = str(item.get("timestamp") or "").strip()
    if not raw:
        context = item.get("context")
        if isinstance(context, dict):
            raw = str(context.get("timestamp") or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except ValueError:
        match = _ISO_DATE.search(raw)
        if match:
            year, month, day = (int(g) for g in match.groups())
            return _safe_date(year, month, day)
    return None


def any_item_within(items: List[Dict[str, Any]], period: Period) -> bool:
    """Whether any recalled memory carries a timestamp inside ``period``.

    An item with no usable timestamp cannot support a claim about when it
    happened, so it does not count.
    """
    for item in items or []:
        moment = _item_date(item)
        if moment is not None and period.contains(moment):
            return True
    return False


def items_within(items: List[Dict[str, Any]], period: Period) -> List[Dict[str, Any]]:
    """Only the recalled memories that fall inside ``period``."""
    kept = []
    for item in items or []:
        moment = _item_date(item)
        if moment is not None and period.contains(moment):
            kept.append(item)
    return kept
