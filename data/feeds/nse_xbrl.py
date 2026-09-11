"""Reading an Ind-AS XBRL results document — MASTER_PLAN §9.

Split from `nse_results`, which parses the filings *list*. Two formats from two
endpoints: one is a JSON index of what was filed and when, which is an earnings
calendar on its own, and this is the accounting document that turns a filing
into numbers.

**The context trap.** Every fact appears several times under different
contexts, and those contexts declare identical start and end dates — in a Q3
filing both `OneD` and `FourD` claim 2024-10-01 to 2024-12-31, but one holds
the quarter and the other the nine-month cumulative. Selecting by declared
period returns year-to-date figures, overstating revenue roughly threefold with
nothing to indicate anything went wrong. Measured across independent filings:
2.87, 2.89, 2.78, 2.17 — every one the shape of a cumulative against its
quarter.

A prefix match is not enough either: `OneReportableSegmentRevenue01D` is a real
context holding one business segment's revenue. Facts are selected structurally
— a quarter context carrying no dimension and spanning a duration — and, where
a document declares nothing at all, by the narrow rule in
`UNDECLARED_QUARTER_CONTEXT`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from lxml import etree

__all__ = [
    "FUNDAMENTAL_FACTS",
    "IMPLAUSIBLE_ABOVE",
    "QUARTER_CONTEXT",
    "UNDECLARED_QUARTER_CONTEXT",
    "ResultsFormatError",
    "XbrlFacts",
    "parse_xbrl",
    "xbrl_is_plausible",
]

logger = logging.getLogger(__name__)


#: Context id prefix holding the reporting period itself.
#:
#: The Ind-AS taxonomy numbers its periods: `One` is the current quarter, `Four`
#: the year-to-date. They are *not* distinguishable by their declared dates —
#: see the module docstring — so this prefix is the only reliable selector.
QUARTER_CONTEXT = "One"

#: The cumulative context, named so the guard below can say what it rejected.
CUMULATIVE_CONTEXT = "Four"

#: The quarter context of a document that declares no contexts at all.
#:
#: Roughly thirty-one thousand NSE filings reference `OneD` and `FourD` without
#: declaring either, which the spec forbids and which leaves the structural
#: test unsatisfiable: there is no declaration to inspect. Every filing before
#: 2022 was refused on this, correctly and uselessly. Measured across two dozen
#: of them, the only undeclared references their facts use are those two and
#: nothing dimensioned — so an undeclared reference is accepted only when it is
#: exactly this id. The cumulative one is still refused.
UNDECLARED_QUARTER_CONTEXT = "OneD"

#: Facts worth extracting, mapped to the column they become.
#:
#: Deliberately small. Every one of these appears in the standard Ind-AS
#: statement of profit and loss, so a name missing one is a real gap rather
#: than a taxonomy variant, and there is no temptation to infer it.
FUNDAMENTAL_FACTS: dict[str, str] = {
    "RevenueFromOperations": "revenue",
    "OtherIncome": "other_income",
    "Income": "total_income",
    "ProfitBeforeTax": "profit_before_tax",
    "ProfitLossForPeriod": "net_profit",
    # The full tag: Ind-AS emits no plain `BasicEarningsLossPerShare`, only the
    # three qualified forms. The short name matched nothing and left the column
    # null across 2,247 filings while revenue and profit filled correctly.
    # Combined rather than continuing-only, because that is the headline EPS the
    # price is quoted against.
    "BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations": "eps_basic",
    "DilutedEarningsLossPerShareFromContinuingAndDiscontinuedOperations": "eps_diluted",
}

#: A revenue above this is a parsing failure, not a company. India's largest
#: listed revenue is ~10 trillion rupees a year; 100 trillion in one quarter is
#: two orders of magnitude beyond anything real and indicates a units error.
IMPLAUSIBLE_ABOVE = Decimal("1e14")


class ResultsFormatError(ValueError):
    """The payload was not what this feed parses. Never guessed at."""


@dataclass(frozen=True)
class XbrlFacts:
    """One filing's numbers, for the reporting period only."""

    #: Column name -> value, for whatever was present.
    values: dict[str, Decimal]
    #: Context id the facts were taken from, kept for provenance.
    context: str

    def __bool__(self) -> bool:
        return bool(self.values)


def _local_name(tag: object) -> str:
    """Strip the namespace. Ind-AS documents use several and vary by filer."""
    text = str(tag)
    return text.rsplit("}", 1)[-1]


def _to_decimal(text: str) -> Decimal | None:
    try:
        return Decimal(text.strip().replace(",", ""))
    except (InvalidOperation, AttributeError):
        return None


def _statement_contexts(root: etree._Element) -> set[str]:
    """Quarter contexts that carry statement totals rather than breakdowns.

    Two filters, both structural rather than positional:

    *Undimensioned.* A context with a dimension reports one slice — a segment,
    a product line, an expense category. `OneReportableSegmentRevenue01D` is a
    real context in a real filing, it starts with `One`, and it holds one
    business segment's revenue. Selecting the first `One*` fact encountered
    would record that as the company's, silently, in whatever documents happen
    to emit segments before totals.

    *Duration, not instant.* Every fact here is a flow over the quarter.
    Instant contexts (`OneI`) carry balance-sheet positions and would be a
    different quantity wearing the same tag.
    """
    usable: set[str] = set()
    for element in root.iter():
        if _local_name(element.tag) != "context":
            continue
        context_id = str(element.get("id") or "")
        if not context_id.startswith(QUARTER_CONTEXT):
            continue
        names = {_local_name(node.tag) for node in element.iter()}
        if "explicitMember" in names or "typedMember" in names:
            continue
        if "endDate" not in names or "startDate" not in names:
            continue
        usable.add(context_id)
    return usable


def parse_xbrl(payload: bytes) -> XbrlFacts:
    """Extract the reporting period's figures from an Ind-AS XBRL document.

    Raises:
        ResultsFormatError: if the document does not parse, or carries no
            quarter context.

    **Facts are selected by context id prefix, never by declared period.** The
    quarter and the year-to-date contexts carry identical start and end dates,
    so a date-based selection silently returns cumulative figures for a name
    that filed them in a different order. See the module docstring.

    A document holding only a cumulative context is refused. Its numbers are
    real, but they are not the quarter's, and a table mixing the two is worse
    than one missing the name entirely.
    """
    # `resolve_entities=False` and `no_network=True`: this parses a document
    # fetched over the network, and an entity that reads a local file or
    # re-fetches a URL is a real capability to hand it.
    parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False)
    try:
        root = etree.fromstring(payload, parser=parser)
    except etree.XMLSyntaxError as exc:
        raise ResultsFormatError(f"not parseable XML: {exc}") from exc

    usable = _statement_contexts(root)
    declared = {str(e.get("id") or "") for e in root.iter() if _local_name(e.tag) == "context"}
    values: dict[str, Decimal] = {}
    contexts: set[str] = set()
    skipped: set[str] = set()
    chosen = ""

    for element in root.iter():
        name = _local_name(element.tag)
        if name not in FUNDAMENTAL_FACTS:
            continue
        context = str(element.get("contextRef") or "")
        contexts.add(context)
        # Declared and structurally sound, or undeclared and named as the
        # quarter — see UNDECLARED_QUARTER_CONTEXT.
        if context not in usable and not (
            context not in declared and context == UNDECLARED_QUARTER_CONTEXT
        ):
            continue
        # Pinned to the first usable context that yields a fact. Measured on 25
        # real filings, every document has exactly one, so this costs nothing —
        # but taking facts from two while reporting one context would make the
        # provenance a claim rather than a record.
        if chosen and context != chosen:
            skipped.add(context)
            continue
        value = _to_decimal(element.text or "")
        if value is None:
            continue
        chosen = chosen or context
        values.setdefault(FUNDAMENTAL_FACTS[name], value)

    if not values:
        cumulative = sorted(c for c in contexts if c.startswith(CUMULATIVE_CONTEXT))
        if cumulative:
            raise ResultsFormatError(
                f"no usable {QUARTER_CONTEXT}* context; document holds only cumulative "
                f"figures ({', '.join(cumulative)}), which are not this quarter's"
            )
        raise ResultsFormatError("document carries none of the expected facts")
    if skipped:
        # Never seen in real filings. If it starts happening the taxonomy has
        # changed shape and the selection needs revisiting, so it is loud.
        logger.warning(
            "%s usable contexts in one document; kept %s, ignored %s",
            len(skipped) + 1,
            chosen,
            ", ".join(sorted(skipped)),
        )
    return XbrlFacts(values=values, context=chosen)


def xbrl_is_plausible(facts: XbrlFacts) -> str:
    """Why these numbers should be refused, or an empty string.

    A units error in a filing does not look like an error downstream — it looks
    like a company that grew ten-thousandfold, which is exactly the kind of
    outlier a ranking factor puts straight at the top.
    """
    revenue = facts.values.get("revenue")
    if revenue is not None and abs(revenue) > IMPLAUSIBLE_ABOVE:
        return f"revenue {revenue:.0f} is beyond any real company; likely a units error"
    if revenue is not None and revenue < 0:
        return f"negative revenue {revenue:.0f}"
    return ""
