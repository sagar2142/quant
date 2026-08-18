"""Dependency-free SVG charts — MASTER_PLAN §12.6.

**Why raw SVG rather than matplotlib or plotly.** The research surface has to
produce an *artifact*, not a session: §5 demands every number be traceable to an
experiment row a year later, and a chart that only exists inside a running
server is not traceable. A self-contained SVG embedded in one HTML file can sit
beside the experiment row, attach to a CI run, and open on a machine with
nothing installed.

The cost of that independence is about two hundred lines of path arithmetic,
paid once. The alternative is ~200MB of plotting dependencies and a server
process to look at a line.

**Colours come from the design tokens** (§12.2) so a report and the operations
console never disagree about what "loss" looks like. Semantic or absent — no
decoration.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

__all__ = [
    "PALETTE",
    "Series",
    "area_chart",
    "bar_chart",
    "line_chart",
]

#: Mirrors apps/web/src/tokens.css. Duplicated deliberately: a Python report
#: cannot import a stylesheet, and one shared vocabulary matters more than one
#: shared file. Change both together.
PALETTE = {
    "bg": "#13161a",
    "inset": "#080a0c",
    "grid": "#252a31",
    "axis": "#5e6772",
    "text": "#98a1ac",
    "profit": "#26a69a",
    "loss": "#ef5350",
    "warn": "#f0a93b",
    "accent": "#4a9eff",
    "flat": "#98a1ac",
}

#: Plot box. Wide and short: an equity curve is read for shape and drawdown
#: depth, and a tall aspect ratio exaggerates both (§12.6).
WIDTH = 900
HEIGHT = 260
PAD_LEFT = 64
PAD_RIGHT = 16
PAD_TOP = 16
PAD_BOTTOM = 28

#: Horizontal gridlines. Five is enough to read a level and few enough that the
#: lines stay behind the data rather than competing with it.
GRID_LINES = 5

#: Below this slot width a bar label would overlap its neighbour. Dropping
#: labels beats printing an unreadable smear of overlapping text.
MIN_SLOT_FOR_LABEL = 28


#: Turns an axis value into its label. Passed in rather than fixed because a
#: drawdown axis reads as a percentage and an equity axis as a currency amount,
#: and a chart that labels one as the other is worse than an unlabelled one.
Formatter = Callable[[float], str]


def _thousands(value: float) -> str:
    return f"{value:,.0f}"


def _percent(value: float) -> str:
    return f"{value:.0%}"


def _two_places(value: float) -> str:
    return f"{value:.2f}"


@dataclass(frozen=True)
class Series:
    """One named line, already reduced to plottable floats."""

    label: str
    values: list[float]
    colour: str = PALETTE["accent"]


def _escape(text: str) -> str:
    """Minimal XML escaping.

    Labels come from strategy names and instrument ids, which are ours — but a
    report that renders a stray ampersand as broken XML is a report nobody
    trusts, so it costs nothing to be correct here.
    """
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def _plot_box() -> tuple[float, float]:
    return WIDTH - PAD_LEFT - PAD_RIGHT, HEIGHT - PAD_TOP - PAD_BOTTOM


def _bounds(values: list[float]) -> tuple[float, float]:
    """Low and high for the y axis, never a zero-height range.

    A flat series (an untraded account, a constant) would otherwise divide by
    zero and render as a line through infinity.
    """
    low, high = min(values), max(values)
    if high == low:
        pad = abs(high) * 0.05 or 1.0
        return low - pad, high + pad
    margin = (high - low) * 0.08
    return low - margin, high + margin


def _project(values: list[float], low: float, high: float) -> list[tuple[float, float]]:
    """Data space to SVG space. Y is inverted: SVG grows downward."""
    plot_w, plot_h = _plot_box()
    span = high - low
    if len(values) == 1:
        return [(PAD_LEFT, PAD_TOP + plot_h / 2)]
    step = plot_w / (len(values) - 1)
    return [
        (PAD_LEFT + i * step, PAD_TOP + plot_h - ((v - low) / span) * plot_h)
        for i, v in enumerate(values)
    ]


def _grid(low: float, high: float, formatter: Formatter) -> str:
    plot_w, plot_h = _plot_box()
    parts = []
    for i in range(GRID_LINES + 1):
        fraction = i / GRID_LINES
        y = PAD_TOP + plot_h - fraction * plot_h
        value = low + fraction * (high - low)
        parts.append(
            f'<line x1="{PAD_LEFT}" y1="{y:.1f}" x2="{PAD_LEFT + plot_w}" '
            f'y2="{y:.1f}" stroke="{PALETTE["grid"]}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{PAD_LEFT - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'fill="{PALETTE["text"]}" font-size="11" '
            f'font-family="ui-monospace, monospace">{_escape(formatter(value))}</text>'
        )
    return "".join(parts)


def _frame(body: str, title: str) -> str:
    return (
        f'<svg viewBox="0 0 {WIDTH} {HEIGHT}" width="100%" '
        f'preserveAspectRatio="xMidYMid meet" role="img" '
        f'aria-label="{_escape(title)}" xmlns="http://www.w3.org/2000/svg">'
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{PALETTE["inset"]}"/>'
        f"{body}</svg>"
    )


def line_chart(series: list[Series], formatter: Formatter = _thousands) -> str:
    """One or more lines on a shared axis.

    Shared axis on purpose: two curves drawn on independent scales look
    comparable and are not, which is among the easier ways to mislead yourself
    about a strategy.
    """
    populated = [s for s in series if s.values]
    if not populated:
        return _frame(
            f'<text x="{WIDTH / 2}" y="{HEIGHT / 2}" text-anchor="middle" '
            f'fill="{PALETTE["text"]}" font-size="12">no data</text>',
            "empty chart",
        )

    every = [v for s in populated for v in s.values]
    low, high = _bounds(every)

    body = [_grid(low, high, formatter)]
    for line in populated:
        points = _project(line.values, low, high)
        path = " ".join(
            f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(points)
        )
        body.append(
            f'<path d="{path}" fill="none" stroke="{line.colour}" '
            f'stroke-width="1.5" stroke-linejoin="round"/>'
        )
    return _frame("".join(body), populated[0].label)


def area_chart(
    values: list[float],
    colour: str = PALETTE["loss"],
    formatter: Formatter = _percent,
) -> str:
    """A filled series against zero. Built for drawdown.

    Drawdown is always negative, so the fill hangs from the top of the box and
    depth reads immediately — which is the one thing a drawdown chart exists to
    show.
    """
    if not values:
        return line_chart([])

    low, high = _bounds([*values, 0.0])
    points = _project(values, low, high)
    plot_h = HEIGHT - PAD_TOP - PAD_BOTTOM
    zero_y = PAD_TOP + plot_h - ((0.0 - low) / (high - low)) * plot_h

    path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(points))
    fill = f"{path} L{points[-1][0]:.1f},{zero_y:.1f} L{points[0][0]:.1f},{zero_y:.1f} Z"
    body = (
        _grid(low, high, formatter)
        + f'<path d="{fill}" fill="{colour}" fill-opacity="0.18"/>'
        + f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="1.5"/>'
    )
    return _frame(body, "drawdown")


def bar_chart(
    labels: list[str],
    values: list[float],
    formatter: Formatter = _two_places,
    threshold: float | None = None,
) -> str:
    """Discrete comparison. Built for the parameter neighbourhood (§5.4 test 6).

    This is the mesa-versus-needle picture: a spike at one parameter value with
    collapse either side is a fitted artefact of one noise realisation, and it
    is far more obvious here than in a column of numbers.

    Bars below `threshold` are drawn in the loss colour, so "which settings
    actually survive" is answerable at a glance rather than by reading.
    """
    if not values:
        return line_chart([])

    low, high = _bounds([*values, 0.0])
    plot_w, plot_h = _plot_box()
    zero_y = PAD_TOP + plot_h - ((0.0 - low) / (high - low)) * plot_h
    slot = plot_w / len(values)
    bar_w = max(2.0, slot * 0.62)

    body = [_grid(low, high, formatter)]
    for i, value in enumerate(values):
        centre = PAD_LEFT + slot * (i + 0.5)
        y = PAD_TOP + plot_h - ((value - low) / (high - low)) * plot_h
        top, height = (y, zero_y - y) if value >= 0 else (zero_y, y - zero_y)
        colour = (
            PALETTE["loss"] if threshold is not None and value < threshold else PALETTE["profit"]
        )
        body.append(
            f'<rect x="{centre - bar_w / 2:.1f}" y="{top:.1f}" '
            f'width="{bar_w:.1f}" height="{max(height, 1):.1f}" fill="{colour}" '
            f'fill-opacity="0.75"/>'
        )
        if len(labels) == len(values) and slot > MIN_SLOT_FOR_LABEL:
            body.append(
                f'<text x="{centre:.1f}" y="{HEIGHT - 8}" text-anchor="middle" '
                f'fill="{PALETTE["text"]}" font-size="10" '
                f'font-family="ui-monospace, monospace">{_escape(labels[i])}</text>'
            )

    if threshold is not None:
        ty = PAD_TOP + plot_h - ((threshold - low) / (high - low)) * plot_h
        body.append(
            f'<line x1="{PAD_LEFT}" y1="{ty:.1f}" x2="{PAD_LEFT + plot_w}" '
            f'y2="{ty:.1f}" stroke="{PALETTE["warn"]}" stroke-width="1" '
            f'stroke-dasharray="4 3"/>'
        )
    return _frame("".join(body), "parameter neighbourhood")
