"""HTML assembly for the research report — MASTER_PLAN §5.4, §12.6.

**The verdict goes first, above every chart.** This is the whole design
constraint. The plan is hostile to eyeballing equity curves, and for good
reason: "that looks good" is the bias the gauntlet exists to kill. A report
that opens with a rising line invites exactly that judgement before the reader
reaches the statistics.

So the page opens with PASSED or REJECTED and the failing checks. The charts
below it are *diagnostic* — they explain a verdict that has already been
reached, and they are labelled that way. Nothing here is a decision surface.

Self-contained by construction: one file, inline CSS, inline SVG, no scripts
and no network. It opens from a CI artifact, an email attachment, or a folder
on a laptop with nothing installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Panel", "ReportPage", "render_page", "stat_grid", "table", "verdict_row"]

STYLE = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 32px;
  background: #0b0d10; color: #e6e9ed;
  font: 14px/1.55 Inter, system-ui, -apple-system, sans-serif;
}
.wrap { max-width: 980px; margin: 0 auto; }
h1 { font-size: 20px; font-weight: 600; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 13px; font-weight: 600; margin: 0 0 12px;
     text-transform: uppercase; letter-spacing: 0.06em; color: #98a1ac; }
.sub { color: #5e6772; font-size: 13px; margin: 0 0 24px;
       font-family: ui-monospace, monospace; }
.verdict { border: 1px solid; border-radius: 4px; padding: 16px 20px; margin: 0 0 32px; }
.verdict.pass { border-color: #26a69a; background: rgba(38,166,154,0.08); }
.verdict.fail { border-color: #ef5350; background: rgba(239,83,80,0.08); }
.verdict h2 { color: inherit; margin-bottom: 8px; }
.verdict.pass h2 { color: #26a69a; }
.verdict.fail h2 { color: #ef5350; }
.verdict ul { margin: 8px 0 0; padding-left: 20px; color: #98a1ac; }
.verdict li { margin: 3px 0; font-family: ui-monospace, monospace; font-size: 12.5px; }
.panel { background: #13161a; border: 1px solid #252a31; border-radius: 4px;
         padding: 20px; margin: 0 0 20px; }
.panel .note { color: #5e6772; font-size: 12.5px; margin: 10px 0 0; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1px; background: #252a31; border: 1px solid #252a31;
        border-radius: 4px; overflow: hidden; margin: 0 0 20px; }
.stat { background: #13161a; padding: 12px 16px; }
.stat .k { color: #5e6772; font-size: 11px; text-transform: uppercase;
           letter-spacing: 0.05em; }
.stat .v { font-family: ui-monospace, monospace; font-size: 17px;
           margin-top: 3px; font-variant-numeric: tabular-nums; }
table { width: 100%; border-collapse: collapse; font-size: 12.5px;
        font-family: ui-monospace, monospace; }
th { text-align: left; color: #5e6772; font-weight: 500; padding: 6px 10px;
     border-bottom: 1px solid #252a31; text-transform: uppercase;
     font-size: 10.5px; letter-spacing: 0.05em; }
td { padding: 6px 10px; border-bottom: 1px solid #1a1e24;
     font-variant-numeric: tabular-nums; }
tr:last-child td { border-bottom: none; }
.scroll { overflow-x: auto; }
.ok { color: #26a69a; } .bad { color: #ef5350; }
.skip { color: #5e6772; } .warn { color: #f0a93b; }
footer { color: #5e6772; font-size: 12px; margin-top: 32px;
         border-top: 1px solid #252a31; padding-top: 16px; }
@media (max-width: 640px) { body { padding: 16px; } }
"""


def escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


@dataclass(frozen=True)
class Panel:
    """One titled block: a chart, a table, or both."""

    title: str
    body: str
    note: str = ""

    def render(self) -> str:
        note = f'<p class="note">{escape(self.note)}</p>' if self.note else ""
        return f'<section class="panel"><h2>{escape(self.title)}</h2>{self.body}{note}</section>'


@dataclass
class ReportPage:
    """A complete research report."""

    title: str
    subtitle: str
    passed: bool
    verdict_line: str
    failures: list[str] = field(default_factory=list)
    stats: list[tuple[str, str, str]] = field(default_factory=list)
    panels: list[Panel] = field(default_factory=list)
    footer: str = ""


def stat_grid(stats: list[tuple[str, str, str]]) -> str:
    """Key figures. `tone` is "", "ok", "bad" or "warn"."""
    if not stats:
        return ""
    cells = "".join(
        f'<div class="stat"><div class="k">{escape(k)}</div>'
        f'<div class="v {tone}">{escape(v)}</div></div>'
        for k, v, tone in stats
    )
    return f'<div class="grid">{cells}</div>'


def table(headers: list[str], rows: list[list[str]]) -> str:
    """A table whose wide content scrolls inside itself, never the page."""
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def render_page(page: ReportPage) -> str:
    """One self-contained HTML document. No scripts, no network."""
    tone = "pass" if page.passed else "fail"
    heading = "PASSED" if page.passed else "REJECTED"
    failures = (
        "<ul>" + "".join(f"<li>{escape(f)}</li>" for f in page.failures) + "</ul>"
        if page.failures
        else ""
    )
    panels = "".join(p.render() for p in page.panels)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(page.title)}</title>
<style>{STYLE}</style></head>
<body><div class="wrap">
<h1>{escape(page.title)}</h1>
<p class="sub">{escape(page.subtitle)}</p>

<div class="verdict {tone}">
  <h2>{heading}</h2>
  <div>{escape(page.verdict_line)}</div>
  {failures}
</div>

{stat_grid(page.stats)}
{panels}

<footer>{escape(page.footer)}<br>
Charts below the verdict are diagnostic. They explain a decision the gauntlet
already made; they are not a basis for making one. A rising line is not
evidence — that is what the twelve checks are for (&#167;5.4).
</footer>
</div></body></html>"""


def verdict_row(name: str, passed: bool, skipped: bool, statistic: str, reason: str) -> list[str]:
    """One gauntlet check as a table row, coloured by outcome.

    A skipped check renders grey and says SKIP, never blank. An unfilled slot
    that looks like a pass is the failure mode this whole report defends
    against (§5.4).
    """
    if skipped:
        mark, klass = "SKIP", "skip"
    elif passed:
        mark, klass = "PASS", "ok"
    else:
        mark, klass = "FAIL", "bad"
    return [
        f'<span class="{klass}">{mark}</span>',
        escape(name),
        escape(statistic),
        escape(reason),
    ]
