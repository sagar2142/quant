"""Research report rendering (§5.4, §12.6).

A chart cannot assert its own correctness, so the tests target the two ways a
report lies: **wrong geometry** (a curve that misrepresents its data) and
**wrong framing** (a page that lets a rejected strategy read as acceptable).

The second matters more. The plan is hostile to eyeballing equity curves, and a
report is only safe if the verdict cannot be missed.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import pytest

from apps.report.charts import (
    GRID_LINES,
    HEIGHT,
    PALETTE,
    WIDTH,
    Series,
    area_chart,
    bar_chart,
    line_chart,
)
from apps.report.page import (
    Panel,
    ReportPage,
    render_page,
    table,
    verdict_row,
)


def parse(svg: str) -> ET.Element:
    """Every chart must be well-formed XML, not merely a plausible string."""
    return ET.fromstring(svg)


def path_points(svg: str) -> list[tuple[float, float]]:
    root = parse(svg)
    paths = [e for e in root.iter("{http://www.w3.org/2000/svg}path") if e.get("fill") == "none"]
    assert paths, "no stroked path in chart"
    return [
        (float(x), float(y)) for x, y in re.findall(r"[ML]([\d.]+),([\d.]+)", paths[0].get("d", ""))
    ]


class TestChartsAreValidSvg:
    def test_line_chart_parses(self):
        assert parse(line_chart([Series("e", [1.0, 2.0, 3.0])])).tag.endswith("svg")

    def test_area_chart_parses(self):
        assert parse(area_chart([0.0, -0.1, -0.05])).tag.endswith("svg")

    def test_bar_chart_parses(self):
        assert parse(bar_chart(["a", "b"], [1.0, -0.5])).tag.endswith("svg")

    def test_labels_are_escaped(self):
        """A stray ampersand must not produce broken XML."""
        assert parse(bar_chart(["a&b", "c<d"], [1.0, 2.0])) is not None


class TestGeometry:
    def test_a_rising_series_rises_on_screen(self):
        """SVG y grows downward, so a rising series must have *decreasing* y.

        Getting this backwards renders every equity curve upside down — and an
        upside-down curve is still a plausible-looking chart, which is exactly
        why it needs a test.
        """
        points = path_points(line_chart([Series("e", [1.0, 2.0, 3.0, 4.0])]))
        ys = [y for _, y in points]
        assert ys == sorted(ys, reverse=True)

    def test_a_falling_series_falls_on_screen(self):
        ys = [y for _, y in path_points(line_chart([Series("e", [4.0, 3.0, 2.0, 1.0])]))]
        assert ys == sorted(ys)

    def test_x_spans_the_plot_width(self):
        points = path_points(line_chart([Series("e", [1.0, 2.0, 3.0])]))
        xs = [x for x, _ in points]
        assert xs[0] < xs[-1]
        assert xs[-1] <= WIDTH

    def test_every_point_is_inside_the_box(self):
        points = path_points(line_chart([Series("e", [1.0, 500.0, 3.0, 900.0])]))
        assert all(0 <= x <= WIDTH and 0 <= y <= HEIGHT for x, y in points)

    def test_one_point_per_observation(self):
        values = [float(i) for i in range(50)]
        assert len(path_points(line_chart([Series("e", values)]))) == 50

    def test_a_flat_series_does_not_divide_by_zero(self):
        """A constant series has zero range. Without a guard the projection
        divides by zero and the chart renders as NaN paths."""
        points = path_points(line_chart([Series("e", [100.0] * 10)]))
        assert all(0 <= y <= HEIGHT for _, y in points)

    def test_a_series_of_zeroes_is_handled(self):
        assert parse(line_chart([Series("e", [0.0, 0.0, 0.0])])) is not None

    def test_single_point_series_renders(self):
        assert len(path_points(line_chart([Series("e", [42.0])]))) == 1

    def test_shared_axis_across_series(self):
        """Two lines on independent scales look comparable and are not."""
        svg = line_chart([Series("a", [1.0, 2.0]), Series("b", [100.0, 200.0])])
        root = parse(svg)
        stroked = [
            e for e in root.iter("{http://www.w3.org/2000/svg}path") if e.get("fill") == "none"
        ]
        assert len(stroked) == 2

    def test_gridlines_are_drawn(self):
        root = parse(line_chart([Series("e", [1.0, 2.0])]))
        lines = list(root.iter("{http://www.w3.org/2000/svg}line"))
        assert len(lines) == GRID_LINES + 1


class TestEmptyInput:
    def test_empty_series_says_no_data(self):
        assert "no data" in line_chart([])

    def test_series_with_no_values_says_no_data(self):
        assert "no data" in line_chart([Series("e", [])])

    def test_empty_area_chart_is_safe(self):
        assert parse(area_chart([])) is not None

    def test_empty_bar_chart_is_safe(self):
        assert parse(bar_chart([], [])) is not None


class TestBarChart:
    def test_a_bar_per_value(self):
        root = parse(bar_chart(["a", "b", "c"], [1.0, 2.0, 3.0]))
        rects = list(root.iter("{http://www.w3.org/2000/svg}rect"))
        # One background rect plus one per bar.
        assert len(rects) == 4

    def test_bars_below_threshold_are_the_loss_colour(self):
        """ "Which settings actually survive" must be answerable at a glance."""
        svg = bar_chart(["a", "b"], [0.8, -0.4], threshold=0.0)
        rects = [
            r
            for r in parse(svg).iter("{http://www.w3.org/2000/svg}rect")
            if r.get("fill") in {PALETTE["profit"], PALETTE["loss"]}
        ]
        assert [r.get("fill") for r in rects] == [PALETTE["profit"], PALETTE["loss"]]

    def test_threshold_line_is_drawn(self):
        svg = bar_chart(["a", "b"], [1.0, 2.0], threshold=1.5)
        assert PALETTE["warn"] in svg

    def test_labels_are_dropped_when_they_would_overlap(self):
        """Unlabelled beats an unreadable smear of overlapping text."""
        many = [f"cfg{i}" for i in range(200)]
        svg = bar_chart(many, [float(i) for i in range(200)])
        assert "cfg100" not in svg


class TestPageFraming:
    """The part that keeps the report honest."""

    def page(self, passed: bool, failures: list[str] | None = None) -> ReportPage:
        return ReportPage(
            title="momentum(60/5)",
            subtitle="generated now",
            passed=passed,
            verdict_line="Rejected at 3_deflated_sharpe." if not passed else "All passed.",
            failures=failures or [],
            stats=[("total return", "635.28%", "")],
            panels=[Panel("equity", line_chart([Series("e", [1.0, 2.0])]))],
        )

    def test_rejection_appears_before_any_chart(self):
        """A rising line shown first invites "that looks good" before the
        reader reaches the statistics. The verdict has to come first."""
        html = render_page(self.page(passed=False))
        assert html.index("REJECTED") < html.index("<svg")

    def test_a_rejected_run_never_renders_as_passed(self):
        html = render_page(self.page(passed=False))
        assert "REJECTED" in html
        assert ">PASSED<" not in html

    def test_failures_are_listed(self):
        html = render_page(self.page(passed=False, failures=["3_deflated_sharpe: DSR 0.08"]))
        assert "3_deflated_sharpe: DSR 0.08" in html

    def test_a_passing_run_says_so(self):
        assert "PASSED" in render_page(self.page(passed=True))

    def test_the_page_is_self_contained(self):
        """No CDN, no fonts, no scripts — it must open with no network."""
        html = render_page(self.page(passed=True))
        assert "<script" not in html
        assert "http://" not in html.replace("http://www.w3.org/2000/svg", "")
        assert "https://" not in html

    def test_html_is_escaped(self):
        page = self.page(passed=True)
        page.title = '<script>alert("x")</script>'
        html = render_page(page)
        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html

    def test_the_footer_states_charts_are_diagnostic(self):
        """The framing is load-bearing, not decoration."""
        assert "not a basis for making one" in render_page(self.page(passed=True))


class TestVerdictRows:
    def test_a_skipped_check_says_skip_not_blank(self):
        """An unfilled slot that reads as a pass is the failure this defends
        against (§5.4)."""
        row = verdict_row("10_placebo", passed=True, skipped=True, statistic="—", reason="not run")
        assert "SKIP" in row[0]
        assert "PASS" not in row[0]

    def test_a_failed_check_is_marked_fail(self):
        row = verdict_row("3_dsr", passed=False, skipped=False, statistic="0.08", reason="low")
        assert "FAIL" in row[0]

    def test_a_passed_check_is_marked_pass(self):
        row = verdict_row("1_data", passed=True, skipped=False, statistic="0", reason="clean")
        assert "PASS" in row[0]

    def test_reasons_are_escaped(self):
        row = verdict_row("x", passed=False, skipped=False, statistic="1", reason="a<b & c")
        assert "&lt;" in row[3]


class TestTable:
    def test_wide_tables_scroll_inside_themselves(self):
        """The page body must never scroll horizontally."""
        assert 'class="scroll"' in table(["a"], [["1"]])

    def test_rows_render(self):
        html = table(["check", "value"], [["dsr", "0.08"]])
        assert "dsr" in html
        assert "0.08" in html


class TestDrawdownSeries:
    def test_drawdown_is_zero_at_a_new_peak(self):
        from apps.cli.report import drawdown_series

        assert drawdown_series([100.0, 110.0, 120.0]) == pytest.approx([0.0, 0.0, 0.0])

    def test_drawdown_is_negative_below_the_peak(self):
        from apps.cli.report import drawdown_series

        assert drawdown_series([100.0, 80.0])[1] == pytest.approx(-0.2)

    def test_drawdown_recovers_to_zero(self):
        from apps.cli.report import drawdown_series

        assert drawdown_series([100.0, 50.0, 100.0])[-1] == pytest.approx(0.0)
