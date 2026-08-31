"""
NOVA Agent Evaluation Dashboard generator.

Reads reports/latest_run.json (written by tests/conftest.py's
pytest_sessionfinish hook) and, if present, reports/previous_run.json,
and renders a single self-contained reports/dashboard.html - no
external CDN, no extra dependency, opens straight in a browser.

Usage:
    pytest tests/ -v
    python generate_dashboard.py
    # then open reports/dashboard.html
"""

from __future__ import annotations

import html
import json
import statistics
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "reports"
LATEST_PATH = REPORTS_DIR / "latest_run.json"
PREVIOUS_PATH = REPORTS_DIR / "previous_run.json"
OUTPUT_PATH = REPORTS_DIR / "dashboard.html"


def _load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(n, d):
    return round(100 * n / d, 1) if d else 0.0


def _percentile(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * p
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def _aggregate(tests):
    total = len(tests)
    passed = sum(1 for t in tests if t["outcome"] == "passed")
    failed = sum(1 for t in tests if t["outcome"] == "failed")
    skipped = sum(1 for t in tests if t["outcome"] == "skipped")
    durations = [t["duration"] for t in tests]

    by_marker = defaultdict(lambda: {"total": 0, "passed": 0, "failed": 0, "durations": []})
    for t in tests:
        markers = t["markers"] or ["(unmarked)"]
        for m in markers:
            by_marker[m]["total"] += 1
            by_marker[m]["durations"].append(t["duration"])
            if t["outcome"] == "passed":
                by_marker[m]["passed"] += 1
            elif t["outcome"] == "failed":
                by_marker[m]["failed"] += 1

    by_suite = defaultdict(lambda: {"total": 0, "passed": 0, "failed": 0, "durations": []})
    for t in tests:
        s = by_suite[t["suite"]]
        s["total"] += 1
        s["durations"].append(t["duration"])
        if t["outcome"] == "passed":
            s["passed"] += 1
        elif t["outcome"] == "failed":
            s["failed"] += 1

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "pass_rate": _pct(passed, total),
        "failure_rate": _pct(failed, total),
        "avg_duration": round(statistics.fmean(durations), 4) if durations else 0.0,
        "p95_duration": round(_percentile(durations, 0.95), 4),
        "by_marker": dict(by_marker),
        "by_suite": {k: v for k, v in by_suite.items()},
    }


def _find_regressions(latest_tests, previous_tests):
    if previous_tests is None:
        return None
    prev_by_id = {t["nodeid"]: t for t in previous_tests}
    regressions = []
    for t in latest_tests:
        prev = prev_by_id.get(t["nodeid"])
        if prev and prev["outcome"] == "passed" and t["outcome"] == "failed":
            regressions.append(t)
    return regressions


def _bar(pct, color):
    pct = max(0, min(100, pct))
    return (
        f'<div class="bar-track"><div class="bar-fill" '
        f'style="width:{pct}%;background:{color}"></div></div>'
    )


def _color_for_rate(pct):
    if pct >= 95:
        return "#2e7d32"
    if pct >= 80:
        return "#f9a825"
    return "#c62828"


def build_html(latest, previous) -> str:
    tests = latest["tests"]
    agg = _aggregate(tests)
    prev_tests = previous["tests"] if previous else None
    regressions = _find_regressions(tests, prev_tests)

    prev_agg = _aggregate(prev_tests) if prev_tests else None

    def delta_html(current, previous_val, suffix=""):
        if previous_val is None:
            return ""
        diff = round(current - previous_val, 2)
        if diff == 0:
            return '<span class="delta neutral">±0</span>'
        arrow = "▲" if diff > 0 else "▼"
        cls = "up" if diff > 0 else "down"
        return f'<span class="delta {cls}">{arrow} {abs(diff)}{suffix}</span>'

    summary_cards = f"""
    <div class="cards">
      <div class="card">
        <div class="card-label">Total tests</div>
        <div class="card-value">{agg['total']}</div>
      </div>
      <div class="card">
        <div class="card-label">Pass rate</div>
        <div class="card-value" style="color:{_color_for_rate(agg['pass_rate'])}">{agg['pass_rate']}%</div>
        {delta_html(agg['pass_rate'], prev_agg['pass_rate'] if prev_agg else None, '%')}
      </div>
      <div class="card">
        <div class="card-label">Failed</div>
        <div class="card-value" style="color:{'#c62828' if agg['failed'] else '#2e7d32'}">{agg['failed']}</div>
      </div>
      <div class="card">
        <div class="card-label">Skipped</div>
        <div class="card-value">{agg['skipped']}</div>
      </div>
      <div class="card">
        <div class="card-label">Avg duration</div>
        <div class="card-value">{agg['avg_duration']}s</div>
      </div>
      <div class="card">
        <div class="card-label">p95 duration</div>
        <div class="card-value">{agg['p95_duration']}s</div>
      </div>
    </div>
    """

    # ---- Agent Evaluation Metrics ----
    # Named metrics mapping directly onto the evaluation-framework
    # requirements (accuracy / response time / success rate / failure
    # rate), computed from the same per-test data as everything else
    # here - just labeled by what each test category actually proves,
    # not generic pytest terminology.
    METRIC_DEFINITIONS = [
        ("routing", "Routing Accuracy",
         "% of queries where the correct agent was selected"),
        ("agent", "Agent Action Success Rate",
         "% of agent actions (apply/approve/reject/send/schedule) that produced the correct result"),
        ("a2a", "Cross-System Validation Rate",
         "% of agent-to-agent / cross-system checks (Leave\u2194PO\u2194Calendar) that passed"),
        ("failure", "Failure-Handling Correctness",
         "% of failure scenarios (timeout, unavailable agent, invalid data, permission) handled gracefully"),
    ]

    metric_cards = ""
    for marker_key, label, description in METRIC_DEFINITIONS:
        stats = agg["by_marker"].get(marker_key)
        if not stats or stats["total"] == 0:
            continue
        rate = _pct(stats["passed"], stats["total"])
        avg_dur = (
            round(statistics.fmean(stats["durations"]), 4)
            if stats["durations"] else 0.0
        )
        prev_stats = prev_agg["by_marker"].get(marker_key) if prev_agg else None
        prev_rate = _pct(prev_stats["passed"], prev_stats["total"]) if prev_stats else None
        metric_cards += f"""
        <div class="metric-card">
          <div class="metric-label">{html.escape(label)}</div>
          <div class="metric-value" style="color:{_color_for_rate(rate)}">{rate}%</div>
          {delta_html(rate, prev_rate, '%')}
          <div class="metric-sub">{stats['passed']}/{stats['total']} passed &middot; avg {avg_dur}s</div>
          <div class="metric-desc">{html.escape(description)}</div>
        </div>
        """

    agent_metrics_html = (
        f'<div class="metric-cards">{metric_cards}</div>'
        if metric_cards else
        '<div class="banner neutral">No routing/agent/a2a/failure-marked tests found in this run.</div>'
    )

    # ---- Regression banner ----
    if regressions is None:
        regression_html = (
            '<div class="banner neutral">No previous run to compare against yet - '
            "run the suite again after a code change to enable regression detection.</div>"
        )
    elif len(regressions) == 0:
        regression_html = (
            '<div class="banner ok">✅ No regressions - every test that passed last run '
            "still passes.</div>"
        )
    else:
        rows = "".join(
            f'<tr><td>{html.escape(r["nodeid"])}</td>'
            f'<td>{", ".join(r["markers"]) or "-"}</td></tr>'
            for r in regressions
        )
        regression_html = f"""
        <div class="banner bad">🚨 {len(regressions)} REGRESSION(S) - passed last run, failing now</div>
        <table class="data-table">
          <thead><tr><th>Test</th><th>Category</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
        """

    # ---- By category (marker) ----
    marker_rows = ""
    for marker, stats in sorted(agg["by_marker"].items()):
        rate = _pct(stats["passed"], stats["total"])
        avg_dur = (
            round(statistics.fmean(stats["durations"]), 4)
            if stats["durations"] else 0.0
        )
        marker_rows += f"""
        <tr>
          <td>{html.escape(marker)}</td>
          <td>{stats['total']}</td>
          <td>{stats['passed']}</td>
          <td>{stats['failed']}</td>
          <td>{rate}%</td>
          <td>{avg_dur}s</td>
          <td>{_bar(rate, _color_for_rate(rate))}</td>
        </tr>
        """

    # ---- By suite (test file / agent) ----
    suite_rows = ""
    for suite, stats in sorted(agg["by_suite"].items()):
        rate = _pct(stats["passed"], stats["total"])
        avg_dur = round(statistics.fmean(stats["durations"]), 4) if stats["durations"] else 0.0
        suite_rows += f"""
        <tr>
          <td>{html.escape(suite)}</td>
          <td>{stats['total']}</td>
          <td>{stats['passed']}</td>
          <td>{stats['failed']}</td>
          <td>{rate}%</td>
          <td>{avg_dur}s</td>
          <td>{_bar(rate, _color_for_rate(rate))}</td>
        </tr>
        """

    # ---- Failures table ----
    failures = [t for t in tests if t["outcome"] == "failed"]
    if failures:
        failure_rows = "".join(
            f"""
            <tr>
              <td>{html.escape(f['nodeid'])}</td>
              <td>{', '.join(f['markers']) or '-'}</td>
              <td><pre class="msg">{html.escape((f['message'] or '')[:400])}</pre></td>
            </tr>
            """
            for f in failures
        )
        failures_html = f"""
        <table class="data-table">
          <thead><tr><th>Test</th><th>Category</th><th>Error</th></tr></thead>
          <tbody>{failure_rows}</tbody>
        </table>
        """
    else:
        failures_html = '<div class="banner ok">✅ No failing tests in this run.</div>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NOVA Agent Evaluation Dashboard</title>
<style>
  :root {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; }}
  body {{ margin: 0; background: #f4f5f7; color: #1a1a1a; }}
  header {{ background: #14213d; color: white; padding: 24px 32px; }}
  header h1 {{ margin: 0; font-size: 22px; }}
  header .meta {{ opacity: 0.75; font-size: 13px; margin-top: 4px; }}
  main {{ max-width: 1100px; margin: 0 auto; padding: 24px 32px 64px; }}
  section {{ margin-bottom: 36px; }}
  h2 {{ font-size: 16px; text-transform: uppercase; letter-spacing: 0.04em;
        color: #444; border-bottom: 2px solid #e0e0e0; padding-bottom: 8px; }}
  .cards {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; }}
  .card {{ background: white; border-radius: 10px; padding: 16px;
           box-shadow: 0 1px 3px rgba(0,0,0,0.08); text-align: center; }}
  .card-label {{ font-size: 12px; color: #777; text-transform: uppercase; }}
  .card-value {{ font-size: 26px; font-weight: 700; margin-top: 4px; }}
  .delta {{ font-size: 12px; display: block; margin-top: 2px; }}
  .delta.up {{ color: #2e7d32; }}
  .delta.down {{ color: #c62828; }}
  .delta.neutral {{ color: #999; }}
  .banner {{ padding: 12px 16px; border-radius: 8px; font-weight: 600; margin-bottom: 12px; }}
  .banner.ok {{ background: #e8f5e9; color: #2e7d32; }}
  .banner.bad {{ background: #fdecea; color: #c62828; }}
  .banner.neutral {{ background: #eef1f5; color: #555; font-weight: 400; }}
  table.data-table {{ width: 100%; border-collapse: collapse; background: white;
                       border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  table.data-table th, table.data-table td {{ text-align: left; padding: 10px 12px;
                       border-bottom: 1px solid #eee; font-size: 13px; vertical-align: top; }}
  table.data-table th {{ background: #fafafa; color: #555; }}
  .bar-track {{ background: #eee; border-radius: 4px; height: 8px; width: 120px; overflow: hidden; }}
  .bar-fill {{ height: 100%; }}
  pre.msg {{ white-space: pre-wrap; margin: 0; font-size: 11px; color: #c62828;
             max-height: 120px; overflow-y: auto; }}
  .metric-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
  .metric-card {{ background: white; border-radius: 10px; padding: 16px;
                  box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .metric-label {{ font-size: 12px; color: #777; text-transform: uppercase; font-weight: 600; }}
  .metric-value {{ font-size: 28px; font-weight: 700; margin-top: 4px; }}
  .metric-sub {{ font-size: 12px; color: #555; margin-top: 4px; }}
  .metric-desc {{ font-size: 12px; color: #888; margin-top: 8px; line-height: 1.4; }}
</style>
</head>
<body>
<header>
  <h1>NOVA Agent Evaluation Dashboard</h1>
  <div class="meta">Run: {html.escape(latest['run_timestamp'])} &nbsp;|&nbsp;
    Total duration: {latest['total_duration_seconds']}s &nbsp;|&nbsp;
    Exit status: {latest['exit_status']}</div>
</header>
<main>
  <section>
    {summary_cards}
  </section>

  <section>
    <h2>Agent Evaluation Metrics</h2>
    {agent_metrics_html}
  </section>

  <section>
    <h2>Regression Check</h2>
    {regression_html}
  </section>

  <section>
    <h2>By Test Category</h2>
    <table class="data-table">
      <thead><tr><th>Category</th><th>Total</th><th>Passed</th><th>Failed</th><th>Pass Rate</th><th>Avg Duration</th><th></th></tr></thead>
      <tbody>{marker_rows}</tbody>
    </table>
  </section>

  <section>
    <h2>By Suite / Agent</h2>
    <table class="data-table">
      <thead><tr><th>Suite</th><th>Total</th><th>Passed</th><th>Failed</th><th>Pass Rate</th><th>Avg Duration</th><th></th></tr></thead>
      <tbody>{suite_rows}</tbody>
    </table>
  </section>

  <section>
    <h2>Failures</h2>
    {failures_html}
  </section>
</main>
</body>
</html>
"""


def main():
    latest = _load(LATEST_PATH)
    if latest is None:
        raise SystemExit(
            f"No {LATEST_PATH} found - run `pytest tests/ -v` first."
        )
    previous = _load(PREVIOUS_PATH)

    html_out = build_html(latest, previous)
    OUTPUT_PATH.write_text(html_out, encoding="utf-8")
    print(f"Dashboard written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()