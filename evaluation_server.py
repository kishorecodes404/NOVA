"""
NOVA Evaluation Server.

Serves reports/dashboard.html and exposes a /run-evaluation endpoint
that the dashboard's "Run Evaluation" button calls. That endpoint
runs `pytest tests/ -v` and then `python generate_dashboard.py`,
exactly as if you'd typed both commands yourself, then tells the
browser to reload with the fresh results.

Usage:
    python evaluation_server.py

Then open the URL it prints and click "Run Evaluation" whenever you
want to re-run the suite - no terminal needed after this one command.
"""

from __future__ import annotations

import http.server
import json
import socketserver
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "reports"
DASHBOARD_PATH = REPORTS_DIR / "dashboard.html"
PORT = 8765

_run_lock = threading.Lock()


def _run_full_evaluation() -> dict:
    """Runs the real pytest suite, then the real dashboard generator,
    as actual subprocesses - not a simulation of either."""
    pytest_result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )

    dashboard_result = subprocess.run(
        [sys.executable, "generate_dashboard.py"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )

    ok = dashboard_result.returncode == 0
    return {
        "status": "ok" if ok else "error",
        "pytest_exit_code": pytest_result.returncode,
        "pytest_tail": "\n".join(pytest_result.stdout.strip().splitlines()[-25:]),
        "dashboard_stdout": dashboard_result.stdout.strip(),
        "dashboard_stderr": dashboard_result.stderr.strip(),
    }


class DashboardRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(REPORTS_DIR), **kwargs)

    def do_GET(self):
        if self.path.startswith("/run-evaluation"):
            self._handle_run_evaluation()
            return
        if self.path in ("", "/"):
            self.path = "/dashboard.html"
        super().do_GET()

    def _handle_run_evaluation(self):
        if not _run_lock.acquire(blocking=False):
            self._send_json(409, {
                "status": "busy",
                "message": "An evaluation is already running - wait for it to finish.",
            })
            return
        try:
            print("Running full evaluation (pytest tests/ -v + generate_dashboard.py)...")
            result = _run_full_evaluation()
            print(f"Done. pytest exit code: {result['pytest_exit_code']}")
            if result["status"] != "ok":
                print(result["dashboard_stderr"])
            self._send_json(200 if result["status"] == "ok" else 500, result)
        except subprocess.TimeoutExpired as exc:
            self._send_json(500, {"status": "error", "message": f"Timed out: {exc}"})
        finally:
            _run_lock.release()

    def _send_json(self, code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # keep the console clean; we print our own progress lines


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DASHBOARD_PATH.exists():
        print("No dashboard yet - running the evaluation once to create one...")
        _run_full_evaluation()

    url = f"http://127.0.0.1:{PORT}/"
    with socketserver.TCPServer(("127.0.0.1", PORT), DashboardRequestHandler) as httpd:
        print(f"NOVA Evaluation Dashboard running at {url}")
        print("Click 'Run Evaluation' on the page any time you want to re-run everything.")
        print("Press Ctrl+C here to stop the server.")
        try:
            webbrowser.open(url)
        except Exception:
            pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
