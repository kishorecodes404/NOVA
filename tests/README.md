# NOVA Agent Evaluation & Testing Framework

## Status: Step 2 complete (harness scaffolding)

## Setup
```
pip install -r requirements-test.txt --break-system-packages   # if not already installed
pytest tests/ -v
```

## What's here
- `pytest.ini` — markers (`routing`, `agent`, `a2a`, `edge_case`,
  `failure`, `regression`, `live`, `slow`) used to select subsets,
  e.g. `pytest -m regression`.
- `tests/conftest.py` — every shared fixture:
  - `app_module`, `rag_module`, `planning_agent_module`,
    `autonomous_executor_module` — session-scoped imports.
  - `reset_session_state` (autouse) — clears Streamlit's
    `st.session_state` between tests (route_query() reads/writes
    `last_route` for follow-up inheritance).
  - `mock_ollama` / `mock_groq` — script or fail any LLM call
    (router, field extraction, follow-ups, local/Groq answers) with
    no real network/model needed.
  - `mock_mail` — fake IMAP inbox + fake SMTP send, faithful to the
    exact calls `search_mail()`/`send_mail()` make.
  - `mock_web_search` — fake DDGS results/errors.
  - `isolated_stores` — redirects Leave/PO/Expense JSON stores to a
    fresh `tmp_path`, resets budget limits/item catalog/eligibility
    to permissive defaults.
  - `mock_calendar` / `fixture_ics_path` — a real temp .ics file
    wired into `MEETINGS_CALENDARS`.
- `tests/test_00_harness_smoke.py` — 11 tests proving every fixture
  above actually works end-to-end. Not part of the real coverage;
  delete or keep as a canary once the full suite exists.

## Key gotchas discovered while building this (matters for every
later test file)
1. **rag.py's config is a mix of import-time and call-time reads.**
   Store paths, `PO_BUDGET_LIMITS`, `PO_ITEM_CATALOG`,
   `LEAVE_MAX_SPAN_DAYS`, and `MEETINGS_CALENDARS` are all computed
   ONCE at import from env vars — you must
   `monkeypatch.setattr(rag_module, "X", ...)` the module attribute
   directly, setting the env var alone does nothing after import.
   `get_leave_eligible_users()` (and PO/expense equivalents) IS
   read live from `os.environ` on every call, so
   `monkeypatch.setenv(...)` works fine for those.
2. **`MEETINGS_ICS_PATH` alone doesn't repoint the calendar.**
   Every real lookup (`get_events_in_range`, `search_meetings`,
   `check_group_availability`) reads `MEETINGS_CALENDARS["me"]`,
   which is built from `MEETINGS_ICS_PATH` at import time. Patch
   both (`mock_calendar` fixture does this).
3. **`icalendar` must be installed** or `rag.Calendar` is `None` and
   every calendar read silently no-ops (by the app's own design —
   graceful degradation, not a bug) — a test with no assertion
   failure but zero events is a signal to check this first.
4. **`send_mail()` uses `smtplib.SMTP(...).sendmail(...)`**, not
   `.send_message()` — the fake SMTP class must implement
   `sendmail(from_addr, to_addrs, message_string)`.
5. **Return-tuple arity varies by function** — e.g. `apply_leave()`
   returns `(success, message, details)` (3-tuple), while
   `send_mail()` returns `(success, message)` (2-tuple). Always
   check the real signature before writing an assertion.
6. **`app.py`/`rag.py` import cleanly outside `streamlit run`** (bare
   mode) — only cosmetic "missing ScriptRunContext" warnings, no
   errors — so no Streamlit test harness/AppTest is needed for
   testing agent logic directly.

## Next step
Step 3 — Routing Test Suite (`route_query()`): labeled
`(query, expected_route)` cases across every fast-path branch, the
LLM-fallback path (via `mock_ollama`), and follow-up-inheritance
behavior.