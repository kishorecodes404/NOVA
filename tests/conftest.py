"""
Shared fixtures for the NOVA agent test suite.

Design principle: app.py and rag.py are written to run inside
`streamlit run`, but every function under test (route_query,
validate_*_request, apply_*, search_*, send_mail, schedule_meeting,
planning_agent.*, autonomous_executor.*) is a plain, importable
function/module that works fine under a bare `import` - confirmed by
hand before writing this file. So instead of standing up a real
Streamlit runtime, we:

    1. Set a dummy GEMINI_API_KEY before rag.py/app.py are imported
       (both read it at import time via load_dotenv()/get_api_key()).
    2. Import app/rag/planning_agent/autonomous_executor exactly ONCE
       per test session (they're heavy - chromadb, streamlit, etc.)
       and hand the same module objects to every test.
    3. Reset st.session_state between tests, since route_query() and
       friends read/write it (e.g. "last_route" for follow-up
       inheritance) and tests must not leak state into each other.
    4. Monkeypatch the exact external call sites - OLLAMA_SESSION,
       GROQ_SESSION, genai client, imaplib.IMAP4_SSL, smtplib.SMTP,
       DDGS, and the rag.py module-level config constants (store
       paths, budget limits, item catalog, eligible-user lists) -
       so every test is deterministic and touches no real network,
       inbox, calendar file, or vector DB.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------
# 0. Make the app importable, and set env vars BEFORE first import
# ---------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
# Make sure no real mail/calendar config leaks in from the
# developer's own shell into the test run.
for _leaky_var in (
    "NOVA_IMAP_HOST", "NOVA_IMAP_USER", "NOVA_IMAP_PASSWORD",
    "NOVA_SMTP_HOST", "NOVA_SMTP_USER", "NOVA_SMTP_PASSWORD",
    "NOVA_MEETINGS_ICS_PATH", "NOVA_MEETINGS_ICS_PATHS",
):
    os.environ.pop(_leaky_var, None)


@pytest.fixture(scope="session")
def rag_module():
    import rag
    return rag


@pytest.fixture(scope="session")
def app_module():
    import app
    return app


@pytest.fixture(scope="session")
def planning_agent_module():
    import planning_agent
    return planning_agent


@pytest.fixture(scope="session")
def autonomous_executor_module():
    import autonomous_executor
    return autonomous_executor


# ---------------------------------------------------------------
# 1. Streamlit session_state isolation
# ---------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_session_state(app_module):
    """
    route_query() reads/writes st.session_state (e.g. "last_route",
    for follow-up-question inheritance). Without this, a route
    decided in one test could silently change the outcome of the
    next one.
    """
    import streamlit as st
    st.session_state.clear()
    yield
    st.session_state.clear()


# ---------------------------------------------------------------
# 2. Mocked LLM call sites (Ollama router/extraction/answer, Groq,
#    Gemini) - deterministic, no network, no local model required.
# ---------------------------------------------------------------

class OllamaRouterStub:
    """
    Patches app.OLLAMA_SESSION.post so ANY Ollama call (router
    fallback, field extraction, follow-up generation, local answer
    streaming) returns a scripted response instead of hitting a
    real model.

    Usage in a test:
        mock_ollama.set_response("MEETINGS")          # plain classify call
        mock_ollama.set_json_response({"leave_type": "sick", ...})
        mock_ollama.set_error(TimeoutError("slow"))   # failure injection
    """

    def __init__(self, monkeypatch, app_module):
        self._monkeypatch = monkeypatch
        self._app = app_module
        self._next_text = "CHAT"
        self._error = None
        self.calls = []

    def _fake_post(self, url, json=None, timeout=None, **kwargs):
        self.calls.append({"url": url, "json": json, "timeout": timeout})

        if self._error is not None:
            raise self._error

        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json = MagicMock(return_value={"response": self._next_text})
        return response

    def install(self):
        self._monkeypatch.setattr(
            self._app.OLLAMA_SESSION, "post", self._fake_post
        )
        return self

    def set_response(self, text: str):
        """Next call's raw '.response' text, e.g. a route label."""
        self._error = None
        self._next_text = text

    def set_json_response(self, obj: dict):
        """Next call's '.response' text, JSON-encoded (field extraction)."""
        self._error = None
        self._next_text = json.dumps(obj)

    def set_error(self, exc: Exception):
        """Every subsequent call raises `exc` (timeout/connection failure)."""
        self._error = exc


@pytest.fixture
def mock_ollama(monkeypatch, app_module):
    return OllamaRouterStub(monkeypatch, app_module).install()


class GroqStub:
    """Patches app.GROQ_SESSION.post for the Groq chat-completions call."""

    def __init__(self, monkeypatch, app_module):
        self._monkeypatch = monkeypatch
        self._app = app_module
        self._next_text = "OK"
        self._error = None
        self.calls = []

    def _fake_post(self, url, json=None, headers=None, timeout=None, **kwargs):
        self.calls.append({"url": url, "json": json, "timeout": timeout})

        if self._error is not None:
            raise self._error

        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json = MagicMock(
            return_value={
                "choices": [{"message": {"content": self._next_text}}]
            }
        )
        return response

    def install(self):
        self._monkeypatch.setattr(
            self._app.GROQ_SESSION, "post", self._fake_post
        )
        return self

    def set_response(self, text: str):
        self._error = None
        self._next_text = text

    def set_error(self, exc: Exception):
        self._error = exc


@pytest.fixture
def mock_groq(monkeypatch, app_module):
    return GroqStub(monkeypatch, app_module).install()


# ---------------------------------------------------------------
# 3. Mocked mail (IMAP read / SMTP send)
# ---------------------------------------------------------------

class FakeIMAPConnection:
    """
    Minimal stand-in for imaplib.IMAP4_SSL, faithful to exactly the
    calls search_mail() makes: login(), select(), search(None,
    "ALL"), fetch(id_set, "(BODY.PEEK[HEADER...])"), and per-message
    full fetch. Messages are supplied as a list of dicts:
        {"id": b"1", "subject": "...", "from": "...", "date": "...",
         "body": "..."}
    """

    def __init__(self, messages):
        self._messages = messages
        self.logged_in = False
        self.selected_folder = None

    def login(self, user, password):
        self.logged_in = True
        return "OK", [b"Logged in"]

    def select(self, folder, readonly=True):
        self.selected_folder = folder
        return "OK", [str(len(self._messages)).encode()]

    def search(self, charset, criteria):
        ids = b" ".join(m["id"] for m in self._messages)
        return "OK", [ids]

    def fetch(self, id_set, spec):
        import email.message

        wanted_ids = set(id_set.split(b","))
        results = []

        for m in self._messages:
            if m["id"] not in wanted_ids:
                continue

            msg = email.message.EmailMessage()
            msg["Subject"] = m["subject"]
            msg["From"] = m["from"]
            msg["Date"] = m.get("date", "")

            if "HEADER" in spec:
                results.append((m["id"] + b" (HEADER)", bytes(msg)))
            else:
                msg.set_content(m.get("body", ""))
                results.append((m["id"] + b" (BODY)", bytes(msg)))

        return "OK", results

    def close(self):
        pass

    def logout(self):
        pass


@pytest.fixture
def mock_mail(monkeypatch, rag_module):
    """
    Configures rag.py's IMAP_* / SMTP_* module constants so
    search_mail()/send_mail() think mail IS configured, and patches
    imaplib.IMAP4_SSL / smtplib.SMTP so no real connection is made.

    Returns a SimpleNamespace with:
        .set_inbox(messages)   -- messages search_mail() will "find"
        .sent                  -- list of dicts, one per send_mail() call
        .smtp_error             -- set to an Exception to make SMTP fail
    """

    monkeypatch.setattr(rag_module, "IMAP_HOST", "imap.test.local")
    monkeypatch.setattr(rag_module, "IMAP_USER", "nova@test.local")
    monkeypatch.setattr(rag_module, "IMAP_PASSWORD", "test-pass")
    monkeypatch.setattr(rag_module, "IMAP_FOLDER", "INBOX")

    monkeypatch.setattr(rag_module, "SMTP_HOST", "smtp.test.local")
    monkeypatch.setattr(rag_module, "SMTP_PORT", 587)
    monkeypatch.setattr(rag_module, "SMTP_USER", "nova@test.local")
    monkeypatch.setattr(rag_module, "SMTP_PASSWORD", "test-pass")

    state = SimpleNamespace(messages=[], sent=[], smtp_error=None)

    def fake_imap4_ssl(host):
        return FakeIMAPConnection(state.messages)

    monkeypatch.setattr(rag_module.imaplib, "IMAP4_SSL", fake_imap4_ssl)

    def _extract_header(message_string, header_name):
        for line in message_string.splitlines():
            if line.lower().startswith(header_name.lower() + ":"):
                return line.split(":", 1)[1].strip()
        return None

    class FakeSMTP:
        def __init__(self, host, port, timeout=None):
            if state.smtp_error is not None:
                raise state.smtp_error

        def starttls(self):
            pass

        def login(self, user, password):
            pass

        def sendmail(self, from_addr, to_addrs, message_string):
            state.sent.append(
                {
                    "to": to_addrs,
                    "from": from_addr,
                    "raw_message": message_string,
                    "subject": _extract_header(message_string, "Subject"),
                }
            )

        def quit(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(rag_module.smtplib, "SMTP", FakeSMTP)
    monkeypatch.setattr(rag_module.smtplib, "SMTP_SSL", FakeSMTP)

    def set_inbox(messages):
        # messages: list of {"id": int, "subject", "from", "date"?, "body"?}
        state.messages = [
            {**m, "id": str(m["id"]).encode()} for m in messages
        ]

    state.set_inbox = set_inbox
    return state


# ---------------------------------------------------------------
# 4. Mocked web search (DDGS)
# ---------------------------------------------------------------

@pytest.fixture
def mock_web_search(monkeypatch, rag_module):
    """
    Patches rag.DDGS so web_search() returns scripted results with
    no real HTTP call. Usage:
        mock_web_search.set_results([
            {"title": "...", "href": "https://...", "body": "..."}
        ])
        mock_web_search.set_error(Exception("rate limited"))
    """

    state = SimpleNamespace(results=[], error=None)

    class FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def text(self, query, max_results=6, timelimit=None):
            if state.error is not None:
                raise state.error
            return list(state.results[:max_results])

    monkeypatch.setattr(rag_module, "DDGS", FakeDDGS)

    def set_results(results):
        state.results = results

    def set_error(exc):
        state.error = exc

    state.set_results = set_results
    state.set_error = set_error
    return state


# ---------------------------------------------------------------
# 5. Isolated Leave / PO / Expense JSON stores + calendar (ICS)
# ---------------------------------------------------------------

@pytest.fixture
def isolated_stores(monkeypatch, rag_module, tmp_path):
    """
    Redirects every JSON-backed store (leave/PO/expense) to a fresh
    tmp_path per test, and resets the module-level config that's
    otherwise fixed at import time (budget limits, item catalog,
    max leave span, eligible-user sets) to permissive, known
    defaults. Individual tests override what they need via
    monkeypatch.setattr(rag, "PO_BUDGET_LIMITS", {...}) etc., or via
    monkeypatch.setenv(...) for the live-read env vars
    (NOVA_*_ELIGIBLE_USERS).
    """

    monkeypatch.setattr(rag_module, "LEAVE_STORE_PATH", str(tmp_path / "leave_store.json"))
    monkeypatch.setattr(rag_module, "PO_STORE_PATH", str(tmp_path / "po_store.json"))
    monkeypatch.setattr(rag_module, "EXPENSE_STORE_PATH", str(tmp_path / "expense_store.json"))

    # Permissive defaults - no eligibility restriction, generous caps -
    # so a "normal" test doesn't accidentally hit a business rule it
    # wasn't trying to test. Tests targeting the rule itself
    # override these explicitly.
    monkeypatch.delenv("NOVA_LEAVE_ELIGIBLE_USERS", raising=False)
    monkeypatch.delenv("NOVA_PO_ELIGIBLE_USERS", raising=False)
    monkeypatch.delenv("NOVA_EXPENSE_ELIGIBLE_USERS", raising=False)
    monkeypatch.setattr(rag_module, "LEAVE_MAX_SPAN_DAYS", 30)
    monkeypatch.setattr(rag_module, "PO_BUDGET_LIMITS", {})
    monkeypatch.setattr(rag_module, "PO_ITEM_CATALOG", {})

    return SimpleNamespace(tmp_path=tmp_path)


@pytest.fixture
def fixture_ics_path(tmp_path):
    """A minimal, valid single-event ICS file for meetings tests."""
    ics_content = (
        "BEGIN:VCALENDAR\r\n"
        "VERSION:2.0\r\n"
        "BEGIN:VEVENT\r\n"
        "UID:test-event-1@nova.local\r\n"
        "SUMMARY:Sprint Planning\r\n"
        "DTSTART:20260901T090000Z\r\n"
        "DTEND:20260901T100000Z\r\n"
        "END:VEVENT\r\n"
        "END:VCALENDAR\r\n"
    )
    path = tmp_path / "test_calendar.ics"
    path.write_text(ics_content, encoding="utf-8")
    return path


@pytest.fixture
def mock_calendar(monkeypatch, rag_module, fixture_ics_path):
    """
    Points the "me" calendar at the fixture ICS file. MEETINGS_ICS_PATH
    alone isn't enough - MEETINGS_CALENDARS (the dict every lookup
    function actually reads: get_events_in_range, search_meetings,
    check_group_availability, ...) is built from it once at import
    time via _parse_user_calendar_map(), so it must be patched too.
    """
    monkeypatch.setattr(rag_module, "MEETINGS_ICS_PATH", str(fixture_ics_path))
    monkeypatch.setattr(
        rag_module,
        "MEETINGS_CALENDARS",
        {**rag_module.MEETINGS_CALENDARS, "me": str(fixture_ics_path)},
    )
    return fixture_ics_path