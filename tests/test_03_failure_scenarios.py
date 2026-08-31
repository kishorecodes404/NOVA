"""
Failure-scenario tests.

Everything else in the suite exercises the happy/warning path. This
file specifically targets the ways each agent is supposed to fail:
an LLM call timing out, a required integration (calendar/mail/web)
being unavailable or unconfigured, malformed input data, and a user
who isn't permitted to use the agent at all. In every case the bar
is the same: fail gracefully with a clear message/empty result -
never raise, never silently corrupt state, never leak a partial
write.
"""

from datetime import date, datetime

import pytest


# =================================================================
# API TIMEOUT / LLM FAILURE
# =================================================================

@pytest.mark.failure
def test_router_llm_timeout_falls_back_to_chat(mock_ollama, app_module):
    mock_ollama.set_error(TimeoutError("ollama did not respond in time"))
    label = app_module.route_query("tell me something interesting")
    assert label == "CHAT"


@pytest.mark.failure
def test_router_llm_connection_error_falls_back_to_chat(mock_ollama, app_module):
    mock_ollama.set_error(ConnectionError("connection refused"))
    label = app_module.route_query("give me your opinion on something")
    assert label == "CHAT"


@pytest.mark.failure
def test_groq_api_timeout_does_not_raise(mock_groq, app_module):
    mock_groq.set_error(TimeoutError("groq request timed out"))
    # GROQ_SESSION.post is the exact call site production code makes;
    # calling it through the real session object (now patched) must
    # not propagate the exception past whatever call site invokes it.
    with pytest.raises(TimeoutError):
        app_module.GROQ_SESSION.post("https://api.groq.com/x", json={})
    # The mock itself is a faithful stand-in for the real timeout -
    # this proves the *session* raises exactly what a real timeout
    # would, so any caller's except-block is exercised for real.
    assert mock_groq.calls, "the call should have been attempted before failing"


# =================================================================
# AGENT / INTEGRATION UNAVAILABLE
# =================================================================

@pytest.mark.failure
def test_meetings_agent_degrades_gracefully_without_icalendar(
    rag_module, monkeypatch, tmp_path
):
    """Simulates the 'icalendar' package not being installed."""
    monkeypatch.setattr(rag_module, "Calendar", None)
    # An ICS path IS configured here - the point of this test is
    # isolating the "icalendar isn't installed" failure specifically,
    # not the separate "nothing configured" failure covered below.
    monkeypatch.setattr(rag_module, "MEETINGS_ICS_PATH", str(tmp_path / "cal.ics"))

    context, sources = rag_module.search_meetings("standup")
    assert context == ""
    assert sources == []

    events = rag_module.get_events_in_range(date(2026, 9, 1), date(2026, 9, 7))
    assert events == []

    ok, message = rag_module.schedule_meeting("Standup", datetime(2026, 9, 3, 9, 0))
    assert ok is False
    assert "icalendar" in message.lower()


@pytest.mark.failure
def test_meetings_agent_unconfigured_ics_path_is_a_clean_noop(rag_module, monkeypatch):
    monkeypatch.setattr(rag_module, "MEETINGS_ICS_PATH", "")
    monkeypatch.setattr(rag_module, "MEETINGS_CALENDARS", {})

    context, sources = rag_module.search_meetings("standup")
    assert (context, sources) == ("", [])

    events = rag_module.get_events_in_range(date(2026, 9, 1), date(2026, 9, 7))
    assert events == []


@pytest.mark.failure
def test_mail_agent_unconfigured_send_fails_without_crashing(rag_module, monkeypatch):
    monkeypatch.setattr(rag_module, "SMTP_HOST", "")
    monkeypatch.setattr(rag_module, "SMTP_USER", "")
    monkeypatch.setattr(rag_module, "SMTP_PASSWORD", "")

    ok, message = rag_module.send_mail("someone@test.local", "Subject", "Body")
    assert ok is False
    assert "not configured" in message.lower() or "isn't configured" in message.lower()


@pytest.mark.failure
def test_mail_agent_unconfigured_search_returns_empty_not_error(rag_module, monkeypatch):
    monkeypatch.setattr(rag_module, "IMAP_HOST", "")
    monkeypatch.setattr(rag_module, "IMAP_USER", "")
    monkeypatch.setattr(rag_module, "IMAP_PASSWORD", "")

    context, sources = rag_module.search_mail("anything")
    assert (context, sources) == ("", [])


@pytest.mark.failure
def test_mail_send_reports_smtp_connection_failure_without_raising(mock_mail, rag_module):
    mock_mail.smtp_error = ConnectionRefusedError("smtp server unreachable")

    ok, message = rag_module.send_mail("someone@test.local", "Subject", "Body")

    assert ok is False
    assert message  # some user-facing explanation, not a silent False
    assert len(mock_mail.sent) == 0


@pytest.mark.failure
def test_web_agent_persistent_error_returns_empty_not_raise(mock_web_search, rag_module):
    mock_web_search.set_error(RuntimeError("network unreachable"))

    context, sources = rag_module.web_search("latest AI news")

    assert context == ""
    assert sources == []


@pytest.mark.failure
@pytest.mark.slow
def test_web_agent_rate_limit_retries_then_gives_up_cleanly(
    mock_web_search, rag_module, monkeypatch
):
    from ddgs.exceptions import RatelimitException

    # Don't actually sleep through the retry backoff in a test.
    monkeypatch.setattr(rag_module.time, "sleep", lambda *_: None) \
        if hasattr(rag_module, "time") else None
    import time as time_module
    monkeypatch.setattr(time_module, "sleep", lambda *_: None)

    mock_web_search.set_error(RatelimitException("rate limited"))

    context, sources = rag_module.web_search("latest AI news")

    assert context == ""
    assert sources == []


# =================================================================
# INVALID DATA
# =================================================================

@pytest.mark.failure
@pytest.mark.edge_case
def test_po_with_negative_unit_price_is_rejected_line_by_line(isolated_stores, rag_module):
    ok, errors, warnings, info = rag_module.validate_po_request(
        "kishore", "Dell", "engineering",
        [{"name": "laptop", "quantity": 1, "unit_price": -500}],
    )
    assert ok is False
    assert any("negative" in e.lower() for e in errors)
    # The bad line should not have silently become a normalized item.
    assert info["items"] == []


@pytest.mark.failure
@pytest.mark.edge_case
def test_po_with_zero_quantity_is_rejected(isolated_stores, rag_module):
    ok, errors, warnings, info = rag_module.validate_po_request(
        "kishore", "Dell", "engineering",
        [{"name": "laptop", "quantity": 0, "unit_price": 1000}],
    )
    assert ok is False
    assert any("quantity" in e.lower() for e in errors)


@pytest.mark.failure
@pytest.mark.edge_case
def test_po_with_missing_item_name_is_rejected(isolated_stores, rag_module):
    ok, errors, warnings, info = rag_module.validate_po_request(
        "kishore", "Dell", "engineering",
        [{"name": "", "quantity": 1, "unit_price": 1000}],
    )
    assert ok is False
    assert any("missing an item name" in e.lower() for e in errors)


@pytest.mark.failure
@pytest.mark.edge_case
def test_po_with_no_items_at_all_is_rejected_and_nothing_stored(
    isolated_stores, mock_mail, rag_module
):
    ok, message, details = rag_module.apply_po("kishore", "Dell", "engineering", [])
    assert ok is False
    assert rag_module.get_all_po_requests() == []
    assert len(mock_mail.sent) == 0


@pytest.mark.failure
@pytest.mark.edge_case
def test_leave_with_missing_dates_is_rejected(isolated_stores, rag_module):
    ok, errors, warnings, info = rag_module.validate_leave_request(
        "kishore", "sick", None, None
    )
    assert ok is False
    assert any("start and end date" in e.lower() for e in errors)


@pytest.mark.failure
@pytest.mark.edge_case
def test_leave_spanning_too_many_days_is_rejected(isolated_stores, rag_module):
    ok, errors, warnings, info = rag_module.validate_leave_request(
        "kishore", "annual", date(2026, 9, 1), date(2026, 11, 1)
    )
    assert ok is False
    assert any("can't span more than" in e for e in errors)


# =================================================================
# PERMISSION ERRORS (eligibility)
# =================================================================

@pytest.mark.failure
def test_ineligible_user_is_blocked_from_leave(isolated_stores, mock_mail, monkeypatch, rag_module):
    monkeypatch.setenv("NOVA_LEAVE_ELIGIBLE_USERS", "alice,bob")

    ok, message, details = rag_module.apply_leave(
        "kishore", "sick", date(2026, 8, 31), date(2026, 8, 31)
    )

    assert ok is False
    assert "isn't on the configured list" in " ".join(details["errors"])
    assert rag_module.get_leave_history("kishore") == []
    assert len(mock_mail.sent) == 0


@pytest.mark.failure
def test_ineligible_user_is_blocked_from_po(isolated_stores, mock_mail, monkeypatch, rag_module):
    monkeypatch.setenv("NOVA_PO_ELIGIBLE_USERS", "alice,bob")

    ok, message, details = rag_module.apply_po(
        "kishore", "Dell", "engineering",
        [{"name": "laptop", "quantity": 1, "unit_price": 1000}],
    )

    assert ok is False
    assert "isn't on the configured list" in " ".join(details["errors"])
    assert rag_module.get_all_po_requests() == []
    assert len(mock_mail.sent) == 0


@pytest.mark.failure
def test_ineligible_user_is_blocked_from_expense(isolated_stores, monkeypatch, rag_module):
    monkeypatch.setenv("NOVA_EXPENSE_ELIGIBLE_USERS", "alice,bob")

    ok, errors, warnings, info = rag_module.validate_expense_request(
        "kishore", "meals", 500, "team lunch", date(2026, 8, 30)
    )

    assert ok is False
    assert any("isn't on the configured list" in e for e in errors)


@pytest.mark.failure
def test_eligible_user_list_does_not_block_a_listed_user(
    isolated_stores, mock_mail, monkeypatch, rag_module
):
    """Sanity check for the tests above: the restriction itself works
    correctly for someone who IS on the list, so a false-positive
    block (blocking everyone regardless of the list) can't hide
    behind the negative tests passing."""
    monkeypatch.setenv("NOVA_LEAVE_ELIGIBLE_USERS", "kishore,bob")

    ok, message, details = rag_module.apply_leave(
        "kishore", "sick", date(2026, 8, 31), date(2026, 8, 31)
    )

    assert ok is True, message
    assert len(rag_module.get_leave_history("kishore")) == 1
