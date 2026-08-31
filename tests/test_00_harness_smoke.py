"""
Harness smoke tests - not part of the real suite's coverage, just
proof that every fixture in conftest.py actually does what it
claims before we build ~10 more test files on top of it.
"""

import pytest


def test_app_and_rag_import(app_module, rag_module):
    assert hasattr(app_module, "route_query")
    assert hasattr(rag_module, "validate_leave_request")


def test_fast_path_routing_needs_no_mocks(app_module):
    # Fast regex/keyword paths never touch the LLM at all.
    assert app_module.route_query("what's on my calendar today") == "MEETINGS"
    assert app_module.route_query("check my document for the policy") == "DOCUMENT"


def test_session_state_resets_between_tests(app_module):
    import streamlit as st
    assert st.session_state.get("last_route") is None
    st.session_state["last_route"] = "MAIL"


def test_session_state_actually_reset(app_module):
    # If the previous test's write leaked, this fails.
    import streamlit as st
    assert st.session_state.get("last_route") is None


def test_mock_ollama_router_returns_scripted_label(mock_ollama, app_module):
    mock_ollama.set_response("WEB")
    # A query with no fast-path keyword falls through to the LLM router.
    label = app_module.route_query("tell me something interesting")
    assert label == "WEB"
    assert len(mock_ollama.calls) == 1


def test_mock_ollama_error_injection_falls_back_to_chat(mock_ollama, app_module):
    mock_ollama.set_error(TimeoutError("ollama down"))
    label = app_module.route_query("tell me something interesting")
    # route_query()'s except-branch returns CHAT on router failure.
    assert label == "CHAT"


def test_isolated_leave_store_is_empty_and_writable(isolated_stores, rag_module):
    from datetime import date
    balances = rag_module.get_leave_balances("kishore")
    assert isinstance(balances, dict)
    ok, message, details = rag_module.apply_leave(
        "kishore", "sick", date(2026, 9, 10), date(2026, 9, 10), reason="test"
    )
    assert ok is True, message
    history = rag_module.get_leave_history("kishore")
    assert len(history) == 1


def test_mock_mail_search_returns_scripted_inbox(mock_mail, rag_module):
    mock_mail.set_inbox([
        {"id": 1, "subject": "Budget approval needed", "from": "sarah@test.local"},
        {"id": 2, "subject": "Lunch?", "from": "bob@test.local"},
    ])
    context, sources = rag_module.search_mail("budget", require_keyword_match=True)
    assert "Budget approval needed" in context
    assert any("Budget approval needed" in s for s in sources)


def test_mock_mail_send_records_message(mock_mail, rag_module):
    ok, message = rag_module.send_mail(
        "someone@test.local", "Test subject", "Test body"
    )
    assert ok is True
    assert len(mock_mail.sent) == 1
    assert mock_mail.sent[0]["subject"] == "Test subject"


def test_mock_web_search_returns_scripted_results(mock_web_search, rag_module):
    mock_web_search.set_results([
        {"title": "Example", "href": "https://example.com", "body": "Example body"}
    ])
    context, sources = rag_module.web_search("some query")
    assert "Example" in context
    assert sources


def _backdate_leave_request(rag_module, request_id, days_ago):
    """Test helper: rewrites a stored leave request's requested_at
    to `days_ago` days in the past, so staleness logic can be
    exercised without waiting for real time to pass."""
    from datetime import datetime, timedelta

    store = rag_module._load_leave_store()
    for request in store["requests"]:
        if request["id"] == request_id:
            request["requested_at"] = (
                datetime.now() - timedelta(days=days_ago)
            ).isoformat(timespec="seconds")
    assert rag_module._save_leave_store(store)


def test_recommendation_evidence_flags_stale_request_with_matching_preference(
    isolated_stores, rag_module, app_module
):
    from datetime import date

    # Two "casual" requests -> a real repeat pattern (>=2). Backdate
    # the first past the staleness threshold; leave the second
    # fresh, to check both branches in one pass.
    ok1, message1, details1 = rag_module.apply_leave(
        "me", "casual", date(2026, 9, 10), date(2026, 9, 10), reason="test"
    )
    assert ok1 is True, message1
    stale_id = details1["record"]["id"]
    _backdate_leave_request(rag_module, stale_id, days_ago=5)

    ok2, message2, _ = rag_module.apply_leave(
        "me", "casual", date(2026, 9, 21), date(2026, 9, 21), reason="test"
    )
    assert ok2 is True, message2

    context, sources = app_module._gather_recommendation_evidence()

    # The stale one is flagged as a genuine recommendation, with the
    # matching preference attached inline - never as a standalone
    # "INFERRED PREFERENCES" fact with nothing to anchor it to.
    assert "MAY BE WORTH A FOLLOW-UP" in context
    assert "your most-requested leave type" in context
    assert any("may need a follow-up" in s for s in sources)

    # The fresh one is present as status only, with no preference
    # note attached (it's not stale, so nothing to anchor it to).
    assert "no action needed from you yet" in context


def test_recommendation_evidence_no_preference_note_without_a_repeat(
    isolated_stores, rag_module, app_module
):
    from datetime import date

    # Only one request ever made -> no repeat pattern, so even a
    # stale request gets no preference note.
    ok, message, details = rag_module.apply_leave(
        "me", "casual", date(2026, 9, 10), date(2026, 9, 10), reason="test"
    )
    assert ok is True, message
    _backdate_leave_request(rag_module, details["record"]["id"], days_ago=5)

    context, _ = app_module._gather_recommendation_evidence()

    assert "MAY BE WORTH A FOLLOW-UP" in context
    assert "your most-requested leave type" not in context


def test_recommendation_evidence_history_is_context_only(
    isolated_stores, rag_module, app_module
):
    from datetime import date

    ok, message, _ = rag_module.apply_leave(
        "me", "sick", date(2026, 9, 10), date(2026, 9, 10), reason="test"
    )
    assert ok is True, message

    context, _ = app_module._gather_recommendation_evidence()

    assert "RECENT LEAVE HISTORY (context only" in context
    assert "sick" in context


def test_mock_calendar_reads_fixture_ics(mock_calendar, rag_module):
    from datetime import date, timedelta
    # The fixture event is 2026-09-01 09:00 UTC; get_events_in_range()
    # converts to local time before filtering, so check a small
    # window around the date rather than assuming no timezone shift.
    events = rag_module.get_events_in_range(
        date(2026, 8, 31), date(2026, 9, 2)
    )
    assert any("Sprint Planning" in e for e in events)