"""
Agent-to-agent (A2A) communication and cross-system validation.

test_01_routing.py's @a2a-marked tests only prove a compound request
gets routed to the PLAN/AUTO_EXECUTE path. They never check that the
individual agents actually exchange information correctly. This file
does: it proves the Leave Agent really sees the PO Agent's data (and
vice versa), and that both really see the Calendar Agent's data, by
driving the real functions and checking the real cross-system
warnings/conflicts they produce - the same three-way relationship
apply_leave()/validate_po_request()'s own docstrings describe.

All three of these cross-checks are documented as NON-BLOCKING: they
should surface a warning, not refuse the request outright. Every
test here checks both halves of that contract.
"""

from datetime import date, datetime, timedelta

import pytest


# =================================================================
# LEAVE AGENT <-> PO AGENT
# (validate_leave_request() reading PO Agent's store via
# check_po_due_conflict())
# =================================================================

@pytest.mark.a2a
def test_leave_request_warns_about_conflicting_po_due_date(isolated_stores, rag_module):
    # Raise a PO due squarely inside the leave window we're about to request.
    ok_po, _msg, po_details = rag_module.apply_po(
        "kishore", "Dell", "engineering",
        [{"name": "laptop", "quantity": 1, "unit_price": 1000}],
        due_date=date(2026, 8, 31),
    )
    assert ok_po is True

    ok, errors, warnings, info = rag_module.validate_leave_request(
        "kishore", "sick", date(2026, 8, 31), date(2026, 8, 31)
    )

    # Non-blocking: the leave itself is still approvable...
    assert ok is True
    # ...but the PO Agent's data really made it into the Leave
    # Agent's warnings and structured info, referencing the actual
    # vendor from the PO record.
    assert info["po_due_conflicts"], "PO Agent's due-date data never reached the Leave Agent"
    assert info["po_due_conflicts"][0]["vendor"] == "Dell"
    assert any("PO due" in w for w in warnings)


@pytest.mark.a2a
def test_leave_request_has_no_po_warning_when_po_due_date_is_outside_window(
    isolated_stores, rag_module
):
    rag_module.apply_po(
        "kishore", "Dell", "engineering",
        [{"name": "laptop", "quantity": 1, "unit_price": 1000}],
        due_date=date(2026, 12, 1),  # well outside the leave window below
    )

    ok, errors, warnings, info = rag_module.validate_leave_request(
        "kishore", "sick", date(2026, 8, 31), date(2026, 8, 31)
    )

    assert ok is True
    assert info["po_due_conflicts"] == []
    assert not any("PO due" in w for w in warnings)


@pytest.mark.a2a
def test_leave_request_ignores_rejected_pos_for_due_date_conflict(
    isolated_stores, rag_module
):
    ok_po, _msg, po_details = rag_module.apply_po(
        "kishore", "Dell", "engineering",
        [{"name": "laptop", "quantity": 1, "unit_price": 1000}],
        due_date=date(2026, 8, 31),
    )
    rag_module.reject_po_request(po_details["record"]["id"])

    ok, errors, warnings, info = rag_module.validate_leave_request(
        "kishore", "sick", date(2026, 8, 31), date(2026, 8, 31)
    )

    # A rejected PO is dead - it shouldn't still block/warn about a
    # due date nobody's actually going to be held to.
    assert info["po_due_conflicts"] == []


# =================================================================
# PO AGENT <-> LEAVE AGENT
# (validate_po_request() reading Leave Agent's store via
# check_leave_conflict_for_date())
# =================================================================

@pytest.mark.a2a
def test_po_request_warns_about_requester_being_on_leave_on_due_date(
    isolated_stores, mock_mail, rag_module
):
    rag_module.LEAVE_APPROVER_EMAIL = "manager@test.local"
    ok_leave, _msg, leave_details = rag_module.apply_leave(
        "kishore", "sick", date(2026, 8, 31), date(2026, 8, 31)
    )
    assert ok_leave is True

    ok, errors, warnings, info = rag_module.validate_po_request(
        "kishore", "Dell", "engineering",
        [{"name": "laptop", "quantity": 1, "unit_price": 1000}],
        due_date=date(2026, 8, 31),
    )

    # Non-blocking: the PO itself is still approvable...
    assert ok is True
    # ...but the Leave Agent's pending request really reached the PO
    # Agent's cross-check.
    assert info["leave_conflicts"], "Leave Agent's data never reached the PO Agent"
    assert info["leave_conflicts"][0]["leave_type"] == "sick"
    assert any("leave scheduled" in w for w in warnings)


@pytest.mark.a2a
def test_po_request_leave_conflict_disappears_once_leave_is_rejected(
    isolated_stores, mock_mail, rag_module
):
    rag_module.LEAVE_APPROVER_EMAIL = "manager@test.local"
    ok_leave, _msg, leave_details = rag_module.apply_leave(
        "kishore", "sick", date(2026, 8, 31), date(2026, 8, 31)
    )
    rag_module.reject_leave_request(leave_details["record"]["id"])

    ok, errors, warnings, info = rag_module.validate_po_request(
        "kishore", "Dell", "engineering",
        [{"name": "laptop", "quantity": 1, "unit_price": 1000}],
        due_date=date(2026, 8, 31),
    )

    # A rejected leave request shouldn't still make the PO Agent
    # think the requester is unavailable that day.
    assert info["leave_conflicts"] == []


@pytest.mark.a2a
def test_po_request_has_no_leave_warning_without_a_due_date(isolated_stores, mock_mail, rag_module):
    rag_module.LEAVE_APPROVER_EMAIL = "manager@test.local"
    rag_module.apply_leave("kishore", "sick", date(2026, 8, 31), date(2026, 8, 31))

    # No due_date given at all - there's nothing to cross-check against.
    ok, errors, warnings, info = rag_module.validate_po_request(
        "kishore", "Dell", "engineering",
        [{"name": "laptop", "quantity": 1, "unit_price": 1000}],
    )

    assert ok is True
    assert info["due_date"] is None
    assert not any("leave scheduled" in w for w in warnings)


# =================================================================
# LEAVE AGENT <-> CALENDAR AGENT
# (validate_leave_request() reading a real .ics file via
# check_group_availability())
# =================================================================

@pytest.mark.a2a
def test_leave_request_warns_about_existing_calendar_event(mock_calendar, isolated_stores, rag_module):
    # The fixture .ics has a "Sprint Planning" event on 2026-09-01.
    ok, errors, warnings, info = rag_module.validate_leave_request(
        "me", "annual", date(2026, 9, 1), date(2026, 9, 1)
    )

    assert ok is True
    assert info["calendar_conflicts"], "Calendar Agent's event never reached the Leave Agent"
    assert any("Sprint Planning" in c for c in info["calendar_conflicts"])
    assert any("calendar event" in w for w in warnings)


@pytest.mark.a2a
def test_leave_request_has_no_calendar_warning_on_a_free_day(mock_calendar, isolated_stores, rag_module):
    ok, errors, warnings, info = rag_module.validate_leave_request(
        "me", "annual", date(2026, 9, 10), date(2026, 9, 10)
    )

    assert ok is True
    assert info["calendar_conflicts"] == []


# =================================================================
# PO AGENT <-> CALENDAR AGENT
# (validate_po_request() reading a real .ics file via
# check_group_availability())
# =================================================================

@pytest.mark.a2a
def test_po_request_warns_about_meeting_on_due_date(mock_calendar, isolated_stores, rag_module):
    ok, errors, warnings, info = rag_module.validate_po_request(
        "me", "Dell", "engineering",
        [{"name": "laptop", "quantity": 1, "unit_price": 1000}],
        due_date=date(2026, 9, 1),  # same day as the fixture's Sprint Planning event
    )

    assert ok is True
    assert info.get("calendar_conflicts"), "Calendar Agent's event never reached the PO Agent"
    assert any("meeting" in w.lower() for w in warnings)


# =================================================================
# THREE-WAY: all three agents in a single request
# =================================================================

@pytest.mark.a2a
def test_leave_request_surfaces_conflicts_from_both_po_and_calendar_at_once(
    mock_calendar, isolated_stores, rag_module
):
    rag_module.apply_po(
        "me", "Dell", "engineering",
        [{"name": "laptop", "quantity": 1, "unit_price": 1000}],
        due_date=date(2026, 9, 1),
    )

    ok, errors, warnings, info = rag_module.validate_leave_request(
        "me", "annual", date(2026, 9, 1), date(2026, 9, 1)
    )

    assert ok is True
    assert info["po_due_conflicts"], "missing PO Agent conflict"
    assert info["calendar_conflicts"], "missing Calendar Agent conflict"
    # Both distinct cross-system warnings must both be present, not
    # just whichever check happened to run last overwriting the other.
    assert any("PO due" in w for w in warnings)
    assert any("calendar event" in w for w in warnings)
