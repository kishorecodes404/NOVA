"""
Agent action/result validation.

test_01_routing.py proves route_query() sends a query to the right
agent. It never checks what that agent actually DOES once it's
picked. This file closes that gap: it calls the real agent functions
(apply_leave, approve_leave_request, apply_po, approve_po_request,
send_mail, search_mail, schedule_meeting) and asserts on their real,
observable side effects - the stored record, the balance, the
mocked-SMTP outbox, the .ics file on disk - not just the returned
success flag.

Every test here uses isolated_stores / mock_mail / mock_calendar so
nothing touches a real inbox, calendar file, or JSON store shared
between tests.
"""

from datetime import date, datetime, timedelta

import pytest


# =================================================================
# LEAVE AGENT - does apply_leave() really create the record, and
# does approve/reject really move status + touch the balance?
# =================================================================

@pytest.mark.agent
def test_apply_leave_creates_pending_record_and_notifies_approver(
    isolated_stores, mock_mail, rag_module
):
    rag_module.LEAVE_APPROVER_EMAIL = "manager@test.local"

    ok, message, details = rag_module.apply_leave(
        "kishore", "sick", date(2026, 8, 31), date(2026, 8, 31), reason="fever"
    )

    assert ok is True, message
    record = details["record"]
    assert record["status"] == "pending"
    assert record["days"] == 1
    assert details["approver_notified"] is True

    # It's really in the store, not just the returned dict.
    history = rag_module.get_leave_history("kishore")
    assert len(history) == 1
    assert history[0]["id"] == record["id"]
    assert history[0]["status"] == "pending"

    # And it really emailed the approver - not just claimed to.
    assert len(mock_mail.sent) == 1
    assert mock_mail.sent[0]["to"] == ["manager@test.local"]

    # Balance is NOT touched by the request itself (only by approval).
    assert rag_module.get_leave_balance("kishore", "sick") == 10


@pytest.mark.agent
@pytest.mark.edge_case
def test_apply_leave_with_no_approver_configured_still_saves_record(
    isolated_stores, mock_mail, rag_module
):
    rag_module.LEAVE_APPROVER_EMAIL = ""

    ok, message, details = rag_module.apply_leave(
        "kishore", "casual", date(2026, 8, 31), date(2026, 8, 31)
    )

    assert ok is True, message
    assert details["approver_notified"] is False
    assert len(mock_mail.sent) == 0
    # The request still exists even though nobody got emailed.
    assert len(rag_module.get_leave_history("kishore")) == 1


@pytest.mark.agent
@pytest.mark.failure
@pytest.mark.edge_case
def test_apply_leave_rejects_end_before_start_and_creates_no_record(
    isolated_stores, mock_mail, rag_module
):
    rag_module.LEAVE_APPROVER_EMAIL = "manager@test.local"

    ok, message, details = rag_module.apply_leave(
        "kishore", "sick", date(2026, 9, 5), date(2026, 9, 1)
    )

    assert ok is False
    assert "before the start date" in message
    # Nothing should have been written or emailed for a rejected request.
    assert rag_module.get_leave_history("kishore") == []
    assert len(mock_mail.sent) == 0


@pytest.mark.agent
@pytest.mark.failure
def test_apply_leave_rejects_insufficient_balance(isolated_stores, rag_module):
    # Default sick balance is 10 days; ask for a span with MORE than
    # 10 working days so it strictly exceeds the balance (3 full
    # Mon-Fri weeks = 15 working days).
    ok, message, details = rag_module.apply_leave(
        "kishore", "sick", date(2026, 8, 31), date(2026, 9, 18)
    )

    assert ok is False
    assert "Insufficient" in " ".join(details["errors"])
    assert rag_module.get_leave_history("kishore") == []


@pytest.mark.agent
def test_approve_leave_request_deducts_balance_and_notifies_requester(
    isolated_stores, mock_mail, rag_module
):
    rag_module.LEAVE_APPROVER_EMAIL = "manager@test.local"
    mock_mail.messages = []  # own the mailbox for this test

    ok, _msg, details = rag_module.apply_leave(
        "kishore", "sick", date(2026, 8, 31), date(2026, 8, 31)
    )
    assert ok is True
    request_id = details["record"]["id"]
    balance_before = rag_module.get_leave_balance("kishore", "sick")

    approved, approve_message = rag_module.approve_leave_request(
        request_id, approver_note="approved, feel better"
    )

    assert approved is True, approve_message

    # Status really flipped in the store.
    history = rag_module.get_leave_history("kishore")
    assert history[0]["status"] == "approved"
    assert history[0]["approver_note"] == "approved, feel better"

    # Balance really got deducted, exactly once, by the requested days.
    balance_after = rag_module.get_leave_balance("kishore", "sick")
    assert balance_after == balance_before - 1

    # The requester (SMTP_USER mailbox) really got a confirmation email -
    # this is the SECOND send_mail call (first was the approver request).
    assert len(mock_mail.sent) == 2
    assert "approved" in mock_mail.sent[-1]["subject"].lower()


@pytest.mark.agent
def test_reject_leave_request_does_not_touch_balance(
    isolated_stores, mock_mail, rag_module
):
    rag_module.LEAVE_APPROVER_EMAIL = "manager@test.local"

    ok, _msg, details = rag_module.apply_leave(
        "kishore", "sick", date(2026, 8, 31), date(2026, 8, 31)
    )
    request_id = details["record"]["id"]
    balance_before = rag_module.get_leave_balance("kishore", "sick")

    rejected, reject_message = rag_module.reject_leave_request(request_id)

    assert rejected is True, reject_message
    assert rag_module.get_leave_history("kishore")[0]["status"] == "rejected"
    # Balance was never deducted for a pending request, so rejecting
    # it must leave the balance exactly where it was.
    assert rag_module.get_leave_balance("kishore", "sick") == balance_before


@pytest.mark.agent
@pytest.mark.failure
def test_approve_nonexistent_leave_request_fails_cleanly(isolated_stores, rag_module):
    ok, message = rag_module.approve_leave_request("not-a-real-id")
    assert ok is False
    assert "no longer exists" in message


@pytest.mark.agent
@pytest.mark.failure
@pytest.mark.edge_case
def test_approving_an_already_approved_request_is_rejected(isolated_stores, rag_module):
    ok, _msg, details = rag_module.apply_leave(
        "kishore", "casual", date(2026, 8, 31), date(2026, 8, 31)
    )
    request_id = details["record"]["id"]
    rag_module.approve_leave_request(request_id)

    ok2, message2 = rag_module.approve_leave_request(request_id)
    assert ok2 is False
    assert "already approved" in message2


# =================================================================
# PO AGENT - does apply_po() really create the record and email the
# APPROVER (never the vendor), and does approving it really email
# the vendor only afterwards?
# =================================================================

def _basic_po_items():
    return [{"name": "laptop", "quantity": 1, "unit_price": 1000}]


@pytest.mark.agent
def test_apply_po_creates_pending_record_and_only_emails_approver(
    isolated_stores, mock_mail, rag_module
):
    rag_module.PO_APPROVER_EMAIL = "approver@test.local"
    rag_module.PO_AUTO_APPROVE_THRESHOLD = 0.0  # force manual approval path

    ok, message, details = rag_module.apply_po(
        "kishore", "Dell", "engineering", _basic_po_items(),
        vendor_email="vendor@test.local",
    )

    assert ok is True, message
    record = details["record"]
    assert record["status"] == "pending"
    assert record["total_amount"] == 1000
    assert record["vendor_notified"] is False  # vendor NOT emailed yet

    # It's really in the store.
    stored = rag_module.get_pending_po_requests()
    assert len(stored) == 1
    assert stored[0]["id"] == record["id"]

    # Only the approver was emailed - not the vendor.
    assert len(mock_mail.sent) == 1
    assert mock_mail.sent[0]["to"] == ["approver@test.local"]


@pytest.mark.agent
def test_approve_po_request_marks_approved_and_emails_vendor(
    isolated_stores, mock_mail, rag_module
):
    rag_module.PO_APPROVER_EMAIL = "approver@test.local"
    rag_module.PO_AUTO_APPROVE_THRESHOLD = 0.0

    ok, _msg, details = rag_module.apply_po(
        "kishore", "Dell", "engineering", _basic_po_items(),
        vendor_email="vendor@test.local",
    )
    request_id = details["record"]["id"]
    mock_mail.sent = []  # only care about what approval triggers

    approved, approve_message = rag_module.approve_po_request(request_id)

    assert approved is True, approve_message

    stored = rag_module.get_all_po_requests()
    approved_record = next(r for r in stored if r["id"] == request_id)
    assert approved_record["status"] == "approved"
    assert approved_record["vendor_notified"] is True
    assert approved_record["email_status"] == "sent"

    # The vendor - not the approver - got the actual PO email now.
    assert len(mock_mail.sent) == 1
    assert mock_mail.sent[0]["to"] == ["vendor@test.local"]


@pytest.mark.agent
def test_reject_po_request_never_emails_vendor(isolated_stores, mock_mail, rag_module):
    rag_module.PO_APPROVER_EMAIL = "approver@test.local"
    rag_module.PO_AUTO_APPROVE_THRESHOLD = 0.0

    ok, _msg, details = rag_module.apply_po(
        "kishore", "Dell", "engineering", _basic_po_items(),
        vendor_email="vendor@test.local",
    )
    request_id = details["record"]["id"]
    mock_mail.sent = []

    rejected, reject_message = rag_module.reject_po_request(request_id)

    assert rejected is True, reject_message
    stored = rag_module.get_all_po_requests()
    rejected_record = next(r for r in stored if r["id"] == request_id)
    assert rejected_record["status"] == "rejected"
    assert rejected_record["vendor_notified"] is False
    # No mail at all - the vendor must never hear about a rejected PO.
    assert len(mock_mail.sent) == 0


@pytest.mark.agent
@pytest.mark.regression
def test_apply_po_auto_approves_under_threshold_and_emails_vendor_immediately(
    isolated_stores, mock_mail, rag_module
):
    """
    Regression test for a fixed bug: apply_po() used to save the
    record with status="auto_approved" and THEN call
    approve_po_request(id) to send the vendor email - but
    approve_po_request() refuses to act on any request whose status
    isn't "pending", so it immediately rejected itself with "That
    request is already auto_approved." and the vendor was silently
    never notified. apply_po() now stores it as "pending" and lets
    approve_po_request() do the real status flip + vendor email, so
    the persisted record (not the possibly-stale dict returned by
    apply_po() itself) must end up "approved" with the vendor
    actually notified.
    """
    rag_module.PO_AUTO_APPROVE_THRESHOLD = 5000.0  # our 1000 PO qualifies

    ok, message, details = rag_module.apply_po(
        "kishore", "Dell", "engineering", _basic_po_items(),
        vendor_email="vendor@test.local",
    )

    assert ok is True, message
    record_id = details["record"]["id"]

    # Check the PERSISTED record, not the in-memory dict apply_po()
    # returned - approve_po_request() re-loads the store from disk
    # to make its update, so that's the only authoritative source.
    stored = rag_module.get_all_po_requests()
    final_record = next(r for r in stored if r["id"] == record_id)
    assert final_record["status"] == "approved"
    assert final_record["vendor_notified"] is True
    assert len(mock_mail.sent) == 1
    assert mock_mail.sent[0]["to"] == ["vendor@test.local"]


@pytest.mark.agent
@pytest.mark.failure
@pytest.mark.edge_case
def test_apply_po_over_budget_limit_is_blocked_and_not_stored(
    isolated_stores, mock_mail, rag_module
):
    rag_module.PO_BUDGET_LIMITS = {"engineering": 500}  # our PO totals 1000

    ok, message, details = rag_module.apply_po(
        "kishore", "Dell", "engineering", _basic_po_items(),
        vendor_email="vendor@test.local",
    )

    assert ok is False
    assert "budget" in message.lower()
    assert rag_module.get_all_po_requests() == []
    assert len(mock_mail.sent) == 0


@pytest.mark.agent
@pytest.mark.failure
def test_approve_already_rejected_po_request_fails(isolated_stores, rag_module):
    rag_module.PO_AUTO_APPROVE_THRESHOLD = 0.0
    ok, _msg, details = rag_module.apply_po(
        "kishore", "Dell", "engineering", _basic_po_items()
    )
    request_id = details["record"]["id"]
    rag_module.reject_po_request(request_id)

    ok2, message2 = rag_module.approve_po_request(request_id)
    assert ok2 is False
    assert "already rejected" in message2


# =================================================================
# MAIL AGENT - does send_mail() really reach the fake SMTP server,
# and does search_mail() really return a matching message (not just
# "some" message)?
# =================================================================

@pytest.mark.agent
def test_send_mail_actually_invokes_fake_smtp_sendmail(mock_mail, rag_module):
    ok, message = rag_module.send_mail(
        "someone@test.local", "Quarterly report", "See attached.", cc="cc@test.local"
    )

    assert ok is True, message
    assert len(mock_mail.sent) == 1
    sent = mock_mail.sent[0]
    assert sent["subject"] == "Quarterly report"
    assert "someone@test.local" in sent["to"]
    assert "cc@test.local" in sent["to"]  # cc goes into the envelope recipients too


@pytest.mark.agent
@pytest.mark.failure
def test_send_mail_with_no_recipient_fails_without_hitting_smtp(mock_mail, rag_module):
    ok, message = rag_module.send_mail("", "Subject", "Body")
    assert ok is False
    assert "recipient" in message.lower()
    assert len(mock_mail.sent) == 0


@pytest.mark.agent
def test_search_mail_returns_only_the_matching_message(mock_mail, rag_module):
    mock_mail.set_inbox([
        {"id": 1, "subject": "Budget approval needed", "from": "sarah@test.local"},
        {"id": 2, "subject": "Lunch plans", "from": "bob@test.local"},
        {"id": 3, "subject": "Weekend hiking trip", "from": "amy@test.local"},
    ])

    context, sources = rag_module.search_mail("budget", require_keyword_match=True)

    assert "Budget approval needed" in context
    assert any("Budget approval needed" in s for s in sources)
    # It should not have pulled in the unrelated messages as evidence.
    assert "Lunch plans" not in context
    assert "Weekend hiking trip" not in context


@pytest.mark.agent
@pytest.mark.failure
def test_search_mail_with_no_keyword_match_returns_no_evidence(mock_mail, rag_module):
    mock_mail.set_inbox([
        {"id": 1, "subject": "Lunch plans", "from": "bob@test.local"},
    ])

    context, sources = rag_module.search_mail(
        "nonexistent-topic-xyz", require_keyword_match=True
    )

    assert context == ""
    assert sources == []


# =================================================================
# MEETINGS AGENT - does schedule_meeting() really write a new event
# to the .ics file, such that it's then found by a real lookup?
# =================================================================

@pytest.mark.agent
def test_schedule_meeting_writes_event_that_is_then_found(mock_calendar, rag_module):
    start = datetime(2026, 9, 3, 10, 0)
    end = datetime(2026, 9, 3, 10, 30)

    ok, message = rag_module.schedule_meeting(
        "Standup", start, end, location="Zoom"
    )

    assert ok is True, message

    # Prove it via an independent read path, not the write call's
    # own success flag - this is a different function reading the
    # same file back from disk.
    events = rag_module.get_events_in_range(date(2026, 9, 3), date(2026, 9, 3))
    assert any("Standup" in e for e in events)

    # The pre-existing fixture event must still be intact too - a
    # buggy write path could have clobbered the calendar instead of
    # appending to it.
    older_events = rag_module.get_events_in_range(date(2026, 8, 31), date(2026, 9, 2))
    assert any("Sprint Planning" in e for e in older_events)


@pytest.mark.agent
@pytest.mark.failure
def test_schedule_meeting_fails_cleanly_when_not_configured(rag_module, monkeypatch):
    monkeypatch.setattr(rag_module, "MEETINGS_ICS_PATH", "")

    ok, message = rag_module.schedule_meeting(
        "Standup", datetime(2026, 9, 3, 10, 0)
    )

    assert ok is False
    assert "not configured" in message.lower() or "isn't configured" in message.lower()