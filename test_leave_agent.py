"""
Standalone test for the Leave Agent - exercises validate_leave_request()
and apply_leave() directly against a scratch JSON store, without going
through the Streamlit UI.

Run this in the same venv/terminal you run `streamlit run app.py` in,
from the same directory as rag.py.

Usage:
    python test_leave_agent.py
"""

import os

# Use a throwaway store file so this test never touches real data,
# and is repeatable (deleted + recreated every run).
os.environ["NOVA_LEAVE_STORE_PATH"] = "test_leave_store.json"
os.environ["NOVA_LEAVE_HOLIDAYS"] = "2026-08-24"  # pretend Monday holiday
os.environ.pop("NOVA_LEAVE_ELIGIBLE_USERS", None)

if os.path.exists("test_leave_store.json"):
    os.remove("test_leave_store.json")

import rag
from datetime import date, datetime

def section(title):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


section("Leave balance before any requests (defaults)")
print(rag.get_leave_balances("me"))
print("  -> expect: {'annual': 18, 'sick': 10, 'casual': 6}")


section("Past date -> should be BLOCKED")
ok, errors, warnings, info = rag.validate_leave_request(
    "me", "annual", date(2020, 1, 1), date(2020, 1, 2)
)
print("ok:", ok)
print("errors:", errors)
print("  -> expect ok=False, an 'in the past' error")


section("End before start -> should be BLOCKED")
ok, errors, warnings, info = rag.validate_leave_request(
    "me", "annual", date(2026, 9, 5), date(2026, 9, 1)
)
print("ok:", ok)
print("errors:", errors)
print("  -> expect ok=False, an 'end date is before the start date' error")


section("Entirely weekend/holiday range -> should be BLOCKED")
# 2026-08-22 is a Saturday, 2026-08-23 a Sunday, 2026-08-24 a
# holiday (configured above).
ok, errors, warnings, info = rag.validate_leave_request(
    "me", "annual", date(2026, 8, 22), date(2026, 8, 24)
)
print("ok:", ok)
print("errors:", errors)
print("  -> expect ok=False, 'falls entirely on weekends/holidays'")


section("Normal valid request (includes a weekend -> warning only)")
# 2026-08-28 Fri, 08-29 Sat, 08-30 Sun, 08-31 Mon -> 2 working days
# (Fri + Mon), 2 weekend days.
ok, errors, warnings, info = rag.validate_leave_request(
    "me", "annual", date(2026, 8, 28), date(2026, 8, 31)
)
print("ok:", ok)
print("errors:", errors)
print("warnings:", warnings)
print("info:", info)
print("  -> expect ok=True, no errors, a weekend warning, working_days=2")


section("Apply that leave for real (no approver configured)")
success, message, details = rag.apply_leave(
    "me", "annual", date(2026, 8, 28), date(2026, 8, 31), reason="Trip"
)
print("success:", success)
print("message:", message)
print("record status:", details["record"]["status"])
print("approver_notified:", details["approver_notified"])
print("  -> expect success=True, status='pending', approver_notified=False")
print("     (no NOVA_LEAVE_APPROVER_EMAIL set in this test)")


section("Balance is NOT deducted yet (still pending approval)")
print(rag.get_leave_balances("me"))
print("  -> expect annual: 18 (unchanged - only deducted once approved,")
print("     which isn't built yet)")


section("Duplicate/overlapping request -> should be BLOCKED")
ok, errors, warnings, info = rag.validate_leave_request(
    "me", "annual", date(2026, 8, 29), date(2026, 9, 1)
)
print("ok:", ok)
print("errors:", errors)
print("  -> expect ok=False, an 'Overlapping leave request' error")


section("Insufficient balance -> should be BLOCKED")
ok, errors, warnings, info = rag.validate_leave_request(
    "me", "sick", date(2026, 9, 7), date(2026, 9, 21)
)
print("ok:", ok)
print("errors:", errors)
print("info:", info)
print("  -> expect ok=False, an 'Insufficient sick leave balance' error"
      " (10 available, more requested)")


section("Eligibility check")
os.environ["NOVA_LEAVE_ELIGIBLE_USERS"] = "priya, arjun"
ok, errors, warnings, info = rag.validate_leave_request(
    "me", "annual", date(2026, 9, 7), date(2026, 9, 8)
)
print("ok:", ok)
print("errors:", errors)
print("  -> expect ok=False, an 'isn't on the configured list' error"
      " ('me' isn't in priya/arjun)")

ok2, errors2, warnings2, info2 = rag.validate_leave_request(
    "priya", "annual", date(2026, 9, 7), date(2026, 9, 8)
)
print("priya ok:", ok2, "errors:", errors2)
print("  -> expect priya ok=True (eligible)")

os.environ.pop("NOVA_LEAVE_ELIGIBLE_USERS", None)


section("Leave history for 'me'")
print(rag.get_leave_history("me"))


section("Pending leave requests (approver's queue)")
pending = rag.get_pending_leave_requests()
print(pending)
print("  -> expect 1 pending request (the 08-28 to 08-31 annual one)")


section("Approve it")
request_id = pending[0]["id"]
success, message = rag.approve_leave_request(request_id)
print("success:", success)
print("message:", message)
print("balances after approval:", rag.get_leave_balances("me"))
print("  -> expect annual: 16 (18 - 2 working days, deducted on approval)")


section("Approving the same request again -> should be BLOCKED")
success2, message2 = rag.approve_leave_request(request_id)
print("success:", success2)
print("message:", message2)
print("  -> expect success=False, 'already approved'")


section("Reject a different (still-pending) request")
# The sick-leave over-balance one from earlier never got submitted
# (it was blocked by validate_leave_request), so submit a small
# valid one first, then reject it.
success3, message3, details3 = rag.apply_leave(
    "me", "casual", date(2026, 9, 10), date(2026, 9, 10), reason="Errand"
)
print("submitted:", success3, message3)

new_id = details3["record"]["id"]
success4, message4 = rag.reject_leave_request(new_id, approver_note="Team is short-staffed that day.")
print("reject success:", success4)
print("message:", message4)
print("balance unaffected:", rag.get_leave_balances("me"))
print("  -> expect casual balance unchanged (6) - rejection never deducts")

print()
print("pending requests now:", rag.get_pending_leave_requests())
print("  -> expect empty list (both requests resolved)")