"""
Standalone test for the multi-user meeting-schedule-check feature -
exercises check_group_availability() directly against the sample
calendars, without going through the Streamlit UI.

Run this in the same venv/terminal you run `streamlit run app.py`
in (needs the same installed packages), from the same directory
as rag.py.

Usage:
    1. Put priya.ics / arjun.ics / me.ics somewhere on disk
       (or use the test_calendars/ folder as-is).
    2. Adjust the paths in the os.environ lines below if needed.
    3. python test_group_availability.py
"""

import os

os.environ["NOVA_MEETINGS_ICS_PATH"] = "test_calendars/me.ics"
os.environ["NOVA_MEETINGS_ICS_PATHS"] = (
    "priya:test_calendars/priya.ics,arjun:test_calendars/arjun.ics"
)

import rag
from datetime import datetime

print("=" * 60)
print("Configured calendar users")
print("=" * 60)
print(rag.get_configured_calendar_users())
print()

print("=" * 60)
print("Name resolution")
print("=" * 60)
for name in ("Priya", "priya sharma", "arjun", "nobody"):
    print(f"resolve_calendar_user({name!r}) -> {rag.resolve_calendar_user(name)!r}")
print()

print("=" * 60)
print("check_group_availability - Priya has a conflict, Arjun doesn't,")
print("'Random Person' isn't a configured calendar at all")
print("=" * 60)

# Matches the Design Review event seeded in priya.ics (14:00-15:00)
start = datetime(2026, 8, 21, 14, 0)
end = datetime(2026, 8, 21, 15, 0)

conflicts, unchecked = rag.check_group_availability(
    "Priya, Arjun, Random Person", start, end
)

print("conflicts:", conflicts)
print("  -> expect: {'priya': ['Design Review (2026-08-21 14:00-15:00)']}")
print("unchecked:", unchecked)
print("  -> expect: ['Random Person'] (no matching calendar configured)")
print()

print("=" * 60)
print("Same attendees, a time slot that DOESN'T overlap Priya's event")
print("=" * 60)

start2 = datetime(2026, 8, 21, 16, 0)
end2 = datetime(2026, 8, 21, 17, 0)

conflicts2, unchecked2 = rag.check_group_availability(
    "Priya, Arjun", start2, end2
)

print("conflicts:", conflicts2)
print("  -> expect: {} (no overlap this time)")
print("unchecked:", unchecked2)
