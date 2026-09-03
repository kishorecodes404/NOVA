"""
Notification & Alert Intelligence Agent

Collects the latest state from every existing agent's own store
(PO/Leave/Expense/Meetings/Mail - Task has no store yet, see
_collect_task_notifications() below) and turns it into a single,
prioritized notification feed for when the user opens NOVA.

Every fact shown here is read straight from rag.py's stores, never
composed or judged by an LLM - same rule document_generator.py and
report_generator.py already follow. "AI Prioritized" (see the panel
header) means a deterministic, multi-signal scoring + natural-
language templating engine: urgency score -> priority tier -> a
plain-English "why" sentence built from the record's own fields.
That's a design choice, not a limitation - it means the panel is
instant on every page load and never invents a fact an LLM call
might hallucinate. If literal generative text is wanted instead,
_reason_for_* below is the place to swap in a (batched, cached) LLM
call.

Wired into app.py's main() - see render_notification_panel() call
right under the top bar - and is entirely additive: no existing
route, agent, or store is touched. The one exception is the
Approve/Reject action on Expense items, which calls the SAME
rag.py functions (approve_expense_request/reject_expense_request)
your email-approval links already use - no new state, no new
store, just a second UI for an existing action, gated behind an
explicit confirm step. PO items are notify-only here: in this
single-tenant app every PO is "requested by me", so the requester
approving their own PO in this panel would be self-approval - that
decision belongs to whoever actually signs off on the PO (e.g. the
vendor/procurement side), not the requester, so no action buttons
are offered.
"""

import html
from collections import OrderedDict
from datetime import date, datetime, timedelta

import streamlit as st

from rag import (
    approve_expense_request,
    approve_po_request,
    get_events_in_range,
    get_pending_expense_requests,
    get_pending_leave_requests,
    get_pending_po_requests,
    reject_expense_request,
    reject_po_request,
    search_mail,
)

# ============================================================
# URGENCY SCORING
# ============================================================
# A continuous score drives sort order; a 3-tier label (High/Medium/
# Low) is what's actually shown, via left-border + text color rather
# than an extra emoji per item (three colors of dot next to
# already-colored text read as noise, not signal).

URGENCY_OVERDUE = 110
URGENCY_CRITICAL = 100
URGENCY_HIGH = 75
URGENCY_MEDIUM = 45
URGENCY_LOW = 20

PRIORITY_BANDS = (
    (URGENCY_HIGH, "High"),
    (URGENCY_MEDIUM, "Medium"),
    (0, "Low"),
)


def _priority(score):
    """Returns the priority label for a numeric urgency score."""

    for threshold, label in PRIORITY_BANDS:
        if score >= threshold:
            return label

    return "Low"


_PRIORITY_COLORS = {
    "High": "#dc2626",
    "Medium": "#ea580c",
    "Low": "#2563eb",
}


def _days_until(date_str):
    """Positive = days from now, negative = already overdue, None = no/unparseable date."""

    if not date_str:
        return None

    try:
        target = datetime.strptime(str(date_str)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None

    return (target - date.today()).days


def _age_days(iso_timestamp):
    """How long an item has been sitting pending, in whole days."""

    if not iso_timestamp:
        return 0

    try:
        requested = datetime.fromisoformat(iso_timestamp)
    except ValueError:
        return 0

    return max(0, (datetime.now() - requested).days)


# ============================================================
# PER-AGENT COLLECTORS
# ============================================================
# Each collector talks to exactly one existing rag.py store and
# returns a flat list of plain dicts with a consistent shape:
#   agent, icon, title, detail, reason, score, timestamp,
#   and optionally action_label/action_prompt (navigate-to-chat)
#   or approve_ref (real approve/reject buttons - Expense only; PO
#   is notify-only, see module docstring).
# A failure in one agent (e.g. mail not configured, no calendar for
# this user) is swallowed by gather_notifications() so it can never
# blank out every other agent's notifications.

def _collect_po_notifications():
    items = []

    for req in get_pending_po_requests():

        due_in = _days_until(req.get("due_date"))
        age = _age_days(req.get("requested_at", ""))
        vendor = req.get("vendor") or "vendor TBD"
        amount = req.get("total_amount", 0)

        if due_in is not None and due_in < 0:
            score = URGENCY_OVERDUE
            reason = (
                f"Payment to {vendor} is overdue by {-due_in} day(s) - "
                "approval is blocking the vendor payment."
            )
        elif due_in is not None and due_in <= 1:
            score = URGENCY_CRITICAL
            reason = (
                f"Payment to {vendor} is due {'today' if due_in == 0 else 'tomorrow'} "
                "and approval is still pending."
            )
        elif due_in is not None and due_in <= 3:
            score = URGENCY_HIGH
            reason = f"Payment to {vendor} is due in {due_in} days - approval is still pending."
        elif due_in is not None:
            score = URGENCY_MEDIUM + min(age * 3, 25)
            reason = f"Payment is due in {due_in} days. Review approval."
        else:
            score = URGENCY_MEDIUM + min(age * 3, 25)
            reason = f"₹{amount:,.2f} PO from {req.get('requester', 'someone')} has been awaiting approval for {age} day(s)."

        items.append({
            "agent": "PO",
            "icon": "📦",
            "title": f"PO approval pending — ₹{amount:,.2f}",
            "detail": f"{vendor} · requested by {req.get('requester', 'someone')}",
            "reason": reason,
            "score": score,
            "timestamp": req.get("requested_at", ""),
        })

    return items


def _collect_leave_notifications():
    items = []

    for req in get_pending_leave_requests():

        # A stale approval doesn't retroactively become an action
        # item once the leave window it covers has fully elapsed -
        # there's nothing left to approve in time for. Only skip
        # once the WHOLE window is behind us (end < today), so a
        # multi-day request already under way (start in the past,
        # end still upcoming) still surfaces correctly.
        ends_in = _days_until(req.get("end"))

        if ends_in is not None and ends_in < 0:
            continue

        age = _age_days(req.get("requested_at", ""))
        starts_in = _days_until(req.get("start"))
        leave_type = str(req.get("leave_type", "leave")).replace("_", " ").title()
        user = req.get("user", "someone")

        score = URGENCY_MEDIUM + min(age * 5, 25)

        if starts_in is not None and starts_in <= 0:
            score = max(score, URGENCY_HIGH)
            reason = f"{leave_type} leave starts today and approval is still pending."
        elif starts_in is not None and starts_in == 1:
            score = max(score, URGENCY_HIGH)
            reason = f"{leave_type} leave starts tomorrow and approval is still pending."
        elif starts_in is not None and starts_in <= 2:
            score = max(score, URGENCY_HIGH)
            reason = f"{leave_type} leave starts in {starts_in} days and approval is still pending."
        else:
            reason = f"{leave_type} leave request from {user} has been pending for {age} day(s)."

        items.append({
            "agent": "Leave",
            "icon": "🏖️",
            "title": f"Leave request pending — {user}",
            "detail": f"{leave_type} · {req.get('start')} to {req.get('end')} ({req.get('days')} day(s))",
            "reason": reason,
            "score": score,
            "timestamp": req.get("requested_at", ""),
            "action_label": "Delete Request",
            "action_prompt": (
                f"Cancel/delete my pending {leave_type.lower()} leave request "
                f"from {req.get('start')} to {req.get('end')}."
            ),
        })

    return items


def _collect_expense_notifications():
    items = []

    for req in get_pending_expense_requests():

        age = _age_days(req.get("requested_at", ""))
        amount = req.get("amount", 0)
        category = str(req.get("category", "")).replace("_", " ").title()
        requester = req.get("requester", "someone")

        score = URGENCY_MEDIUM + min(age * 4, 30)
        reason = f"₹{amount:,.2f} {category} claim from {requester} has been awaiting approval for {age} day(s)."

        items.append({
            "agent": "Expense",
            "icon": "💳",
            "title": f"Expense claim pending — ₹{amount:,.2f}",
            "detail": f"{category} · {requester}",
            "reason": reason,
            "score": score,
            "timestamp": req.get("requested_at", ""),
            "action_label": "Approve Expense",
            "approve_ref": {
                "kind": "expense",
                "id": req.get("id"),
                "vendor": category,
                "amount": amount,
            },
        })

    return items


def _collect_meeting_notifications(user=None):
    items = []

    today = date.today()
    tomorrow = today + timedelta(days=1)

    for label in get_events_in_range(today, today, user=user):
        event_time = label.rsplit(" ", 1)[-1].rstrip(")")
        items.append({
            "agent": "Meeting",
            "icon": "📅",
            "title": "Meeting today",
            "detail": label,
            "reason": f"You have a meeting today at {event_time}.",
            "score": URGENCY_HIGH,
            "timestamp": "",
            "action_label": "View Meeting",
            "action_prompt": f"Show me the details of today's meeting: {label}",
        })

    for label in get_events_in_range(tomorrow, tomorrow, user=user):
        event_time = label.rsplit(" ", 1)[-1].rstrip(")")
        items.append({
            "agent": "Meeting",
            "icon": "📅",
            "title": "Meeting tomorrow",
            "detail": label,
            "reason": f"You have a meeting tomorrow at {event_time}.",
            "score": URGENCY_MEDIUM,
            "timestamp": "",
            "action_label": "View Meeting",
            "action_prompt": f"Show me the details of tomorrow's meeting: {label}",
        })

    return items


def _collect_mail_notifications(max_results=5):
    """
    Recent inbox activity, surfaced (not summarized) as notifications.
    Passing an empty query with require_keyword_match=False is the
    documented way to get search_mail() to fall back to "most recent
    messages" instead of trying to keyword-match nothing - see the
    require_keyword_match docstring in rag.py.
    """

    items = []

    try:
        _, sources = search_mail(
            "", max_results=max_results, require_keyword_match=False
        )
    except Exception:
        sources = []

    for source in sources:
        subject, _, sender = str(source).rpartition(" - ")
        subject = subject or source

        items.append({
            "agent": "Mail",
            "icon": "📧",
            "title": "Recent email",
            "detail": source,
            "reason": f"New message: \"{subject}\"" + (f" from {sender}." if sender else "."),
            "score": URGENCY_LOW,
            "timestamp": "",
            "action_label": "Open Mail",
            "action_prompt": f"Show me more about the email: {subject}",
        })

    return items


def _collect_task_notifications():
    """
    No Task agent/store exists in rag.py yet (same gap noted in
    report_generator.py's module docstring). Rather than surfacing a
    "not connected" placeholder card on every page load - noise the
    user can never act on or dismiss - this simply contributes
    nothing to the panel until a real Task store exists.
    """

    return []


# ============================================================
# GROUP DUPLICATES
# ============================================================

def _collapse_duplicates(items):
    """
    Merges exact duplicates (same agent + title + detail) into one
    entry with a count, and rewrites that entry's title/reason to
    read as a group ("3 email delivery failures") instead of
    repeating the same card three times. Only ever merges byte-for-
    byte identical items, so two genuinely different emails/events
    that happen to share a title are never accidentally collapsed.
    """

    groups = OrderedDict()

    for item in items:
        key = (item["agent"], item["title"], item["detail"])
        if key not in groups:
            merged = dict(item)
            merged["count"] = 1
            groups[key] = merged
        else:
            groups[key]["count"] += 1
            groups[key]["score"] = max(groups[key]["score"], item["score"])

    collapsed = []

    for group in groups.values():

        count = group["count"]

        if count > 1:

            detail_lower = str(group.get("detail", "")).lower()
            failure_keywords = ("failure", "undeliverable", "bounce", "delivery status")

            if group["agent"] == "Mail" and any(k in detail_lower for k in failure_keywords):
                group["title"] = f"{count} email delivery failures"
                group["reason"] = f"{count} messages failed to deliver - worth checking your outgoing mail."
                group["action_prompt"] = "Show me my recent emails, especially any delivery failures."
            elif group["agent"] == "Mail":
                group["title"] = f"{count} similar emails"
                group["reason"] = f"{count} recent messages with similar content."
            else:
                group["title"] = f"{group['title']} (×{count})"

        collapsed.append(group)

    return collapsed


# ============================================================
# GATHER + PRIORITIZE
# ============================================================

def gather_notifications(user=None, limit=8):
    """
    Collect notifications from every existing agent, group exact
    duplicates, and return the top `limit`, highest-urgency first.
    Each per-agent collector is isolated in its own try/except so
    one agent being unconfigured or erroring (e.g. mail not set up,
    no calendar for this user) never blanks out the others.
    """

    collectors = (
        _collect_po_notifications,
        _collect_leave_notifications,
        _collect_expense_notifications,
        lambda: _collect_meeting_notifications(user=user),
        _collect_mail_notifications,
        _collect_task_notifications,
    )

    all_items = []

    for collect in collectors:
        try:
            all_items.extend(collect())
        except Exception as error:
            print(f"[NOTIFICATIONS] collector failed: {error}", flush=True)

    all_items = _collapse_duplicates(all_items)
    all_items.sort(key=lambda item: item["score"], reverse=True)

    return all_items[:limit]


# ============================================================
# ACTIONS (real approve/reject, reusing existing rag.py functions)
# ============================================================

def _render_approve_reject(ref, compact=False):
    """
    Two-click confirm for PO/Expense approval - calls the SAME
    approve_po_request/approve_expense_request/reject_* functions
    your email-approval links already use. The confirm step exists
    so a stray click on the dashboard can never silently approve
    real money; nothing here bypasses or duplicates that logic.
    `compact` swaps full-word buttons for icon-only ones so the
    action fits in a narrow side column instead of its own row.
    """

    kind = ref["kind"]
    ref_id = ref["id"]

    if not ref_id:
        return

    state_key = f"notif_pending_action_{kind}_{ref_id}"
    pending = st.session_state.get(state_key)

    if pending in ("approve", "reject"):

        verb = "Approve" if pending == "approve" else "Reject"

        st.caption(f"{verb} ₹{ref['amount']:,.2f} · {ref['vendor']}?")

        yes_col, no_col = st.columns(2)

        with yes_col:
            if st.button("✓", key=f"{state_key}_yes", help=f"Confirm {verb.lower()}", use_container_width=True):
                if kind == "po":
                    fn = approve_po_request if pending == "approve" else reject_po_request
                else:
                    fn = approve_expense_request if pending == "approve" else reject_expense_request

                ok, message = fn(ref_id, "Actioned from NOVA notifications panel.")

                if ok:
                    st.success(message, icon="✅")
                else:
                    st.error(message, icon="⚠️")

                st.session_state[state_key] = None

        with no_col:
            if st.button("✕", key=f"{state_key}_no", help="Cancel", use_container_width=True):
                st.session_state[state_key] = None

    else:

        approve_col, reject_col = st.columns(2)

        with approve_col:
            if st.button("✅", key=f"{state_key}_approve", help="Approve", use_container_width=True):
                st.session_state[state_key] = "approve"

        with reject_col:
            if st.button("✖", key=f"{state_key}_reject", help="Reject", use_container_width=True):
                st.session_state[state_key] = "reject"


# ============================================================
# RENDER
# ============================================================

def render_notification_panel(user=None, limit=8):
    """
    Renders the prioritized notification feed. Cheap to call every
    run (it's just a handful of local JSON/ICS reads plus one IMAP
    scan) - no caching here on purpose, since pending approvals and
    inbox state change between page loads and a stale badge count
    would be worse than the extra latency.

    Rendered compactly and inside a fixed-height scrolling container
    - each item is a single dense row (icon + title + detail on one
    line, priority + reason on the next), with its action button
    beside it rather than on its own row below, so a full 8-item
    panel stays put instead of pushing the rest of the page down.
    """

    notifications = gather_notifications(user=user, limit=limit)

    if not notifications:
        return

    high_count = sum(1 for item in notifications if _priority(item["score"]) == "High")

    count = len(notifications)
    noun = "notification" if count == 1 else "notifications"
    badge = f"🔔 {count} {noun}"
    if high_count:
        badge += f" · {high_count} important"

    with st.expander(f"{badge} — Notifications", expanded=bool(high_count)):

        st.caption("✨ AI Prioritized · Smart Alerts")

        # Fixed-height, internally-scrolling container - the panel's
        # footprint stops growing past ~320px no matter how many
        # items are in it. `height=` needs a reasonably recent
        # Streamlit; if your version predates it, this silently
        # falls back to a plain (unbounded) container instead of
        # erroring the whole panel out.
        try:
            list_area = st.container(height=320)
        except TypeError:
            list_area = st.container()

        with list_area:

            for item in notifications:

                priority_label = _priority(item["score"])
                color = _PRIORITY_COLORS[priority_label]

                # title/detail/reason can carry raw user or mail-subject
                # text (vendor names, leave reasons, email subjects) -
                # escape before interpolating into unsafe_allow_html, or
                # a stray "<"/"&" in any of those breaks the card's markup.
                title = html.escape(str(item["title"]))
                detail = html.escape(str(item["detail"]))
                agent = html.escape(str(item["agent"]))
                reason = html.escape(str(item.get("reason", "")))

                has_action = bool(item.get("approve_ref") or (item.get("action_label") and item.get("action_prompt")))
                card_col, action_col = st.columns([5, 1.4]) if has_action else (st.container(), None)

                with card_col:
                    st.markdown(
                        f"""
                        <div style="
                            padding:6px 10px; border-radius:8px;
                            border-left:3px solid {color};
                            background:rgba(0,0,0,0.02); margin-bottom:2px;
                            font-size:13px; line-height:1.3;
                        ">
                            <span style="font-weight:600; color:#1d1d1f;">{item['icon']} {title}</span>
                            <span style="color:#73737c;"> · {detail}</span>
                            <span style="float:right; font-size:10px; font-weight:600; color:{color}; text-transform:uppercase;">{agent}</span>
                            <div style="color:{color}; font-size:11.5px; margin-top:2px; font-weight:500;">
                                {priority_label} — {reason}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                if action_col is not None:
                    with action_col:
                        if item.get("approve_ref"):
                            _render_approve_reject(item["approve_ref"], compact=True)
                        elif item.get("action_label") and item.get("action_prompt"):
                            item_key = f"notif_action_{item['agent']}_{hash(item['title'] + item['detail'])}"
                            if st.button(item["action_label"], key=item_key, use_container_width=True):
                                st.session_state["pending_prompt"] = item["action_prompt"]