"""
Document Generation Agent
=========================

Generates real, downloadable .docx business documents assembled from
data that already lives inside NOVA (PO/leave/expense stores, the
calendar, indexed documents) - never from anything the LLM invents.

Design mirrors the rest of the app's "read agents":

    detect intent -> pull real records from rag.py -> build a
    grounded evidence block for the answer LLM -> separately, write
    an actual .docx file to disk so the UI can offer it as a
    download.

The evidence block (`context`) is fed to the SAME grounding-prompt
machinery `build_routed_prompt()` already uses for RECOMMEND/PLAN, so
the chat reply describing the document is held to the same
no-invented-facts bar. The .docx itself is built directly from the
raw records in Python (not by the LLM), so the file's actual content
can never drift from what's really in the stores.

Public entry point: generate_document_evidence(question)
    -> (context: str, sources: list[str], file_info: dict | None)

file_info, when a document was actually produced, looks like:
    {"path": "<abs path>.docx", "filename": "...", "doc_type": "..."}
"""

import os
import re
import uuid
from datetime import datetime, date, timedelta

from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

import rag

GENERATED_DOCS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "generated_documents"
)

# Single-tenant convention: every record in the PO/leave/expense
# stores is keyed against the literal string "me" (see rag.py) - that
# key is fine for lookups, but it must never be printed verbatim in a
# document a real person is going to read or sign. Set the actual
# name NOVA should print in generated letters here, or via the
# NOVA_USER_DISPLAY_NAME env var. Falls back to a clearly-a-placeholder
# label (rather than silently printing "me") if neither is set, so a
# missing config is obvious in the output instead of looking like a bug.
USER_DISPLAY_NAME = os.environ.get("NOVA_USER_DISPLAY_NAME", "").strip() or "[Employee Name Not Configured]"


def _is_system_generated_reason(reason):
    """
    True for internal/system-authored reason strings (e.g. the fixed
    string _build_autonomous_leave_plan in app.py attaches to leave
    requests it files on the user's behalf) that describe HOW the
    request was filed, not WHY the person is taking leave. These must
    never be printed as if a human wrote them in a formal letter.
    """
    if not reason:
        return False
    return reason.strip().lower().startswith("requested via nova")


def _ensure_output_dir():
    os.makedirs(GENERATED_DOCS_DIR, exist_ok=True)


# =========================================================
# INTENT DETECTION
# =========================================================

# Checked by app.py's route_query() BEFORE the read-only DOCUMENT
# (RAG-search) signals, and before SEND_MAIL's "draft an email"
# signals (those are matched earlier still, so no clash there).
# Deliberately requires a generation VERB ("generate"/"create"/
# "draft"/"prepare"/"make") together with a document-shaped noun, so
# an ordinary question like "what's my PO status" never lands here.
_GENERATION_VERBS = (
    "generate", "create", "draft", "prepare", "make", "produce",
    "write up", "put together",
)

DOCUMENT_TYPE_SIGNALS = {
    "purchase_order": (
        "po document", "po copy", "purchase order document",
        "purchase order copy", "purchase order form",
    ),
    "leave_letter": (
        "leave letter", "leave certificate", "leave application letter",
        "leave application", "leave approval letter",
    ),
    "expense_report": (
        "expense report", "expense statement", "reimbursement report",
        "reimbursement summary", "expense summary",
    ),
    "meeting_minutes": (
        "meeting minutes", "minutes of meeting", "mom document", " mom ",
    ),
}


def looks_like_document_generation_request(question):
    """
    Fast, keyword-based check used by route_query(). Returns True
    only when the message both (a) asks to GENERATE something and
    (b) names one of the known document shapes above.
    """

    q = " " + question.lower().strip() + " "

    has_verb = any(verb in q for verb in _GENERATION_VERBS)
    if not has_verb:
        return False

    for signals in DOCUMENT_TYPE_SIGNALS.values():
        if any(signal in q for signal in signals):
            return True

    # Generic "generate a/the business document" / "generate a
    # document" with no more specific noun still counts - handled
    # downstream by _detect_document_type() falling back to asking
    # what kind, rather than route_query() missing it entirely.
    if any(
        phrase in q
        for phrase in (" a document ", " a doc ", " a business document ")
    ):
        return True

    return False


def _detect_document_type(question):
    q = " " + question.lower().strip() + " "
    for doc_type, signals in DOCUMENT_TYPE_SIGNALS.items():
        if any(signal in q for signal in signals):
            return doc_type
    return None


# =========================================================
# HELPERS
# =========================================================

def _new_docx_path(doc_type):
    _ensure_output_dir()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{doc_type}_{stamp}_{uuid.uuid4().hex[:6]}.docx"
    return os.path.join(GENERATED_DOCS_DIR, filename), filename


def _add_title(doc, text):
    heading = doc.add_heading(text, level=1)
    heading.alignment = WD_ALIGN_PARAGRAPH.CENTER


def _add_kv(doc, label, value):
    p = doc.add_paragraph()
    run_label = p.add_run(f"{label}: ")
    run_label.bold = True
    p.add_run(str(value))


def _match_vendor(question, po_history):
    """Best-effort match of a vendor name the user typed against
    real vendors in this user's PO history."""
    q = question.lower()
    for record in po_history:
        vendor = (record.get("vendor") or "").strip()
        if vendor and vendor.lower() in q:
            return record
    return None


def _match_leave_type(question):
    q = question.lower()
    for leave_type in ("annual", "sick", "casual"):
        if leave_type in q:
            return leave_type
    return None


def _format_letter_date(value):
    """Render a stored 'YYYY-MM-DD' leave date as 'September 02,
    2026' for the printed letter. Falls back to the raw stored value
    unchanged if it isn't in that shape (never invents a date)."""
    if not value:
        return ""
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").strftime("%B %d, %Y")
    except ValueError:
        return str(value)


# =========================================================
# PURCHASE ORDER DOCUMENT
# =========================================================

def _build_purchase_order_docx(question):
    po_history = rag.get_po_history("me", include_cancelled=True)

    if not po_history:
        return None, "No purchase order requests were found for you to generate a document from.", [], None

    record = _match_vendor(question, po_history) or po_history[0]

    path, filename = _new_docx_path("purchase_order")
    doc = Document()
    _add_title(doc, "Purchase Order")

    _add_kv(doc, "PO ID", record.get("id", ""))
    _add_kv(doc, "Requester", record.get("requester", ""))
    _add_kv(doc, "Vendor", record.get("vendor", ""))
    _add_kv(doc, "Vendor Email", record.get("vendor_email", ""))
    _add_kv(doc, "Department", record.get("department", ""))
    _add_kv(doc, "Status", record.get("status", ""))
    _add_kv(doc, "Requested At", record.get("requested_at", ""))
    if record.get("due_date"):
        _add_kv(doc, "Due Date", record.get("due_date"))

    doc.add_paragraph("")
    doc.add_heading("Line Items", level=2)

    items = record.get("items") or []
    table = doc.add_table(rows=1, cols=4)
    table.style = "Light Grid Accent 1"
    hdr = table.rows[0].cells
    hdr[0].text, hdr[1].text, hdr[2].text, hdr[3].text = (
        "Item", "Quantity", "Unit Price (₹)", "Line Total (₹)"
    )
    for item in items:
        row = table.add_row().cells
        row[0].text = str(item.get("name", ""))
        row[1].text = rag.format_po_quantity(item.get("quantity", 0))
        row[2].text = f"{item.get('unit_price', 0):,.2f}"
        row[3].text = f"{item.get('line_total', 0):,.2f}"

    doc.add_paragraph("")
    total_p = doc.add_paragraph()
    total_run = total_p.add_run(f"Total Amount: ₹{record.get('total_amount', 0):,.2f}")
    total_run.bold = True
    total_run.font.size = Pt(12)

    if record.get("justification"):
        doc.add_paragraph("")
        doc.add_heading("Justification", level=2)
        doc.add_paragraph(record["justification"])

    doc.save(path)

    evidence_lines = [
        f"PO_ID: {record.get('id')}",
        f"VENDOR: {record.get('vendor')}",
        f"DEPARTMENT: {record.get('department')}",
        f"STATUS: {record.get('status')}",
        f"TOTAL_AMOUNT: ₹{record.get('total_amount', 0):,.2f}",
        f"REQUESTED_AT: {record.get('requested_at')}",
    ]
    sources = [f"PO request {record.get('id')} - {record.get('vendor')} (₹{record.get('total_amount', 0):,.2f})"]

    confirmation = (
        f"Your Purchase Order document is ready. It covers the PO for "
        f"**{record.get('vendor')}** (₹{record.get('total_amount', 0):,.2f}, "
        f"{record.get('department')} department), currently "
        f"**{record.get('status')}**."
    )

    return (path, filename), evidence_lines, sources, confirmation


# =========================================================
# LEAVE LETTER
# =========================================================

def _build_leave_letter_docx(question):
    leave_type = _match_leave_type(question)
    history = rag.get_leave_history("me", include_cancelled=True)

    if leave_type:
        history = [r for r in history if r.get("leave_type") == leave_type] or history

    if not history:
        return None, "No leave requests were found for you to generate a letter from.", [], None

    record = history[0]

    path, filename = _new_docx_path("leave_letter")
    doc = Document()
    _add_title(doc, "Leave Letter")

    today_str = datetime.now().strftime("%B %d, %Y")
    doc.add_paragraph(f"Date: {today_str}")
    doc.add_paragraph("")
    doc.add_paragraph("To Whom It May Concern,")
    doc.add_paragraph("")

    when_line = (
        f"from {_format_letter_date(record.get('start'))} to "
        f"{_format_letter_date(record.get('end'))}"
        if record.get("start") != record.get("end")
        else f"on {_format_letter_date(record.get('start'))}"
    )

    days = record.get("days", "?")
    day_word = "day" if days == 1 else "days"

    body = (
        f"This letter confirms that {USER_DISPLAY_NAME} has requested "
        f"{record.get('leave_type', '').title()} Leave {when_line}, "
        f"covering {days} working {day_word}. This request is currently "
        f"{record.get('status', 'pending')}."
    )
    doc.add_paragraph(body)

    # Only print a reason if the employee actually gave one - a
    # system-authored bookkeeping string (e.g. from an autonomous
    # task run) describes how the request was filed, not why the
    # person is taking leave, and has no place in a formal letter.
    reason = record.get("reason")
    if reason and not _is_system_generated_reason(reason):
        doc.add_paragraph(f"Reason stated: {reason}")

    doc.add_paragraph("")
    doc.add_paragraph("Please let us know if any further information or")
    doc.add_paragraph("documentation is required.")
    doc.add_paragraph("")
    doc.add_paragraph("Regards,")
    doc.add_paragraph(USER_DISPLAY_NAME)

    doc.save(path)

    evidence_lines = [
        f"LEAVE_ID: {record.get('id')}",
        f"USER: {record.get('user')}",
        f"LEAVE_TYPE: {record.get('leave_type')}",
        f"DATES: {record.get('start')} to {record.get('end')}",
        f"DAYS: {record.get('days')}",
        f"STATUS: {record.get('status')}",
    ]
    sources = [
        f"Leave request {record.get('id')} - {record.get('leave_type')} "
        f"({record.get('start')} to {record.get('end')})"
    ]

    confirmation = (
        f"Your Leave Letter is ready. It covers **{record.get('leave_type', '').title()}** "
        f"leave from **{_format_letter_date(record.get('start'))}** to "
        f"**{_format_letter_date(record.get('end'))}** "
        f"({record.get('days', '?')} day(s)), status **{record.get('status')}**."
    )

    return (path, filename), evidence_lines, sources, confirmation


# =========================================================
# EXPENSE REPORT (aggregate, not a single record)
# =========================================================

def _resolve_report_window(question):
    """Very small date-window resolver: 'this month' (default),
    'last month', or 'last N days'."""

    q = question.lower()
    today = date.today()

    match = re.search(r"last (\d+) days", q)
    if match:
        days = int(match.group(1))
        return today - timedelta(days=days), today, f"last {days} days"

    if "last month" in q:
        first_of_this_month = today.replace(day=1)
        last_month_end = first_of_this_month - timedelta(days=1)
        last_month_start = last_month_end.replace(day=1)
        return last_month_start, last_month_end, "last month"

    # Default: current month to date.
    return today.replace(day=1), today, "this month"


def _build_expense_report_docx(question):
    start_window, end_window, window_label = _resolve_report_window(question)
    history = rag.get_expense_history("me", include_cancelled=False)

    in_window = [
        record for record in history
        if record.get("date_incurred")
        and start_window.isoformat() <= record["date_incurred"] <= end_window.isoformat()
    ]

    path, filename = _new_docx_path("expense_report")
    doc = Document()
    _add_title(doc, "Expense Report")

    _add_kv(doc, "Employee", "me")
    _add_kv(doc, "Period", f"{start_window} to {end_window} ({window_label})")

    doc.add_paragraph("")

    if not in_window:
        doc.add_paragraph("No expense claims were recorded in this period.")
        doc.save(path)
        evidence_lines = [f"PERIOD: {window_label}", "EXPENSE_COUNT: 0"]
        sources = [f"Expense history for me, {window_label} - no records found"]
        confirmation = (
            f"Your Expense Report is ready for **{window_label}**, though no "
            f"expense claims were found in that period."
        )
        return (path, filename), evidence_lines, sources, confirmation

    table = doc.add_table(rows=1, cols=5)
    table.style = "Light Grid Accent 1"
    hdr = table.rows[0].cells
    hdr[0].text, hdr[1].text, hdr[2].text, hdr[3].text, hdr[4].text = (
        "Date", "Category", "Vendor", "Amount (₹)", "Status"
    )

    total = 0.0
    for record in in_window:
        row = table.add_row().cells
        row[0].text = str(record.get("date_incurred", ""))
        row[1].text = str(record.get("category", "")).replace("_", " ").title()
        row[2].text = str(record.get("vendor", "") or "-")
        row[3].text = f"{record.get('amount', 0):,.2f}"
        row[4].text = str(record.get("status", ""))
        total += record.get("amount", 0) or 0

    doc.add_paragraph("")
    total_p = doc.add_paragraph()
    total_run = total_p.add_run(f"Total Claimed: ₹{total:,.2f}")
    total_run.bold = True
    total_run.font.size = Pt(12)

    doc.save(path)

    evidence_lines = [f"PERIOD: {window_label}", f"EXPENSE_COUNT: {len(in_window)}", f"TOTAL_CLAIMED: ₹{total:,.2f}"]
    evidence_lines += [
        f"EXPENSE: {r.get('date_incurred')} | {r.get('category')} | "
        f"₹{r.get('amount', 0):,.2f} | {r.get('status')}"
        for r in in_window
    ]
    sources = [
        f"Expense claim {r.get('id')} - {r.get('category')} ₹{r.get('amount', 0):,.2f}"
        for r in in_window
    ]

    confirmation = (
        f"Your Expense Report is ready for **{window_label}**: "
        f"{len(in_window)} claim(s) totaling **₹{total:,.2f}**."
    )

    return (path, filename), evidence_lines, sources, confirmation


# =========================================================
# MEETING MINUTES
# =========================================================

def _build_meeting_minutes_docx(question):
    target_date = date.today()
    events = rag.get_events_on_date(target_date, user="me")

    path, filename = _new_docx_path("meeting_minutes")
    doc = Document()
    _add_title(doc, "Meeting Minutes")

    _add_kv(doc, "Date", target_date.strftime("%A, %B %d, %Y"))
    doc.add_paragraph("")

    if not events:
        doc.add_paragraph("No meetings were found on the calendar for this date.")
        doc.save(path)
        evidence_lines = [f"DATE: {target_date}", "MEETING_COUNT: 0"]
        sources = [f"Calendar for {target_date} - no events found"]
        confirmation = (
            f"Your Meeting Minutes document is ready for "
            f"**{target_date.strftime('%B %d, %Y')}**, though no meetings "
            f"were found on the calendar for that date."
        )
        return (path, filename), evidence_lines, sources, confirmation

    doc.add_heading("Agenda / Events", level=2)
    for label in events:
        doc.add_paragraph(label, style="List Bullet")

    doc.add_paragraph("")
    doc.add_heading("Notes", level=2)
    doc.add_paragraph(
        "(Add discussion notes and action items for each agenda item above.)"
    )

    doc.save(path)

    evidence_lines = [f"DATE: {target_date}", f"MEETING_COUNT: {len(events)}"]
    evidence_lines += [f"EVENT: {label}" for label in events]
    sources = [f"Calendar event - {label}" for label in events]

    confirmation = (
        f"Your Meeting Minutes document is ready for "
        f"**{target_date.strftime('%B %d, %Y')}**, covering "
        f"**{len(events)}** meeting(s) from your calendar."
    )

    return (path, filename), evidence_lines, sources, confirmation


_BUILDERS = {
    "purchase_order": _build_purchase_order_docx,
    "leave_letter": _build_leave_letter_docx,
    "expense_report": _build_expense_report_docx,
    "meeting_minutes": _build_meeting_minutes_docx,
}

_DOC_TYPE_LABEL = {
    "purchase_order": "Purchase Order",
    "leave_letter": "Leave Letter",
    "expense_report": "Expense Report",
    "meeting_minutes": "Meeting Minutes",
}


# =========================================================
# PUBLIC ENTRY POINT
# =========================================================

def generate_document_evidence(question):
    """
    Returns (context, sources, file_info, confirmation).

    context: grounded evidence text - kept for the "Sources"-style
        audit trail and for any caller that still wants to run it
        through an LLM, but NOT required to produce an answer.
    sources: list[str] for the UI's "Sources" expander.
    file_info: {"path", "filename", "doc_type"} if a .docx was
        actually written, else None.
    confirmation: a ready-to-display, fully deterministic sentence
        confirming what was generated - built directly from the real
        record in Python, never by an LLM. app.py uses this AS THE
        ANSWER for the GENERATE_DOCUMENT route instead of routing
        through Ollama/Groq/Gemini, so a local-model outage (e.g. the
        Ollama server erroring on /api/generate) can never strand an
        already-generated file behind a failed chat reply. None only
        when no document type could be identified at all.
    """

    doc_type = _detect_document_type(question)

    if doc_type is None:
        context = (
            "The user asked to generate a document but didn't specify "
            "which kind. NOVA can currently generate: a Purchase Order "
            "document, a Leave Letter, an Expense Report, or Meeting "
            "Minutes."
        )
        confirmation = (
            "I can generate a few kinds of documents for you: a "
            "**Purchase Order** document, a **Leave Letter**, an "
            "**Expense Report**, or **Meeting Minutes**. Which one would "
            "you like?"
        )
        return context, [], None, confirmation

    builder = _BUILDERS[doc_type]

    try:
        result, evidence_or_message, sources, confirmation = builder(question)
    except Exception as error:
        message = (
            f"Document generation failed while building the "
            f"{_DOC_TYPE_LABEL[doc_type]}: {error}"
        )
        return message, [], None, message

    if result is None:
        # evidence_or_message is a plain "nothing found" message here,
        # and confirmation is None - fall back to the same text so the
        # user still gets a clear, non-empty reply.
        return evidence_or_message, [], None, evidence_or_message

    (path, filename) = result
    evidence_lines = evidence_or_message

    context = (
        f"A {_DOC_TYPE_LABEL[doc_type]} document was generated and is "
        f"available for download as '{filename}'. It was built from the "
        f"following real, verified data:\n\n" + "\n".join(evidence_lines)
    )

    file_info = {"path": path, "filename": filename, "doc_type": doc_type}

    return context, sources, file_info, confirmation