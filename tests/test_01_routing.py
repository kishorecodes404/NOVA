"""
Step 3 - Routing Test Suite for route_query().

Covers every branch of app.route_query() in the order it actually
checks them:
    DOCUMENT (fast) -> SELF_INFO (fast) -> CHAT (greeting/self-intro)
    -> AUTO_EXECUTE / PLAN (compound) -> SEND_MAIL -> SCHEDULE_MEETING
    -> LEAVE_REQUEST -> PO_REQUEST -> EXPENSE_REQUEST -> MAIL
    -> MEETINGS -> WEB -> follow-up inheritance -> LLM fallback -> CHAT

Each fast-path test asserts routing WITHOUT any LLM mock installed
(mock_ollama is intentionally absent) - if a fast path regresses and
the query falls through to the LLM router, these tests fail loudly
with a real error instead of silently getting the right answer from
a scripted mock.

Precedence tests exist because route_query() is a big if/elif chain
- many phrases contain multiple agents' keywords (e.g. "send an
email" contains "email"), and only the ORDER of the checks makes
that resolve correctly. A refactor that reorders branches should
break precedence tests even if every individual signal list is
untouched.
"""

import pytest


# =====================================================================
# 1. FAST-PATH: DOCUMENT
# =====================================================================

@pytest.mark.routing
@pytest.mark.parametrize("query", [
    "what does my uploaded document say about the policy",
    "summarize the uploaded file",
    "check my uploaded pdf for the clause",
    "according to the document, what is the notice period",
    "according to the file, when does this expire",
    "in the document, is there a mention of overtime",
    "from the file, what's the budget",
    "what does my document say",
    "check my file for the address",
    "search my pdf for the total",
    "what's in the knowledge base about onboarding",
    "check the stored document for that",
])
def test_document_fast_routing(app_module, query):
    assert app_module.route_query(query) == "DOCUMENT"


# =====================================================================
# 2. FAST-PATH: SELF_INFO
# =====================================================================

@pytest.mark.routing
@pytest.mark.parametrize("query", [
    "what embedding model do you use",
    "what model does nova use",
    "what model do you use",
    "what llm are you",
    "which llm powers this",
    "what model are you running",
    "which model are you running",
    "what model powers this app",
    "what is your router model",
    "what is the answer model",
    "what ai model is this",
    "which ai model do you run on",
])
def test_self_info_fast_routing(app_module, query):
    assert app_module.route_query(query) == "SELF_INFO"


# =====================================================================
# 3. FAST-PATH: CHAT (greetings / self-introductions)
# =====================================================================

@pytest.mark.routing
@pytest.mark.parametrize("query", [
    "hi", "hii", "hello", "hey", "yo", "sup",
    "good morning", "good evening",
    "how are you", "how's it going",
    "thanks", "thank you so much",
    "ok", "cool", "lol",
    "bye", "see ya",
    "Hi!", "  hello  ", "HELLO.",
])
def test_chat_greeting_fast_routing(app_module, query):
    assert app_module.route_query(query) == "CHAT"


@pytest.mark.routing
@pytest.mark.parametrize("query", [
    "i'm kishore",
    "I am Kishore",
    "my name is Kishore",
    "this is Kishore",
    "call me Kishore",
    "im priya!",
])
def test_chat_self_introduction_fast_routing(app_module, query):
    assert app_module.route_query(query) == "CHAT"


@pytest.mark.routing
@pytest.mark.edge_case
def test_self_introduction_does_not_swallow_trailing_request(app_module):
    # A longer message that merely STARTS like a self-intro must
    # still route on its real content (MAIL here), not short-circuit
    # to CHAT.
    result = app_module.route_query(
        "I'm Kishore, can you check my email?"
    )
    assert result != "CHAT"


@pytest.mark.routing
@pytest.mark.edge_case
def test_greeting_with_real_question_falls_through(app_module):
    # "hi, what's Apple's stock price today?" - greeting is a
    # substring but the message as a WHOLE isn't in GREETING_SIGNALS,
    # so this must NOT be caught by the exact-match greeting check.
    result = app_module.route_query(
        "hi, what's Apple's stock price today?"
    )
    assert result == "WEB"


# =====================================================================
# 4. FAST-PATH: PLAN / AUTO_EXECUTE (compound requests)
# =====================================================================

@pytest.mark.routing
@pytest.mark.a2a
@pytest.mark.parametrize("query", [
    "check my meetings, pending POs, and emails before I go on leave tomorrow",
    "make sure nothing is impacted - check my calendar and my inbox",
    "check my email and my meetings before I take leave",
    "check on my expenses as well as my POs",
])
def test_plan_fast_routing_for_compound_requests(app_module, query):
    assert app_module.route_query(query) == "PLAN"


@pytest.mark.routing
@pytest.mark.a2a
def test_auto_execute_fast_routing_for_actionable_leave_conflict(app_module):
    # Needs ALL THREE: mentions leave/vacation/pto, mentions
    # apply/submit, AND mentions "reassign" - plus the compound plan
    # shape (>=2 agent categories + a compound cue).
    query = (
        "apply for leave tomorrow, check my meetings and pending POs, "
        "and reassign anything that conflicts"
    )
    assert app_module.route_query(query) == "AUTO_EXECUTE"


@pytest.mark.routing
@pytest.mark.edge_case
def test_compound_shaped_request_without_reassign_stays_plan_not_auto_execute(app_module):
    # Same compound shape and mentions leave+apply, but no
    # "reassign" - _looks_like_autonomous_action_request() should
    # be False, so this stays a read-only PLAN, not AUTO_EXECUTE.
    query = "apply for leave tomorrow, and check my meetings and emails"
    assert app_module.route_query(query) == "PLAN"


@pytest.mark.routing
@pytest.mark.edge_case
def test_single_topic_mentioning_two_nouns_is_not_a_plan(app_module):
    # Mentions MAIL ("emailed") and MEETINGS ("meeting") keywords,
    # but it's one single-topic question with no compound/checklist
    # shape (no comma, no "and", no "make sure") - must NOT trigger
    # the planner.
    result = app_module.route_query(
        "did Priya move our meeting after I emailed her?"
    )
    assert result != "PLAN"
    assert result != "AUTO_EXECUTE"


# =====================================================================
# 5. FAST-PATH: SEND_MAIL (and precedence over bare MAIL)
# =====================================================================

@pytest.mark.routing
@pytest.mark.parametrize("query", [
    "send an email to Sarah about the deadline",
    "send email to the team",
    "send a mail to Priya",
    "compose an email to HR",
    "draft an email to the vendor",
    "draft me an email to legal",
    "reply to Sarah's message",
    "forward this to my manager",
    "forward that to accounts",
])
def test_send_mail_fast_routing_literal_signals(app_module, query):
    assert app_module.route_query(query) == "SEND_MAIL"


@pytest.mark.routing
@pytest.mark.parametrize("query", [
    "reply thanks to Kishore",
    "reply thanks in mail to Kishore",
    "forward the attached file to accounts",
])
def test_send_mail_fast_routing_verb_gap_to_regex(app_module, query):
    assert app_module.route_query(query) == "SEND_MAIL"


@pytest.mark.routing
def test_send_mail_fast_routing_send_to_mail_regex(app_module):
    assert app_module.route_query("send hi to Kishore M S in mail") == "SEND_MAIL"


@pytest.mark.routing
@pytest.mark.edge_case
def test_send_mail_takes_precedence_over_readonly_mail(app_module):
    # "send an email" contains "email", which is also a bare-MAIL
    # signal. SEND_MAIL must win because it's checked first.
    assert app_module.route_query("send an email to Sarah") == "SEND_MAIL"


@pytest.mark.routing
@pytest.mark.edge_case
def test_readonly_reply_question_is_not_send_mail(app_module):
    # "did I get a reply from..." doesn't open with the bare verb,
    # so the anchored verb...to regex must NOT match it.
    result = app_module.route_query("did I get a reply from Sarah yet?")
    assert result != "SEND_MAIL"


# =====================================================================
# 6. FAST-PATH: SCHEDULE_MEETING (and precedence over read-only MEETINGS)
# =====================================================================

@pytest.mark.routing
@pytest.mark.parametrize("query", [
    "schedule a meeting with the design team",
    "schedule a call for Friday",
    "set up a meeting with Priya",
    "set up a call with the vendor",
    "book a meeting for 3pm",
    "book a call with legal",
    "arrange a meeting with HR",
    "add to my calendar: dentist at 5pm",
    "add an event for tomorrow",
    "create a meeting with the team",
    "create an event called standup",
    "schedule an event for Monday",
])
def test_schedule_meeting_fast_routing_literal_signals(app_module, query):
    assert app_module.route_query(query) == "SCHEDULE_MEETING"


@pytest.mark.routing
@pytest.mark.parametrize("query", [
    "schedule me a meeting with the finance team",
    "book a call with priya friday",
    "please set up a quick sync with the team",
    "add a reminder for tomorrow morning",
])
def test_schedule_meeting_fast_routing_verb_gap_regex(app_module, query):
    assert app_module.route_query(query) == "SCHEDULE_MEETING"


@pytest.mark.routing
@pytest.mark.edge_case
def test_schedule_meeting_precedence_over_readonly_meetings(app_module):
    # "schedule a meeting" contains "meeting", also a read-only
    # MEETINGS signal. SCHEDULE_MEETING must win (checked first).
    assert app_module.route_query("schedule a meeting tomorrow") == "SCHEDULE_MEETING"


@pytest.mark.routing
@pytest.mark.edge_case
def test_readonly_appointment_question_is_not_schedule_meeting(app_module):
    # A plain lookup, not a command - must fall through to MEETINGS.
    assert app_module.route_query("what appointments do I have today") == "MEETINGS"


# =====================================================================
# 7. FAST-PATH: LEAVE_REQUEST
# =====================================================================

@pytest.mark.routing
@pytest.mark.parametrize("query", [
    "apply for leave next week",
    "apply leave for Monday",
    "request leave for 3 days",
    "take leave tomorrow",
    "book leave for Friday",
    "leave request for sick leave",
    "I need sick leave",
    "how much annual leave do I have",
    "casual leave balance",
    "vacation leave for next month",
    "what's my leave balance",
    "show my leave history",
    "how many leave days do I have left",
])
def test_leave_request_fast_routing_literal_signals(app_module, query):
    assert app_module.route_query(query) == "LEAVE_REQUEST"


@pytest.mark.routing
@pytest.mark.parametrize("query", [
    "please apply for two days of leave",
    "request some leave",
    "take a week of leave",
])
def test_leave_request_fast_routing_verb_gap_regex(app_module, query):
    assert app_module.route_query(query) == "LEAVE_REQUEST"


# =====================================================================
# 8. FAST-PATH: PO_REQUEST
# =====================================================================

@pytest.mark.routing
@pytest.mark.parametrize("query", [
    "raise a purchase order for laptops",
    "create a po for 10 monitors",
    "submit po for office chairs",
    "what's my po status",
    "show my pos",
    "check my po history",
])
def test_po_request_fast_routing_literal_signals(app_module, query):
    assert app_module.route_query(query) == "PO_REQUEST"


@pytest.mark.routing
@pytest.mark.parametrize("query", [
    "is my po approved",
    "any pending po",
    "status of po 1042",
])
def test_po_request_fast_routing_bare_po_plus_verb(app_module, query):
    assert app_module.route_query(query) == "PO_REQUEST"


@pytest.mark.routing
def test_po_request_fast_routing_verb_gap_regex(app_module):
    assert app_module.route_query("please raise a new purchase order") == "PO_REQUEST"


@pytest.mark.routing
@pytest.mark.edge_case
def test_bare_po_word_without_verb_does_not_force_po_route(app_module):
    # A bare "po" with no PO-ish verb nearby shouldn't force
    # PO_REQUEST via the word-boundary check (it may fall through to
    # WEB/CHAT/LLM depending on the rest of the sentence) - this just
    # asserts it does NOT get PO_REQUEST from a coincidental match.
    result = app_module.route_query("what does po mean")
    assert result != "PO_REQUEST"


# =====================================================================
# 9. FAST-PATH: EXPENSE_REQUEST
# =====================================================================

@pytest.mark.routing
@pytest.mark.parametrize("query", [
    "file an expense claim for travel",
    "submit an expense for the hotel",
    "claim reimbursement for my flight",
    "reimburse me for the taxi",
    "get reimbursed for lunch",
    "what's my expense report status",
    "show my expenses",
    "reimbursement status for last month",
])
def test_expense_request_fast_routing_literal_signals(app_module, query):
    assert app_module.route_query(query) == "EXPENSE_REQUEST"


@pytest.mark.routing
def test_expense_request_fast_routing_verb_gap_regex(app_module):
    assert app_module.route_query("please file a quick expense") == "EXPENSE_REQUEST"


@pytest.mark.routing
def test_expense_request_fast_routing_bare_expense_plus_verb(app_module):
    assert app_module.route_query("is my expense approved") == "EXPENSE_REQUEST"


# =====================================================================
# 10. FAST-PATH: MAIL (read-only)
# =====================================================================

@pytest.mark.routing
@pytest.mark.parametrize("query", [
    "check my email",
    "any new emails",
    "what's in my inbox",
    "check my mailbox",
    "do I have unread mail",
    "who emailed me about the invoice",
    "did Sarah email me back",
])
def test_mail_fast_routing_literal_signals(app_module, query):
    assert app_module.route_query(query) == "MAIL"


@pytest.mark.routing
@pytest.mark.parametrize("query", [
    "summarize the last received mail",
    "any new mail?",
])
def test_mail_fast_routing_bare_mail_word_boundary(app_module, query):
    assert app_module.route_query(query) == "MAIL"


@pytest.mark.routing
@pytest.mark.edge_case
def test_mailbox_lookalike_words_do_not_false_positive(app_module):
    # Word-boundary regex must not fire on "blackmail"/"chainmail" -
    # neither is a real MAIL signal on its own.
    result = app_module.route_query("what's the plot of that blackmail movie")
    assert result != "MAIL"


# =====================================================================
# 11. FAST-PATH: MEETINGS (read-only)
# =====================================================================

@pytest.mark.routing
@pytest.mark.parametrize("query", [
    "what's on my calendar today",
    "do I have any meetings tomorrow",
    "am I free at 3pm",
    "am I busy this afternoon",
    "what's my schedule for Monday",
    "show my agenda for today",
])
def test_meetings_fast_routing(app_module, query):
    assert app_module.route_query(query) == "MEETINGS"


# =====================================================================
# 12. FAST-PATH: WEB
# =====================================================================

@pytest.mark.routing
@pytest.mark.parametrize("query", [
    "who is the CEO of Tesla",
    "who was the first president of the US",
    "what's the weather in Chennai",
    "latest news on AI",
    "current price of gold",
    "what's the stock price of Apple",
    "founder of Microsoft",
    "capital of France",
    "population of India",
    "exchange rate USD to INR",
    "what's trending right now",
])
def test_web_fast_routing(app_module, query):
    assert app_module.route_query(query) == "WEB"


@pytest.mark.routing
@pytest.mark.edge_case
def test_web_word_boundary_does_not_false_positive_on_substring(app_module):
    # "born" as a bare substring shouldn't fire inside "airborne" -
    # word-boundary matching should prevent it. It's fine if this
    # doesn't land on WEB, just must not be a coincidental substring
    # hit; assert no exception and a plausible non-WEB/CHAT result
    # isn't required, just document the behavior.
    result = app_module.route_query("explain how airborne diseases spread")
    assert result != "WEB"


# =====================================================================
# 13. FOLLOW-UP INHERITANCE
# =====================================================================

@pytest.mark.routing
@pytest.mark.parametrize("previous_route", ["MAIL", "MEETINGS", "DOCUMENT", "WEB"])
def test_followup_inherits_eligible_previous_route(app_module, previous_route):
    import streamlit as st
    st.session_state["last_route"] = previous_route
    # Short fragment, no topical keyword, referential pronoun ->
    # classic follow-up shape.
    result = app_module.route_query("what about that")
    assert result == previous_route


@pytest.mark.routing
@pytest.mark.edge_case
def test_followup_does_not_inherit_from_ineligible_previous_route(app_module, mock_ollama):
    import streamlit as st
    # SEND_MAIL/SCHEDULE_MEETING are one-off actions, excluded from
    # follow-up inheritance - a vague follow-up after one of these
    # must NOT silently repeat the action, so it falls through to
    # the LLM router instead.
    st.session_state["last_route"] = "SEND_MAIL"
    mock_ollama.set_response("CHAT")
    result = app_module.route_query("what about that")
    assert result != "SEND_MAIL"


@pytest.mark.routing
@pytest.mark.edge_case
def test_self_contained_new_topic_is_not_treated_as_followup(app_module, mock_ollama):
    import streamlit as st
    st.session_state["last_route"] = "MAIL"
    mock_ollama.set_response("WEB")
    # Complete WH-question with no referential pronoun -> a genuine
    # new topic, must NOT inherit MAIL.
    result = app_module.route_query("what embedding model does nova use")
    # (This actually hits the SELF_INFO fast path before follow-up
    # logic is even reached, which is itself proof it wasn't
    # swallowed as a MAIL follow-up.)
    assert result == "SELF_INFO"


@pytest.mark.routing
def test_referential_followup_with_pronoun_is_still_a_followup(app_module):
    import streamlit as st
    st.session_state["last_route"] = "WEB"
    # Referential pronoun ("he") keeps this a follow-up even though
    # it's phrased as a WH-question with an auxiliary verb.
    result = app_module.route_query("is he still racing")
    assert result == "WEB"


# =====================================================================
# 14. LLM ROUTER FALLBACK (label -> route mapping)
# =====================================================================

@pytest.mark.routing
@pytest.mark.parametrize("llm_label,expected_route", [
    ("DOCUMENT", "DOCUMENT"),
    ("PLAN", "PLAN"),
    ("SEND_MAIL", "SEND_MAIL"),
    ("SEND MAIL", "SEND_MAIL"),
    ("SCHEDULE_MEETING", "SCHEDULE_MEETING"),
    ("LEAVE_REQUEST", "LEAVE_REQUEST"),
    ("PO_REQUEST", "PO_REQUEST"),
    ("PO REQUEST", "PO_REQUEST"),
    ("EXPENSE_REQUEST", "EXPENSE_REQUEST"),
    ("MAIL", "MAIL"),
    ("MEETINGS", "MEETINGS"),
    ("WEB", "WEB"),
    ("SOMETHING_UNRECOGNIZED", "CHAT"),
    ("", "CHAT"),
])
def test_llm_fallback_label_mapping(app_module, mock_ollama, llm_label, expected_route):
    mock_ollama.set_response(llm_label)
    # No fast-path keyword at all, so this MUST reach the LLM fallback.
    result = app_module.route_query("tell me something interesting")
    assert result == expected_route
    assert len(mock_ollama.calls) == 1


@pytest.mark.routing
@pytest.mark.edge_case
def test_llm_fallback_precedence_send_mail_over_mail(app_module, mock_ollama):
    # Label containing both "SEND_MAIL"-ish and "MAIL" text must
    # resolve to SEND_MAIL, since that check runs first.
    mock_ollama.set_response("SEND_MAIL (compose a new message)")
    result = app_module.route_query("tell me something interesting")
    assert result == "SEND_MAIL"


@pytest.mark.routing
@pytest.mark.edge_case
def test_llm_fallback_precedence_schedule_over_meeting(app_module, mock_ollama):
    mock_ollama.set_response("SCHEDULE_MEETING")
    result = app_module.route_query("tell me something interesting")
    assert result == "SCHEDULE_MEETING"


# =====================================================================
# 15. FAILURE INJECTION (router-level)
# =====================================================================

@pytest.mark.routing
@pytest.mark.failure
def test_router_falls_back_to_chat_on_ollama_timeout(app_module, mock_ollama):
    mock_ollama.set_error(TimeoutError("ollama timed out"))
    result = app_module.route_query("tell me something interesting")
    assert result == "CHAT"


@pytest.mark.routing
@pytest.mark.failure
def test_router_falls_back_to_chat_on_connection_error(app_module, mock_ollama):
    mock_ollama.set_error(ConnectionError("ollama unreachable"))
    result = app_module.route_query("tell me something interesting")
    assert result == "CHAT"


@pytest.mark.routing
@pytest.mark.failure
def test_router_falls_back_to_chat_on_malformed_llm_json(app_module, mock_ollama):
    # response.json() itself raising (malformed body) should be
    # caught by the same except-branch as a network failure.
    mock_ollama.set_error(ValueError("Expecting value: line 1 column 1"))
    result = app_module.route_query("tell me something interesting")
    assert result == "CHAT"


# =====================================================================
# 16. INVALID / EDGE-CASE INPUT
# =====================================================================

@pytest.mark.routing
@pytest.mark.edge_case
def test_empty_string_query_does_not_crash(app_module, mock_ollama):
    mock_ollama.set_response("CHAT")
    result = app_module.route_query("")
    assert isinstance(result, str)


@pytest.mark.routing
@pytest.mark.edge_case
def test_whitespace_only_query_does_not_crash(app_module, mock_ollama):
    mock_ollama.set_response("CHAT")
    result = app_module.route_query("     ")
    assert isinstance(result, str)


@pytest.mark.routing
@pytest.mark.edge_case
def test_very_long_query_does_not_crash(app_module, mock_ollama):
    mock_ollama.set_response("WEB")
    long_query = "please tell me about " + ("the weather " * 500)
    result = app_module.route_query(long_query)
    assert isinstance(result, str)


@pytest.mark.routing
@pytest.mark.edge_case
def test_mixed_case_and_punctuation_still_routes(app_module):
    assert app_module.route_query("  SCHEDULE A MEETING with Bob!!  ") == "SCHEDULE_MEETING"


@pytest.mark.routing
@pytest.mark.edge_case
def test_query_with_special_characters_does_not_crash(app_module, mock_ollama):
    mock_ollama.set_response("CHAT")
    result = app_module.route_query("<script>alert('x')</script> && rm -rf /")
    assert isinstance(result, str)


@pytest.mark.routing
@pytest.mark.edge_case
def test_ambiguous_multi_domain_single_topic_query_is_deterministic(app_module):
    # "book a call with legal about the po" - contains SCHEDULE_MEETING
    # signal ("book a call") AND a po-ish word. SCHEDULE_MEETING is
    # checked well before PO_REQUEST, so this must be deterministic,
    # not order-dependent on dict iteration.
    result = app_module.route_query("book a call with legal about the po")
    assert result == "SCHEDULE_MEETING"
