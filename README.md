<div align="center">

# ✦ N O V A ✦
### *the assistant that actually gets things done*

**Ask it a question. Hand it your inbox. Let it fight your calendar. Trust it with a purchase order.**
**One Streamlit app. Four agents. Zero patience for busywork.**

</div>

---

## 🪐 What even is this

Picture a chatbot. Now picture it with a memory that doesn't goldfish after three messages, a librarian's brain full of *your* documents, and the audacity to actually **send emails**, **check your calendar**, **file your leave**, and **route purchase orders through an approval chain** — all because you typed a sentence at it.

That's NOVA. It's not roleplaying as an assistant. It's doing the job.

Under the hood it picks its brain on the fly — **Gemini** in the cloud, **Groq** for stupid-fast inference, or **Ollama** running fully local so nothing ever leaves your machine. Same NOVA, three different engines, your call.

🔗 **[Live Demo](#)** — swap this for your deployed link and go show someone

---

## ⚡ The Agents (the actual main character energy)

NOVA isn't one brain doing five jobs badly. It's one brain that knows *which* job it's doing and calls in the right specialist.

### 💬 The Conversationalist
The base layer. Fast, streaming, remembers what you said five minutes ago, and throws you smart follow-up questions so you never hit a dead end. Doesn't know the answer? It goes and **web-searches** instead of making something up.

### 📚 The Librarian — RAG Knowledge Base
Feed NOVA documents. It chunks them, embeds them with **Qwen3 Embedding 0.6B**, files them away in **ChromaDB**, and pulls the exact right passage back the instant you ask a question — grounded, cited, no hallucinated nonsense.

```
Document → Text Extraction → Chunking → Qwen3 Embedding 0.6B → ChromaDB → Semantic Retrieval → Qwen 2.5 3B → Answer
```

This is a **permanent, admin-curated knowledge base** — the documents that live here are the source of truth for everyone, every session, forever (or until an admin says otherwise).
Supports: **PDF · TXT · DOCX · CSV · XLSX**

### 📧 The Correspondent — Mail Agent
Reads your inbox (IMAP), drafts and fires off emails (SMTP), HTML and all. It's also the quiet engine behind every approval email NOVA sends — which brings us to...

### 📅 The Scheduler — Meetings Agent
Reads one or more `.ics` calendars, checks who's free across a whole group of people, and books the meeting. No more "does Tuesday work for everyone" spirals.

### 🏖️ The HR Rep — Leave Agent
Say "I need Friday off" and NOVA parses the dates, the type of leave, the reason — checks it against holidays, max leave span, and who's actually eligible — then routes it to an approver by email and tracks the balance and history so nobody has to remember anything.

### 📝 The Procurement Officer — PO Agent
The big one. Tell it what to buy and from whom, and it:

- **Validates** everything — required fields, sane quantities and prices, a hard cap per PO, and catches accidental double-submissions
- **Cross-checks reality** — is this vendor even approved? Does this blow the department's budget? Is the price wildly off the reference catalog?
- **Routes for approval** — pending POs get emailed to the approver with real, clickable, cryptographically-signed **✓ APPROVE** / **✕ REJECT** buttons. One click, right from Gmail or Outlook, no separate server, no login wall.
- **Only pays out on approval** — the vendor is emailed *after* the approver says yes, and never if they say no. Every step, every timestamp, every note — logged.

---

## 🧬 The Stack

| Layer | Tech |
|---|---|
| App | Python · Streamlit |
| Cloud brains | Google Gemini · Groq (`openai/gpt-oss-120b`) |
| Local brain | Ollama — Qwen 2.5 3B |
| Embeddings | Qwen3 Embedding 0.6B |
| Vector store | ChromaDB |
| Document parsing | PyPDF · python-docx · pandas |

---

## 🚀 Getting It Running

**1. Clone it**
```bash
git clone https://github.com/kishorecodes404/ProCode-basic-chatbot.git
cd ProCode-basic-chatbot
```

**2. Give it a home**
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**3. Feed it dependencies**
```bash
pip install streamlit google-genai python-dotenv pypdf chromadb requests pandas python-docx
```

### 🦙 Local mode (Ollama)

```bash
ollama pull qwen2.5:3b
ollama pull qwen3-embedding:0.6b
```

Ollama has to actually be *running* before you flip NOVA into local mode. Not optional.

### 🔐 Wire up the brain and the agents

Drop a `.env` in the project root. `GEMINI_API_KEY` is the only non-negotiable — everything past that just unlocks an extra agent.

```env
# --- Core AI ---
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here          # optional, unlocks Groq mode

# --- Admin access ---
ADMIN_PASSWORD=your_admin_password           # gates the admin knowledge base

# --- Mail Agent (IMAP read / SMTP send) ---
NOVA_IMAP_HOST=imap.gmail.com
NOVA_IMAP_USER=you@example.com
NOVA_IMAP_PASSWORD=your_app_password
NOVA_IMAP_FOLDER=INBOX
NOVA_SMTP_HOST=smtp.gmail.com
NOVA_SMTP_PORT=587
NOVA_SMTP_USER=you@example.com
NOVA_SMTP_PASSWORD=your_app_password

# --- Meetings Agent ---
NOVA_MEETINGS_ICS_PATH=/path/to/calendar.ics
NOVA_MEETINGS_ICS_PATHS=/path/to/other1.ics,/path/to/other2.ics   # optional, extra calendars

# --- Leave Agent ---
NOVA_LEAVE_STORE_PATH=leave_store.json
NOVA_LEAVE_MAX_SPAN_DAYS=30
NOVA_LEAVE_APPROVER_EMAIL=manager@example.com
NOVA_LEAVE_ELIGIBLE_USERS=alice,bob          # optional, restricts who can request leave
NOVA_LEAVE_HOLIDAYS=2026-01-01,2026-12-25    # optional, comma-separated dates
NOVA_LEAVE_USER_NAME=Your Display Name

# --- Purchase Order (PO) Agent ---
NOVA_PO_STORE_PATH=po_store.json
NOVA_PO_APPROVER_EMAIL=approver@example.com
NOVA_PO_APPROVAL_BASE_URL=http://localhost:8501   # public URL if the approver isn't on this machine
NOVA_PO_APPROVAL_SECRET=change-this-to-a-random-secret
NOVA_PO_DEFAULT_VENDOR_EMAIL=vendor@example.com
NOVA_PO_USER_NAME=Your Display Name
NOVA_PO_MAX_AMOUNT=500000                    # optional, hard cap per PO
NOVA_PO_AUTO_APPROVE_THRESHOLD=5000          # optional, auto-approve at/under this amount
NOVA_PO_PRICE_VARIANCE_PCT=20                # optional, catalog price deviation warning threshold
NOVA_PO_BUDGET_LIMITS=it:200000,sales:100000 # optional, per-department budget caps
NOVA_PO_ITEM_CATALOG=laptop:45000,chair:3500 # optional, reference prices for variance checks
NOVA_PO_VENDOR_MASTER=reliance digital,dell  # optional, approved-vendor allowlist
NOVA_PO_ELIGIBLE_USERS=alice,bob             # optional, restricts who can raise POs
```

> ⚠️ **`.env` stays out of git. Always.** Use app passwords for IMAP/SMTP, not your real password — and actually randomize `NOVA_PO_APPROVAL_SECRET`, it's signing your approval links.

### 🎬 Action

```bash
streamlit run app.py
```

NOVA takes the stage at **http://localhost:8501**

---

## 🕰️ How We Got Here

| | |
|---|---|
| **V1** | A Gemini chatbot. Humble beginnings. |
| **V2** | It remembered things. Revolutionary, apparently. |
| **V3** | RAG arrives — NOVA starts reading documents instead of guessing. |
| **V4** | Goes fully local with Ollama + Qwen embeddings, admin knowledge base born. |
| **V5** | NOVA stops being a chatbot and starts being *staff* — Mail, Meetings, Leave, and Purchase Order agents, plus Groq mode and one-click email approvals. |

---

<div align="center">

### Built by **Kishore M S**
*NOVA doesn't just talk. It handles it.*

</div>
