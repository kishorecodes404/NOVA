"""
Standalone diagnostic - tests each provider's API key directly,
completely outside the NOVA app, so we can see the REAL error
message instead of just "403 Forbidden".

Run this in the same terminal/venv where you run streamlit, so it
picks up the same .env / env vars.
"""




# GitHub achievement test










import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("GROQ")
print("=" * 60)

groq_key = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY")
print(f"Key found: {bool(groq_key)}")
if groq_key:
    print(f"Key starts with: {groq_key[:8]}...")

if groq_key:
    import requests
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
        json={"model": "openai/gpt-oss-120b", "messages": [{"role": "user", "content": "hi"}]},
        timeout=30,
    )
    print(f"Status: {r.status_code}")
    print(f"Body: {r.text[:500]}")

print()
print("=" * 60)
print("GEMINI")
print("=" * 60)

gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
print(f"Key found: {bool(gemini_key)}")
if gemini_key:
    print(f"Key starts with: {gemini_key[:8]}...")

if gemini_key:
    import requests
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_key}",
        json={"contents": [{"parts": [{"text": "hi"}]}]},
        timeout=30,
    )
    print(f"Status: {r.status_code}")
    print(f"Body: {r.text[:500]}")

print()
print("=" * 60)
print("SMTP (mail send)")
print("=" * 60)

smtp_host = os.getenv("NOVA_SMTP_HOST", "")
smtp_port = int(os.getenv("NOVA_SMTP_PORT", "587") or "587")
smtp_user = os.getenv("NOVA_SMTP_USER", "") or os.getenv("NOVA_IMAP_USER", "")
smtp_password = os.getenv("NOVA_SMTP_PASSWORD", "") or os.getenv("NOVA_IMAP_PASSWORD", "")

print(f"Host: {smtp_host or '(not set)'}")
print(f"Port: {smtp_port}")
print(f"User: {smtp_user or '(not set)'}")
print(f"Password found: {bool(smtp_password)}")

if not (smtp_host and smtp_user and smtp_password):
    print("SKIPPED: set NOVA_SMTP_HOST, NOVA_SMTP_USER, NOVA_SMTP_PASSWORD "
          "(or NOVA_IMAP_USER/PASSWORD as a fallback) to test this.")
else:
    import smtplib
    try:
        # Port 465 = implicit TLS (SMTP_SSL). Port 587 (or anything
        # else) = plaintext then STARTTLS. Using the wrong one for
        # the port is a very common cause of a TLS handshake error
        # that looks unrelated to credentials.
        if smtp_port == 465:
            server = smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=20)
        else:
            server = smtplib.SMTP(smtp_host, smtp_port, timeout=20)

        with server:
            if smtp_port != 465:
                server.starttls()
            server.login(smtp_user, smtp_password)
            print("Login succeeded - SMTP credentials and connection are good.")

    except smtplib.SMTPAuthenticationError as e:
        print(f"AUTH FAILED: {e}")
        print("-> Most providers (Gmail, Outlook, Yahoo) reject your normal "
              "account password here. Generate an app-specific password "
              "(requires 2FA enabled on the account) and use that instead.")
    except Exception as e:
        print(f"FAILED: {e}")

print()
print("=" * 60)
print("OLLAMA")
print("=" * 60)

if True:
    import requests
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "qwen2.5:1.5b", "prompt": "hi", "stream": False},
            timeout=15,
        )
        print(f"Status: {r.status_code}")
        print(f"Body: {r.text[:300]}")
    except Exception as e:
        print(f"FAILED: {e}")