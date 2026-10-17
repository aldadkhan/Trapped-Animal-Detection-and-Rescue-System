"""Send rescue alerts directly via SMTP (no Outlook / mail app popup)."""
import os
import smtplib
import ssl
from email.message import EmailMessage


def _get(name, default=""):
    try:
        import streamlit as st
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # noqa: BLE001
        pass
    return os.environ.get(name, default)


def smtp_config():
    return {
        "host": _get("SMTP_HOST", "smtp.gmail.com"),
        "port": int(_get("SMTP_PORT", "465") or 465),
        "user": _get("SMTP_USER"),
        "password": _get("SMTP_PASSWORD"),
        "sender": _get("SMTP_SENDER") or _get("SMTP_USER"),
        "rescue_to": _get("RESCUE_TEAM_EMAIL"),
    }


def is_configured(cfg=None):
    cfg = cfg or smtp_config()
    return bool(cfg["user"] and cfg["password"])


def send_rescue_email(to, subject, body, attachments=(), cfg=None):
    """attachments: iterable of (filename, bytes, maintype, subtype)."""
    cfg = cfg or smtp_config()
    if not is_configured(cfg):
        raise RuntimeError("Email is not configured. Fill .streamlit/secrets.toml (see README).")
    msg = EmailMessage()
    msg["From"] = cfg["sender"]
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    for name, data, main, sub in attachments:
        msg.add_attachment(data, maintype=main, subtype=sub, filename=name)

    context = ssl.create_default_context()
    if cfg["port"] == 465:
        with smtplib.SMTP_SSL(cfg["host"], cfg["port"], context=context, timeout=20) as s:
            s.login(cfg["user"], cfg["password"])
            s.send_message(msg)
    else:
        with smtplib.SMTP(cfg["host"], cfg["port"], timeout=20) as s:
            s.starttls(context=context)
            s.login(cfg["user"], cfg["password"])
            s.send_message(msg)
