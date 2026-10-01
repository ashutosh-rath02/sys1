"""Signals for phishing that code can compute exactly, so the model is
never asked to guess at them.

Whether a sender's domain matches the link's domain is a string
comparison, not a judgment. Asking a 23M-parameter encoder to infer it
from prose wastes the one thing the model is for. This is the same
division of labour as the Snake demo: deterministic rules stay in code,
the model gets the part that actually needs semantics (is this message
manufacturing urgency? is it a credential lure?).

Kept dependency-free -- no tldextract -- so inference stays cheap. The
public-suffix handling is deliberately shallow; see `registered_domain`.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

FREE_HOSTING_HOSTS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "rebrand.ly", "cutt.ly", "shorturl.at", "rb.gy",
    "github.io", "gitlab.io", "pages.dev", "netlify.app", "vercel.app",
    "firebaseapp.com", "web.app", "herokuapp.com", "glitch.me",
    "weebly.com", "wixsite.com", "blogspot.com", "wordpress.com",
    "000webhostapp.com", "repl.co", "replit.app", "surge.sh",
    "docs.google.com", "drive.google.com", "forms.gle", "sites.google.com",
    "dropbox.com", "onedrive.live.com", "ipfs.io", "dweb.link",
    "s3.amazonaws.com", "blob.core.windows.net",
}

WEBMAIL_DOMAINS = {
    "gmail.com", "outlook.com", "hotmail.com", "yahoo.com", "aol.com",
    "icloud.com", "proton.me", "protonmail.com", "gmx.com", "mail.com",
    "yandex.com", "zoho.com", "live.com", "msn.com",
}

# Two-level public suffixes where the registered domain needs three labels
# (foo.co.uk, not co.uk). Not exhaustive -- a full public-suffix list is a
# dependency this doesn't need.
_TWO_LEVEL_SUFFIXES = {
    "co.uk", "org.uk", "ac.uk", "gov.uk", "co.jp", "co.kr", "co.in",
    "com.au", "net.au", "org.au", "com.br", "com.cn", "com.mx", "co.za",
    "co.nz", "com.sg", "com.tr",
}


def registered_domain(host: str) -> str:
    """eTLD+1, approximately. Shallow by design -- see module docstring."""
    host = (host or "").lower().strip().rstrip(".")
    if not host or _is_ip(host):
        return host
    parts = host.split(".")
    if len(parts) < 2:
        return host
    if ".".join(parts[-2:]) in _TWO_LEVEL_SUFFIXES and len(parts) >= 3:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def _is_ip(host: str) -> bool:
    return bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host or ""))


def link_host(url: str) -> str:
    if not url:
        return ""
    if "://" not in url:
        url = "http://" + url
    return (urlparse(url).hostname or "").lower()


def sender_domain(from_address: str) -> str:
    if not from_address or "@" not in from_address:
        return ""
    return from_address.rsplit("@", 1)[-1].lower().strip().strip(">").strip()


def extract(email: dict) -> dict[str, float]:
    """The mechanically-determinable signals, as 0/1 features."""
    from_addr = email.get("from", "") or ""
    url = email.get("link_url", "") or ""

    s_domain = sender_domain(from_addr)
    l_host = link_host(url)
    s_reg = registered_domain(s_domain)
    l_reg = registered_domain(l_host)

    is_webmail = s_reg in WEBMAIL_DOMAINS
    free_host = any(l_reg == h or l_host.endswith("." + h) or l_host == h for h in FREE_HOSTING_HOSTS)

    # A webmail sender never "matches" a link domain in a meaningful way,
    # so only count a mismatch when the sender has a real org domain --
    # otherwise this signal just restates generic_sender.
    domain_mismatch = bool(s_reg and l_reg and not is_webmail and s_reg != l_reg)

    return {
        "sig_domain_mismatch": float(domain_mismatch),
        "sig_free_hosting": float(free_host),
        "sig_generic_sender": float(is_webmail),
        "sig_ip_literal": float(_is_ip(l_host)),
        "sig_http_only": float(url.lower().startswith("http://")),
    }


FEATURE_NAMES = [
    "sig_domain_mismatch",
    "sig_free_hosting",
    "sig_generic_sender",
    "sig_ip_literal",
    "sig_http_only",
]
