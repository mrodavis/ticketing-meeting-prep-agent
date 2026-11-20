"""Handling for untrusted email text: PII redaction before any model sees it, and
a heuristic prompt-injection check that is OR'd with the model's own flag."""

from __future__ import annotations

import re

_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def redact(text: str) -> tuple[str, list[str]]:
    """Mask card numbers (Luhn-valid) and SSNs. Returns the text and what was redacted."""
    found: list[str] = []

    def card(m: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            found.append("card_number")
            return "[REDACTED CARD]"
        return m.group(0)

    text = _CARD_RE.sub(card, text)
    if _SSN_RE.search(text):
        found.append("ssn")
        text = _SSN_RE.sub("[REDACTED SSN]", text)
    return text, found


_INJECTION_PATTERNS = [
    r"ignore (all |your |the )?(previous|prior|above) (instructions|rules)",
    r"\b(system|developer) (note|prompt|message|instruction)s?\b",
    r"\bnote to the (ai|assistant|model)\b",
    r"\b(ai|assistant|chatbot|language model)\b.{0,40}\b(must|should|will) (state|say|write|include|email|send)\b",
    r"\bdisregard (your|the) (guidelines|policy|rules)\b",
    r"\byou are now\b",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE | re.DOTALL)


def injection_heuristic(text: str) -> bool:
    return bool(_INJECTION_RE.search(text))
