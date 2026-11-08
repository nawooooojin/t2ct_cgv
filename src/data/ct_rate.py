"""CT-RATE helpers: split a challenge prompt into the (findings, impression) pair used in training."""

from __future__ import annotations

import re

_FINDINGS = re.compile(r"\bFindings\s*:", re.IGNORECASE)
_IMPRESSION = re.compile(r"\bImpression[s]?\s*:", re.IGNORECASE)


def split_report(report: str) -> tuple[str, str]:
    """``"<preamble> Findings: ... Impression: ..."`` -> (findings, impression).

    The preamble (e.g. age and sex) is dropped, matching the CT-RATE ``Findings_EN`` /
    ``Impressions_EN`` columns the model was trained on. Without either marker the whole text is
    the findings.
    """
    text = report.strip()
    m_f = _FINDINGS.search(text)
    m_i = _IMPRESSION.search(text, m_f.end() if m_f else 0)
    if m_f is None and m_i is None:
        return text, ""
    findings = text[(m_f.end() if m_f else 0):(m_i.start() if m_i else len(text))].strip()
    return findings, text[m_i.end():].strip() if m_i else ""
