"""JSON, Markdown, standalone HTML, and UI-fragment renderers."""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Collection, Iterable

from .types import FinalJudgeReport, JudgeFinding

_DECISION_TEXT = {
    "accepted": (
        "Accepted",
        "ok",
        "No concerns, no code-verified flags, and every critic had evidence.",
    ),
    "accepted_with_concerns": (
        "Accepted with concerns",
        "warn",
        "Nothing blocks the model, but there are concerns, data flags, missing "
        "evidence, or unevaluated critics to review.",
    ),
    "requires_reaudit": (
        "Re-audit required",
        "alert",
        "At least one high-severity finding must be confirmed or refuted before "
        "this result is trusted.",
    ),
    "rejected": (
        "Rejected",
        "bad",
        "An established, high-confidence defect undermines the model's conclusions.",
    ),
    "insufficient_evidence": (
        "Insufficient evidence",
        "neutral",
        "Too few critics produced a usable verdict. This is not a pass.",
    ),
}

_SPECIAL_NAMES = {"devils_advocate": "Devil's advocate"}

_REPORT_CSS = """
:root{--bg:#fafaf7;--fg:#1d1d1b;--muted:#6b6a65;--card:#ffffff;--line:#e4e2dc;
--ok:#1f7a4d;--ok-bg:#e6f4ec;--warn:#8a5d00;--warn-bg:#fdf3d9;
--alert:#b4511c;--alert-bg:#fbe9dd;--bad:#a42626;--bad-bg:#f9e1e1;
--neutral:#55544f;--neutral-bg:#ecebe6}
@media (prefers-color-scheme: dark){:root{--bg:#161614;--fg:#ecebe6;--muted:#a3a29b;
--card:#201f1d;--line:#363430;--ok:#6fd3a0;--ok-bg:#16301f;--warn:#e8bf5a;--warn-bg:#33290f;
--alert:#f0a273;--alert-bg:#3a2316;--bad:#ef8a8a;--bad-bg:#3b1a1a;
--neutral:#bdbcb5;--neutral-bg:#2a2927}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:920px;margin:0 auto;padding:32px 20px 64px}
h1{font-size:1.5rem;margin:0 0 4px}
h2{font-size:1.1rem;margin:32px 0 6px}
h3{margin:0 0 6px;font-size:1rem}
.sub{color:var(--muted);font-size:.9rem;margin:0 0 20px}
.banner{border:1px solid;border-radius:12px;padding:16px 20px;margin:0 0 20px}
.banner .label{font-size:1.35rem;font-weight:650;margin:0}
.banner p{margin:4px 0 0}
.banner.ok{background:var(--ok-bg);border-color:var(--ok);color:var(--ok)}
.banner.warn{background:var(--warn-bg);border-color:var(--warn);color:var(--warn)}
.banner.alert{background:var(--alert-bg);border-color:var(--alert);color:var(--alert)}
.banner.bad{background:var(--bad-bg);border-color:var(--bad);color:var(--bad)}
.banner.neutral{background:var(--neutral-bg);border-color:var(--neutral);color:var(--neutral)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:0 0 8px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px}
.stat b{display:block;font-size:1.3rem}
.stat span{color:var(--muted);font-size:.82rem}
.card{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--neutral);
border-radius:10px;padding:14px 16px;margin:0 0 12px}
.card.sev-critical,.card.sev-high{border-left-color:var(--bad)}
.card.sev-medium{border-left-color:var(--alert)}
.card.sev-low{border-left-color:var(--warn)}
.badge{display:inline-block;font-size:.75rem;padding:2px 8px;border-radius:999px;
background:var(--neutral-bg);color:var(--neutral);margin:0 4px 4px 0}
.badge.sev-critical,.badge.sev-high{background:var(--bad-bg);color:var(--bad)}
.badge.sev-medium{background:var(--alert-bg);color:var(--alert)}
.badge.sev-low{background:var(--warn-bg);color:var(--warn)}
.finding{margin:0 0 8px;white-space:pre-wrap}
dl{margin:8px 0 0}
dt{font-weight:600;font-size:.85rem;margin-top:8px;color:var(--muted)}
dd{margin:2px 0 0;white-space:pre-wrap}
ul.flags,ol.steps{padding-left:22px;margin:0}
ul.flags li,ol.steps li{margin:0 0 8px}
pre.desc{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;
white-space:pre-wrap;font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;overflow-x:auto}
table{border-collapse:collapse;width:100%;margin:8px 0}
th,td{border:1px solid var(--line);padding:6px 8px;text-align:left}
th{background:var(--neutral-bg);color:var(--neutral)}
footer{margin-top:40px;color:var(--muted);font-size:.82rem;border-top:1px solid var(--line);padding-top:14px}
"""


def _e(value: Any) -> str:
    """HTML-escape any value. All model-generated text goes through this."""
    return html.escape("" if value is None else str(value), quote=True)


def _pretty(name: Any) -> str:
    text = str(name)
    return _SPECIAL_NAMES.get(text) or text.replace("_", " ").capitalize()


def _number(value: Any, digits: int = 4) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.{digits}f}"
    return "—"


def _finding_card(finding: JudgeFinding) -> str:
    severity = finding.severity
    rows = []
    for label, value in (
        ("Evidence", finding.evidence),
        ("Suggested fix", finding.suggested_fix),
        ("Falsification test", finding.falsification_test),
    ):
        rows.append(
            f"<dt>{_e(label)}</dt><dd>{_e(value) if value else '<em>not given</em>'}</dd>"
        )
    return (
        f'<div class="card sev-{_e(severity)}">'
        f"<h3>{_e(_pretty(finding.critic))}</h3>"
        f'<p class="badge sev-{_e(severity)}">{_e(severity)}</p>'
        f'<p class="badge">confidence: {_e(finding.confidence)}</p>'
        f'<p class="badge">{_e(finding.status)}</p>'
        f'<p class="badge">{_e(finding.defect_class)}</p>'
        f'<p class="finding">{_e(finding.finding) or "<em>No finding text.</em>"}</p>'
        f"<dl>{''.join(rows)}</dl></div>"
    )


def _findings_section(
    title: str,
    blurb: str,
    findings: Iterable[JudgeFinding],
) -> str:
    items = list(findings)
    if not items:
        return ""
    cards = "".join(_finding_card(item) for item in items)
    return (
        f"<h2>{_e(title)} ({len(items)})</h2>"
        f"<p>{_e(blurb)}</p>{cards}"
    )


def _stats(report: FinalJudgeReport) -> list[tuple[str, str]]:
    return [
        (
            f"{report.n_critics_evaluated}/{report.min_required}",
            "critics evaluated / minimum",
        ),
        (str(len(report.vetoes) + len(report.pending_reaudit)), "blocking findings"),
        (str(len(report.concerns)), "concerns"),
        (str(len(report.code_verified_flags)), "code-verified flags"),
        (str(len(report.unevaluated_critics)), "unevaluated critics"),
        (str(len(report.missing_evidence)), "missing evidence items"),
    ]


def _model_table(report: FinalJudgeReport) -> str:
    summary = report.model_summary
    if not summary:
        return ""
    rows = [
        ("Target", summary.get("target") or "—"),
        ("Target kind", summary.get("target_kind") or "—"),
        ("Metric", summary.get("metric") or "—"),
        ("Selected family", summary.get("selected_family") or "none"),
        (
            "Features",
            ", ".join(str(item) for item in summary.get("selected_features", []) or [])
            or "—",
        ),
        ("Candidates", summary.get("candidate_count", "—")),
        ("Validation", summary.get("validation_verdict") or "not run"),
    ]
    body = "".join(
        f"<tr><th>{_e(label)}</th><td>{_e(value)}</td></tr>" for label, value in rows
    )
    return f"<h2>Selected model</h2><table>{body}</table>"


def _flags_section(report: FinalJudgeReport) -> str:
    if not report.code_verified_flags:
        return ""
    items = "".join(f"<li>{_e(flag)}</li>" for flag in report.code_verified_flags)
    return (
        "<h2>Code-verified flags</h2>"
        "<p>Computed deterministically in code, not by the language model.</p>"
        f'<ul class="flags">{items}</ul>'
    )


def _unevaluated_section(report: FinalJudgeReport) -> str:
    if not report.unevaluated_critics and not report.missing_evidence:
        return ""
    parts = []
    if report.unevaluated_critics:
        chips = "".join(
            f"<li>{_e(_pretty(name))}</li>" for name in report.unevaluated_critics
        )
        parts.append(
            "<p>No usable verdict: no evidence was supplied, or the critic call "
            f'failed. Absence of evidence is not a pass.</p><ul class="flags">{chips}</ul>'
        )
    if report.missing_evidence:
        chips = "".join(
            f"<li>{_e(item)}</li>" for item in report.missing_evidence
        )
        parts.append(f'<ul class="flags">{chips}</ul>')
    return f"<h2>Unevaluated critics and missing evidence</h2>{''.join(parts)}"


def _steps_section(report: FinalJudgeReport) -> str:
    if not report.reaudit_instructions:
        return ""
    items = "".join(
        f'<li><b>{_e(_pretty(item.get("from")))}:</b> {_e(item.get("fix"))}'
        f'<br><em>Test: {_e(item.get("test"))}</em></li>'
        for item in report.reaudit_instructions
    )
    return (
        "<h2>Next steps</h2>"
        "<p>Each step comes with a test that would confirm or refute the finding.</p>"
        f'<ol class="steps">{items}</ol>'
    )


def _devils_section(report: FinalJudgeReport) -> str:
    summary = report.devils_advocate_summary
    if not summary:
        return ""
    if summary.get("evaluated"):
        text = (
            f"severity {summary.get('severity')}, confidence "
            f"{summary.get('confidence')}, {summary.get('status')}, "
            f"{summary.get('defect_class')}"
        )
    else:
        text = "did not produce a usable verdict"
    return (
        "<h2>Devil's advocate</h2>"
        f"<p>Runs last and looks for what the other critics missed: {_e(text)}.</p>"
    )


def render_fragment(report: FinalJudgeReport) -> str:
    """Render semantic HTML for ``QTextBrowser``/the UI.

    No CSS, JavaScript, external resources, or ``<details>`` elements: the UI
    applies its own stylesheet and Qt's rich-text engine supports only a subset
    of HTML.
    """
    label, tone, blurb = _DECISION_TEXT.get(
        report.decision, (report.decision, "neutral", "")
    )
    stats = "".join(
        f"<tr><th>{_e(label_text)}</th><td>{_e(value)}</td></tr>"
        for value, label_text in _stats(report)
    )
    sections = [
        f"<h1>Final model judgement</h1>",
        f"<p>{_e(report.generated_at)}</p>",
        f'<div class="banner {_e(tone)}">'
        f"<p><b>{_e(label)}</b></p><p>{_e(blurb)}</p></div>",
        f"<h2>Audit summary</h2><table>{stats}</table>",
        _model_table(report),
        _flags_section(report),
        _findings_section(
            "Vetoes",
            "Established, high-confidence defects, or high-severity findings "
            "corroborated by more than one critic.",
            report.vetoes,
        ),
        _findings_section(
            "Pending re-audit",
            "High-severity findings that need to be confirmed or refuted before "
            "the result is trusted.",
            report.pending_reaudit,
        ),
        _findings_section(
            "Concerns",
            "Medium-or-worse findings that do not block, but deserve a look.",
            report.concerns,
        ),
        _findings_section(
            "Advisories",
            "Lower-severity or lower-confidence observations.",
            report.advisories,
        ),
        _unevaluated_section(report),
        _devils_section(report),
        _steps_section(report),
    ]
    if report.note:
        sections.append(f"<p><em>{_e(report.note)}</em></p>")
    return "".join(section for section in sections if section)


def render_document(report: FinalJudgeReport) -> str:
    """Render one self-contained HTML page with the standalone stylesheet."""
    body = render_fragment(report)
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="color-scheme" content="light dark"><title>Final model judgement</title>'
        f"<style>{_REPORT_CSS}</style></head><body><main>{body}</main>"
        "<footer>Final-judge findings are written by a language model and can be "
        "wrong; code-verified flags are computed deterministically. This report is "
        "advisory and does not replace the statistical validation.</footer>"
        "</body></html>\n"
    )


def _markdown_finding(finding: JudgeFinding) -> list[str]:
    lines = [
        f"### {_pretty(finding.critic)}",
        "",
        f"- Severity: **{finding.severity}**",
        f"- Confidence: **{finding.confidence}**",
        f"- Status: **{finding.status}**",
        f"- Defect class: **{finding.defect_class}**",
        "",
        finding.finding or "_No finding text._",
        "",
    ]
    for label, value in (
        ("Evidence", finding.evidence),
        ("Suggested fix", finding.suggested_fix),
        ("Falsification test", finding.falsification_test),
    ):
        lines.append(f"**{label}:** {value or '_not given_'}")
        lines.append("")
    return lines


def _markdown_section(
    title: str,
    findings: Iterable[JudgeFinding],
) -> list[str]:
    items = list(findings)
    if not items:
        return []
    lines = [f"### {title} ({len(items)})", ""]
    for item in items:
        lines.extend(_markdown_finding(item))
    return lines


def render_markdown(
    report: FinalJudgeReport,
    *,
    heading_level: int = 1,
) -> str:
    label, _, blurb = _DECISION_TEXT.get(
        report.decision, (report.decision, "neutral", "")
    )
    heading = "#" * max(1, min(6, heading_level))
    lines = [
        f"{heading} Final model judgement",
        "",
        f"- Generated: {report.generated_at or 'unknown'}",
        f"- Decision: **{label}** — {blurb}",
        f"- Critics evaluated: {report.n_critics_evaluated}/{report.min_required}",
        f"- Blocking findings: {len(report.vetoes) + len(report.pending_reaudit)}",
        f"- Concerns: {len(report.concerns)}",
        f"- Code-verified flags: {len(report.code_verified_flags)}",
        f"- Unevaluated critics: {len(report.unevaluated_critics)}",
        "",
    ]
    summary = report.model_summary
    if summary:
        lines.extend(
            [
                f"{heading}# Selected model",
                "",
                f"- Target: {summary.get('target') or '—'}",
                f"- Target kind: {summary.get('target_kind') or '—'}",
                f"- Metric: {summary.get('metric') or '—'}",
                f"- Selected family: {summary.get('selected_family') or 'none'}",
                f"- Candidates: {summary.get('candidate_count', '—')}",
                f"- Validation: {summary.get('validation_verdict') or 'not run'}",
                "",
            ]
        )
    if report.code_verified_flags:
        lines.extend([f"{heading}# Code-verified flags", ""])
        lines.extend(f"- {flag}" for flag in report.code_verified_flags)
        lines.append("")
    lines.extend(_markdown_section("Vetoes", report.vetoes))
    lines.extend(_markdown_section("Pending re-audit", report.pending_reaudit))
    lines.extend(_markdown_section("Concerns", report.concerns))
    lines.extend(_markdown_section("Advisories", report.advisories))
    if report.unevaluated_critics or report.missing_evidence:
        lines.extend([f"{heading}# Unevaluated critics and missing evidence", ""])
        for name in report.unevaluated_critics:
            lines.append(f"- Unevaluated: {_pretty(name)}")
        for item in report.missing_evidence:
            lines.append(f"- Missing: {item}")
        lines.append("")
    if report.reaudit_instructions:
        lines.extend([f"{heading}# Next steps", ""])
        for index, item in enumerate(report.reaudit_instructions, start=1):
            lines.append(
                f"{index}. **{_pretty(item.get('from'))}:** {item.get('fix')} "
                f"(test: {item.get('test')})"
            )
        lines.append("")
    if report.note:
        lines.extend([f"_{report.note}_", ""])
    return "\n".join(lines).rstrip() + "\n"


def render_json(report: FinalJudgeReport, *, indent: int = 2) -> str:
    return json.dumps(report.model_dump(mode="json"), indent=indent, ensure_ascii=False) + "\n"


_FORMATS = ("json", "markdown", "html", "fragment")


def write_reports(
    directory: str | Path,
    report: FinalJudgeReport,
    *,
    stem: str = "final_judge",
    formats: Collection[str] | None = None,
) -> dict[str, Path]:
    """Write the requested report formats and return their paths.

    ``formats=None`` writes every format.  ``formats`` accepts any subset of
    ``json``, ``markdown``, ``html``, and ``fragment``.
    """
    selected = set(_FORMATS if formats is None else formats)
    unknown = selected - set(_FORMATS)
    if unknown:
        raise ValueError(
            "unknown report format(s): " + ", ".join(sorted(unknown))
        )
    base = Path(directory)
    base.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": base / f"{stem}.json",
        "markdown": base / f"{stem}.md",
        "html": base / f"{stem}.html",
        "fragment": base / f"{stem}.fragment.html",
    }
    if "json" in selected:
        paths["json"].write_text(render_json(report), encoding="utf-8")
    if "markdown" in selected:
        paths["markdown"].write_text(render_markdown(report), encoding="utf-8")
    if "html" in selected:
        paths["html"].write_text(render_document(report), encoding="utf-8")
    if "fragment" in selected:
        paths["fragment"].write_text(render_fragment(report), encoding="utf-8")
    return {name: path for name, path in paths.items() if name in selected}


__all__ = [
    "render_document",
    "render_fragment",
    "render_json",
    "render_markdown",
    "write_reports",
]
