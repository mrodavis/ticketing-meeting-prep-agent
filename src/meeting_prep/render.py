"""Markdown view of a run, for the CLI and for pasting into a CRM note."""

from __future__ import annotations

from typing import Any


def render_markdown(state: dict[str, Any], review: dict[str, Any] | None = None) -> str:
    req, account = state["request"], state.get("account") or {}
    a = state.get("assessment") or {}
    risk = a.get("risk") or {}
    brief = state.get("draft") or {}
    facts = {f["id"]: f["text"] for f in a.get("facts", [])}

    lines = [
        f"# {account.get('name', req['account_id'])}: {req['purpose'].replace('_', ' ')} meeting on {req['meeting_date']}",
        "",
        f"**Risk (code rules):** {risk.get('tier', 'n/a')} (score {risk.get('score', 'n/a')})"
        + ("" if risk.get("complete", True) else "  _incomplete: see data gaps_"),
    ]
    for r in risk.get("reasons", []):
        lines.append(f"- +{r['points']} {r['reason']} `[{r['fact']}]`")
    origin = state.get("brief_origin", "n/a")
    lines += ["", f"**Brief source:** {origin}"]
    if review and review.get("flags"):
        lines.append(f"**Needs attention:** {', '.join(review['flags'])}")
    if not brief:
        return "\n".join(lines) + "\n"

    lines += ["", f"## {brief['headline']}", "", brief["situation"], "", "### Talking points"]
    lines += [f"- {p['text']} `{', '.join(p['evidence'])}`" for p in brief["talking_points"]]
    if brief.get("risks"):
        lines += ["", "### Risks"] + [f"- {p['text']} `{', '.join(p['evidence'])}`" for p in brief["risks"]]
    if brief.get("recommended_offers"):
        lines += ["", "### Offers you can make"]
        lines += [f"- **{o['offer_id']}**: {o['rationale']}" for o in brief["recommended_offers"]]
    if brief.get("questions_to_ask"):
        lines += ["", "### Questions to ask"] + [f"- {q}" for q in brief["questions_to_ask"]]
    if facts:
        lines += ["", "<details><summary>Facts this brief may cite</summary>", ""]
        lines += [f"- `{fid}` {text}" for fid, text in facts.items()]
        lines += ["", "</details>"]
    return "\n".join(lines) + "\n"
