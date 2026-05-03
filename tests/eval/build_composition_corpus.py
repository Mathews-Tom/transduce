"""Build the transduce-composition v0.1 eval corpus.

Run with ``uv run python tests/eval/build_composition_corpus.py``.
Produces ``tests/eval/transduce_composition_v0_1.jsonl`` with at least
100 records covering three categories:

- ``faithful_chain``: ``stage_1`` and ``stage_2`` are paraphrases of the
  original; the composite verifier should accept.
- ``drift_accumulated``: ``stage_1`` is a faithful paraphrase but
  ``stage_2`` introduces a qualifier, claim, or quantity not present
  in the original — the composite verifier should reject end-to-end
  even though each per-stage check might pass in isolation.
- ``intensity_overshoot``: ``stage_1`` is mild; ``stage_2`` rewrites
  too aggressively, dropping or perturbing entity/number content —
  the composite verifier should reject.

These are illustrative cases the composite verifier should catch
(P3-COMP-02 / N10 in docs/system-design.md). The eval harness in v1.5
will run each record through the real composite verifier under
``@pytest.mark.eval``; this commit ships the structured corpus only.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = Path(__file__).resolve().parent

_COMPOSITION_OUT = EVAL_DIR / "transduce_composition_v0_1.jsonl"

_FaithfulRow = tuple[str, str, str]
_DriftRow = tuple[str, str, str, str]


def _faithful_chain_records() -> list[dict[str, Any]]:
    """Three-tuple (original, stage_1, stage_2) where end-to-end matches.

    Both stages are faithful paraphrases. The composite verifier should
    accept — these are the negative class for drift detection.
    """
    rows: list[_FaithfulRow] = [
        (
            "Acme reported $4.2M in Q3 revenue.",
            "Q3 revenue at Acme reached $4.2M.",
            "$4.2M was the Q3 revenue figure for Acme.",
        ),
        (
            "The migration finished within the maintenance window.",
            "Within the maintenance window the migration completed.",
            "The migration ran to completion in the maintenance window.",
        ),
        (
            "Customers reported faster page loads after the deploy.",
            "After the deploy, customers reported faster page loads.",
            "Page loads were reported faster by customers post-deploy.",
        ),
        (
            "Engineering shipped the feature on Monday.",
            "On Monday, Engineering shipped the feature.",
            "Monday saw Engineering ship the feature.",
        ),
        (
            "The CFO presented the plan to the board.",
            "The plan was presented by the CFO to the board.",
            "The board heard the plan from the CFO.",
        ),
        (
            "Latency dropped from 220ms to 180ms.",
            "Latency went from 220ms down to 180ms.",
            "From 220ms to 180ms the latency dropped.",
        ),
        (
            "Q3 adoption reached 12% of active users.",
            "Active-user adoption hit 12% in Q3.",
            "12% of active users had adopted by Q3.",
        ),
        (
            "The vendor delivered on Friday.",
            "Friday saw the vendor's delivery.",
            "Delivery from the vendor occurred on Friday.",
        ),
        (
            "Sales closed the contract in October.",
            "In October, Sales closed the contract.",
            "October was when Sales closed the contract.",
        ),
        (
            "Onboarding completed within seven business days.",
            "All onboarding finished in seven business days or fewer.",
            "Within seven business days, onboarding was complete.",
        ),
        (
            "The board approved the acquisition.",
            "The acquisition received board approval.",
            "Board approval was given for the acquisition.",
        ),
        (
            "Customers can access the dashboard from mobile.",
            "Mobile users have dashboard access.",
            "From mobile, customers can reach the dashboard.",
        ),
        (
            "Engineering reproduced the bug in staging.",
            "The bug was reproduced by Engineering in staging.",
            "In staging, Engineering reproduced the bug.",
        ),
        (
            "The release fixed three reported issues.",
            "Three reported issues were fixed in the release.",
            "The release contained fixes for three reported issues.",
        ),
        (
            "We met the quarterly target.",
            "Quarterly target met.",
            "The quarterly target was achieved.",
        ),
        (
            "The patch resolves the regression we observed.",
            "Our observed regression is resolved by the patch.",
            "The observed regression was resolved by the patch.",
        ),
        (
            "The audit found three minor findings.",
            "Three minor findings came out of the audit.",
            "Three minor findings were the result of the audit.",
        ),
        (
            "Adoption climbed steadily across the quarter.",
            "Steady adoption growth was observed across the quarter.",
            "Across the quarter, adoption rose steadily.",
        ),
        (
            "The team merged the PR after a 24-hour review.",
            "After a 24-hour review the team merged the PR.",
            "The PR was merged by the team after 24 hours of review.",
        ),
        (
            "Customers retained access during the cutover.",
            "Access was retained by customers during the cutover.",
            "During the cutover, customer access was retained.",
        ),
        (
            "The integration supports webhook callbacks.",
            "Webhook callbacks are supported by the integration.",
            "The integration accepts webhook callbacks.",
        ),
        (
            "The deployment ran during the maintenance window.",
            "During the maintenance window, the deployment ran.",
            "The deployment took place in the maintenance window.",
        ),
        (
            "Compliance approved the new data retention policy yesterday.",
            "Yesterday, Compliance approved the new data retention policy.",
            "The new data retention policy was approved by Compliance yesterday.",
        ),
        (
            "The pilot ran from Q1 2024 to Q4 2024 across four regions.",
            "Across four regions, the pilot ran from Q1 2024 to Q4 2024.",
            "From Q1 2024 to Q4 2024, the pilot operated in four regions.",
        ),
        (
            "Engineering merged the PR after a 24-hour review.",
            "The PR was merged by Engineering after 24 hours of review.",
            "After 24 hours of review, Engineering merged the PR.",
        ),
        (
            "Performance reviews happen in November this cycle.",
            "This cycle's performance reviews are in November.",
            "November is when this cycle's performance reviews occur.",
        ),
        (
            "The conference is scheduled for next quarter in San Francisco.",
            "Next quarter, the conference is scheduled for San Francisco.",
            "San Francisco hosts next quarter's scheduled conference.",
        ),
        (
            "Engineering reproduced the bug in staging on 2025-12-31.",
            "On 2025-12-31, Engineering reproduced the bug in staging.",
            "The 2025-12-31 staging session was when Engineering reproduced the bug.",
        ),
        (
            "Quarterly review is scheduled for fiscal year 2026.",
            "FY2026 has the quarterly review scheduled.",
            "The quarterly review for fiscal year 2026 is scheduled.",
        ),
        (
            "Funding closes by end of fiscal year with two co-investors.",
            "By end of fiscal year, funding closes with two co-investors.",
            "End-of-fiscal-year funding closes alongside two co-investors.",
        ),
        (
            "Adoption is up across enterprise accounts.",
            "Across enterprise accounts, adoption is up.",
            "Enterprise-account adoption is up.",
        ),
        (
            "Reception of the keynote was warm and Q&A ran long.",
            "The keynote got a warm reception with a long Q&A.",
            "A warm reception greeted the keynote and the Q&A ran long.",
        ),
        (
            "The team shipped three features and closed five issues this sprint.",
            "This sprint, the team shipped three features and closed five issues.",
            "Three features and five issue closures came out of the sprint.",
        ),
    ]
    return [
        {
            "category": "faithful_chain",
            "original": original,
            "stage_1": stage_1,
            "stage_2": stage_2,
            "expected_composite_verdict": "accept",
        }
        for original, stage_1, stage_2 in rows
    ]


def _drift_accumulated_records() -> list[dict[str, Any]]:
    """End-to-end drift introduced by stage_2 even when stage_1 is faithful."""
    rows: list[_DriftRow] = [
        (
            "Acme reported $4.2M in Q3 revenue.",
            "Acme's Q3 revenue was $4.2M.",
            "Acme's Q3 revenue was $4.2M, beating analyst expectations by 30%.",
            "added unsupported analyst-comparison qualifier",
        ),
        (
            "The CEO will speak at the conference in October.",
            "October's conference will feature the CEO speaking.",
            "October's conference will feature the CEO as keynote speaker.",
            "promoted speaker to keynote without source",
        ),
        (
            "Engineering shipped the feature on Monday.",
            "On Monday Engineering shipped the feature.",
            "On Monday Engineering shipped the feature despite intense executive pressure.",
            "added unsupported pressure context",
        ),
        (
            "Customers reported slower load times after the deploy.",
            "After the deploy customers reported slower load times.",
            "Most customers reported slower load times after the deploy.",
            "introduced unsupported quantifier 'most'",
        ),
        (
            "The migration finished within the maintenance window.",
            "Within the maintenance window the migration finished.",
            "Within the maintenance window the migration finished with zero data loss.",
            "added unverifiable zero-loss claim",
        ),
        (
            "The board approved the acquisition.",
            "The acquisition was approved by the board.",
            "The acquisition was unanimously approved by the board.",
            "added unsupported unanimity",
        ),
        (
            "Latency dropped from 220ms to 180ms.",
            "Latency went from 220ms to 180ms.",
            "Latency went from 220ms to 180ms, a major improvement.",
            "added editorial qualifier 'major improvement'",
        ),
        (
            "Q3 adoption reached 12% of active users.",
            "12% of active users had adopted by Q3.",
            "An impressive 12% of active users had adopted by Q3.",
            "added editorial qualifier 'impressive'",
        ),
        (
            "The vendor delivered on Friday.",
            "Friday saw the vendor deliver.",
            "Friday saw the vendor deliver after a long delay.",
            "added unsupported delay context",
        ),
        (
            "Sales closed the contract in October.",
            "In October Sales closed the contract.",
            "In October Sales closed the lucrative contract.",
            "added unsupported value qualifier 'lucrative'",
        ),
        (
            "The CFO presented the plan to the board.",
            "The CFO presented the plan to the board.",
            "The CFO presented the controversial plan to the board.",
            "added unsupported descriptor 'controversial'",
        ),
        (
            "Onboarding completed within seven business days.",
            "Onboarding finished in seven business days.",
            "Onboarding finished in seven business days, faster than ever before.",
            "added unsupported historical comparison",
        ),
        (
            "The audit found three minor findings.",
            "Three minor findings came out of the audit.",
            "Three minor findings of no material concern came out of the audit.",
            "added unsupported risk-assessment qualifier",
        ),
        (
            "Customers can access the dashboard from mobile.",
            "Mobile users have dashboard access.",
            "Mobile users have dashboard access from any device.",
            "expanded mobile to 'any device'",
        ),
        (
            "Engineering reproduced the bug in staging.",
            "The bug was reproduced by Engineering in staging.",
            "Engineering quickly reproduced the bug in staging.",
            "added unsupported speed qualifier 'quickly'",
        ),
        (
            "The release fixed three reported issues.",
            "Three reported issues were fixed in the release.",
            "All three reported issues were fixed in the release.",
            "added unsupported completeness 'all'",
        ),
        (
            "We met the quarterly target.",
            "Quarterly target met.",
            "We exceeded the quarterly target.",
            "shifted met to exceeded",
        ),
        (
            "The patch resolves the regression we observed.",
            "The patch resolves the regression.",
            "The patch fully resolves the regression.",
            "added unsupported completeness 'fully'",
        ),
        (
            "Adoption is up across enterprise accounts.",
            "Across enterprise accounts adoption is up.",
            "Adoption is way up across enterprise accounts.",
            "added unsupported magnitude 'way'",
        ),
        (
            "The integration supports webhook callbacks.",
            "Webhook callbacks are supported by the integration.",
            "The integration natively supports webhook callbacks.",
            "added unsupported qualifier 'natively'",
        ),
        (
            "Customers retained access during the cutover.",
            "Access was retained by customers during the cutover.",
            "Customers retained full access during the cutover.",
            "added unsupported qualifier 'full'",
        ),
        (
            "The new data retention policy was approved yesterday.",
            "Yesterday the new data retention policy was approved.",
            "Yesterday the new data retention policy was unanimously approved.",
            "added unsupported unanimity",
        ),
        (
            "Engineering shipped four features.",
            "Four features were shipped by Engineering.",
            "Engineering shipped four features ahead of schedule.",
            "added schedule comparison",
        ),
        (
            "The deal closed at the expected price.",
            "The deal closed at the expected price.",
            "The deal closed exactly at the expected price.",
            "added unsupported qualifier 'exactly'",
        ),
        (
            "Customers signed up at record rates.",
            "Record-rate sign-ups happened among customers.",
            "Record-rate sign-ups happened among new customers.",
            "narrowed customers to 'new customers'",
        ),
        (
            "The deployment succeeded yesterday.",
            "Yesterday's deployment was successful.",
            "Yesterday's deployment was a complete success.",
            "added unsupported qualifier 'complete'",
        ),
        (
            "Latency improved after the rebuild.",
            "After the rebuild latency was better.",
            "After the rebuild latency was significantly better.",
            "added unsupported qualifier 'significantly'",
        ),
        (
            "The CFO presented the plan.",
            "The plan was presented by the CFO.",
            "The plan was presented in detail by the CFO.",
            "added unsupported qualifier 'in detail'",
        ),
        (
            "Audit findings closed on time.",
            "All audit findings were closed on time.",
            "All audit findings were closed well before the deadline.",
            "tightened on-time to well-before",
        ),
        (
            "The team merged the PR last night.",
            "Last night the PR was merged.",
            "Last night the long-awaited PR was merged.",
            "added unsupported qualifier 'long-awaited'",
        ),
        (
            "Adoption rose during Q3.",
            "During Q3 adoption rose.",
            "During Q3 adoption rose substantially.",
            "added unsupported magnitude 'substantially'",
        ),
        (
            "The pilot ran from Q1 2024 to Q4 2024 across four regions.",
            "Across four regions the pilot ran for all of 2024.",
            "Across four regions the pilot ran successfully for all of 2024.",
            "added unsupported success qualifier",
        ),
        (
            "Funding closes by end of fiscal year with two co-investors.",
            "Funding closes with two co-investors by end of fiscal year.",
            "Funding closes with two strategic co-investors by end of fiscal year.",
            "added unsupported descriptor 'strategic'",
        ),
        (
            "The conference is scheduled for next quarter in San Francisco.",
            "Next quarter's conference is scheduled for San Francisco.",
            "Next quarter's flagship conference is scheduled for San Francisco.",
            "added unsupported descriptor 'flagship'",
        ),
        (
            "Engineering merged the PR after a 24-hour review.",
            "After 24 hours of review Engineering merged the PR.",
            "After 24 hours of meticulous review Engineering merged the PR.",
            "added unsupported descriptor 'meticulous'",
        ),
        (
            "Performance reviews happen in November this cycle.",
            "This cycle's performance reviews are in November.",
            "This cycle's annual performance reviews are in November.",
            "added unsupported descriptor 'annual'",
        ),
    ]
    return [
        {
            "category": "drift_accumulated",
            "original": original,
            "stage_1": stage_1,
            "stage_2": stage_2,
            "expected_composite_verdict": "reject",
            "drift_reason": reason,
        }
        for original, stage_1, stage_2, reason in rows
    ]


def _intensity_overshoot_records() -> list[dict[str, Any]]:
    """Stage_2 intensity drives end-to-end output too far from the original."""
    rows: list[_DriftRow] = [
        (
            "Acme reported $4.2M in Q3 revenue.",
            "Acme's Q3 revenue was $4.2M.",
            "Quarterly revenue performance.",
            "stage_2 stripped entity, number, and time period",
        ),
        (
            "The CEO will speak at the conference in October.",
            "October's conference will feature the CEO speaking.",
            "Conference talk.",
            "stage_2 dropped speaker identity and date",
        ),
        (
            "Engineering shipped the feature on Monday.",
            "On Monday Engineering shipped the feature.",
            "Feature shipped.",
            "stage_2 dropped agent and day",
        ),
        (
            "Customers reported slower load times after the deploy.",
            "After the deploy customers reported slower load times.",
            "Performance feedback received.",
            "stage_2 dropped specific direction and trigger",
        ),
        (
            "The migration finished within the maintenance window.",
            "Within the maintenance window the migration finished.",
            "Migration completed.",
            "stage_2 dropped maintenance-window qualifier",
        ),
        (
            "The board approved the acquisition.",
            "The acquisition was approved by the board.",
            "Approval granted.",
            "stage_2 dropped agent and object",
        ),
        (
            "Latency dropped from 220ms to 180ms.",
            "Latency went from 220ms to 180ms.",
            "Latency improved.",
            "stage_2 dropped before/after numbers",
        ),
        (
            "Q3 adoption reached 12% of active users.",
            "12% of active users had adopted by Q3.",
            "Adoption rose.",
            "stage_2 dropped percentage and time",
        ),
        (
            "The vendor delivered on Friday.",
            "Friday saw the vendor deliver.",
            "Delivery happened.",
            "stage_2 dropped vendor and day",
        ),
        (
            "Sales closed the contract in October.",
            "In October Sales closed the contract.",
            "Contract closed.",
            "stage_2 dropped agent and month",
        ),
        (
            "The CFO presented the plan to the board.",
            "The plan was presented by the CFO to the board.",
            "Plan presented.",
            "stage_2 dropped both agents",
        ),
        (
            "Onboarding completed within seven business days.",
            "Onboarding finished in seven business days.",
            "Onboarding done.",
            "stage_2 dropped duration",
        ),
        (
            "The audit found three minor findings.",
            "Three minor findings came out of the audit.",
            "Audit findings.",
            "stage_2 dropped count and severity",
        ),
        (
            "Customers can access the dashboard from mobile.",
            "Mobile users have dashboard access.",
            "Dashboard access available.",
            "stage_2 dropped surface qualifier",
        ),
        (
            "Engineering reproduced the bug in staging.",
            "The bug was reproduced by Engineering in staging.",
            "Bug reproduced.",
            "stage_2 dropped agent and environment",
        ),
        (
            "The release fixed three reported issues.",
            "Three reported issues were fixed in the release.",
            "Issues fixed.",
            "stage_2 dropped count",
        ),
        (
            "We met the quarterly target.",
            "Quarterly target met.",
            "Target hit.",
            "stage_2 dropped time period",
        ),
        (
            "The patch resolves the regression we observed.",
            "Our observed regression is resolved by the patch.",
            "Regression resolved.",
            "stage_2 dropped owner",
        ),
        (
            "Adoption is up across enterprise accounts.",
            "Across enterprise accounts adoption is up.",
            "Adoption rising.",
            "stage_2 dropped customer segment",
        ),
        (
            "The integration supports webhook callbacks.",
            "Webhook callbacks are supported by the integration.",
            "Webhooks supported.",
            "stage_2 dropped subject",
        ),
        (
            "Customers retained access during the cutover.",
            "Access was retained by customers during the cutover.",
            "Access retained.",
            "stage_2 dropped subject and event",
        ),
        (
            "The deployment succeeded yesterday.",
            "Yesterday's deployment was successful.",
            "Deployment ok.",
            "stage_2 dropped time and agent",
        ),
        (
            "Latency improved after the rebuild.",
            "After the rebuild latency was better.",
            "Latency ok.",
            "stage_2 dropped trigger and direction",
        ),
        (
            "The CFO presented the plan.",
            "The plan was presented by the CFO.",
            "Presentation given.",
            "stage_2 dropped both agents",
        ),
        (
            "Audit findings closed on time.",
            "All audit findings were closed on time.",
            "Findings closed.",
            "stage_2 dropped completeness and timing",
        ),
        (
            "The team merged the PR last night.",
            "Last night the PR was merged.",
            "PR merged.",
            "stage_2 dropped time",
        ),
        (
            "Adoption rose during Q3.",
            "During Q3 adoption rose.",
            "Adoption changed.",
            "stage_2 dropped direction and time",
        ),
        (
            "Funding closes by end of fiscal year with two co-investors.",
            "Funding closes with two co-investors by end of fiscal year.",
            "Funding closes.",
            "stage_2 dropped investor count and timing",
        ),
        (
            "The conference is scheduled for next quarter in San Francisco.",
            "Next quarter's conference is scheduled for San Francisco.",
            "Conference scheduled.",
            "stage_2 dropped time and place",
        ),
        (
            "Engineering merged the PR after a 24-hour review.",
            "After 24 hours of review Engineering merged the PR.",
            "PR merged.",
            "stage_2 dropped review duration",
        ),
        (
            "Performance reviews happen in November this cycle.",
            "This cycle's performance reviews are in November.",
            "Reviews scheduled.",
            "stage_2 dropped time and category",
        ),
        (
            "The pilot ran from Q1 2024 to Q4 2024 across four regions.",
            "Across four regions the pilot ran for all of 2024.",
            "Pilot ran.",
            "stage_2 dropped duration and region count",
        ),
        (
            "Quarterly review is scheduled for fiscal year 2026.",
            "FY2026 has the quarterly review scheduled.",
            "Review scheduled.",
            "stage_2 dropped fiscal year",
        ),
        (
            "We expect a decision within two weeks on the funding round.",
            "A decision on the funding round is expected within two weeks.",
            "Decision pending.",
            "stage_2 dropped subject and timeframe",
        ),
        (
            "The pilot ran successfully across four regions in 2024.",
            "Across four regions in 2024 the pilot ran successfully.",
            "Pilot ran.",
            "stage_2 dropped year, count, and outcome",
        ),
    ]
    return [
        {
            "category": "intensity_overshoot",
            "original": original,
            "stage_1": stage_1,
            "stage_2": stage_2,
            "expected_composite_verdict": "reject",
            "drift_reason": reason,
        }
        for original, stage_1, stage_2, reason in rows
    ]


def build_composition_corpus() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    records.extend(_faithful_chain_records())
    records.extend(_drift_accumulated_records())
    records.extend(_intensity_overshoot_records())
    return records


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    payload = "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
    path.write_text(payload + "\n", encoding="utf-8")
    return len(payload.splitlines())


def main() -> None:
    records = build_composition_corpus()
    n = _write_jsonl(_COMPOSITION_OUT, records)
    print(f"wrote {n} composition records to {_COMPOSITION_OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
