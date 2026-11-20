from meeting_prep import evals


def test_offline_eval_suite_passes():
    report = evals.run(live=False)
    assert report["failures"] == 0, [c for c in report["cases"] if not c.get("pass", True)]
    assert report["aggregate"]["runs"] == len(report["cases"])


def test_check_case_catches_a_wrong_expectation(graph, make_deps):
    from conftest import request

    from meeting_prep.service import start

    result = start(graph, make_deps(), request("ACC-1002", "upsell"), thread_id="t-eval-neg")
    case = {"expect": {"status": "awaiting_review", "risk_tier": "high", "must_cite": ["crm.case.CASE-551"]}}
    fails = evals.check_case(case, result, llm_calls=2)
    assert "risk low != high" in fails and "brief does not cite crm.case.CASE-551" in fails
