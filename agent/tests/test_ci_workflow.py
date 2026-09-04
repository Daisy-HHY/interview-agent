from pathlib import Path


def test_ci_workflow_runs_fake_llm_without_secret():
    workflow = Path(__file__).parents[2] / ".github" / "workflows" / "ci.yml"
    text = workflow.read_text(encoding="utf-8")

    assert "--fake-llm" in text
    assert "INTERVIEW_API_KEY" not in text
    assert "pytest" in text
    assert "npm test" in text
    assert "npm run compile" in text
