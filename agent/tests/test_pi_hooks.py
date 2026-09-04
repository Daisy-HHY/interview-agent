from agent.pi_hooks import assess_hooks_file


def test_hooks_file_is_disabled_when_not_configured(tmp_path):
    decision = assess_hooks_file(str(tmp_path), None)

    assert decision.allowed is False
    assert decision.reason == "not_configured"


def test_hooks_file_rejects_path_outside_workspace(tmp_path):
    decision = assess_hooks_file(str(tmp_path), str(tmp_path.parent / "hooks.py"))

    assert decision.allowed is False
    assert decision.reason == "outside_workspace"


def test_hooks_file_is_never_loaded_in_022(tmp_path):
    hook_file = tmp_path / "hooks.py"
    hook_file.write_text("raise RuntimeError('must not import')", encoding="utf-8")

    decision = assess_hooks_file(str(tmp_path), str(hook_file))

    assert decision.allowed is False
    assert decision.reason == "disabled_by_policy"
