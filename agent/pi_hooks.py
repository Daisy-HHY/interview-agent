"""Pi hook 文件的安全评估。

0.2.2 只评估显式路径，不导入或执行外部 Python 文件。
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HooksFileDecision:
    """hook 文件评估结果，不携带文件内容。"""

    allowed: bool
    reason: str
    path: str | None = None


def assess_hooks_file(workspace: str, configured_path: str | None) -> HooksFileDecision:
    """校验 hook 路径并按 0.2.2 默认策略拒绝加载。"""
    if not configured_path:
        return HooksFileDecision(False, "not_configured")

    root = Path(workspace).resolve()
    requested = Path(configured_path)
    candidate = (root / requested if not requested.is_absolute() else requested).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return HooksFileDecision(False, "outside_workspace", str(candidate))

    current = root
    for part in candidate.relative_to(root).parts:
        current /= part
        if current.is_symlink():
            return HooksFileDecision(False, "symlink_path", str(candidate))

    if candidate.suffix.lower() != ".py":
        return HooksFileDecision(False, "invalid_extension", str(candidate))
    if candidate.exists() and candidate.is_dir():
        return HooksFileDecision(False, "directory", str(candidate))
    if not candidate.exists():
        return HooksFileDecision(False, "missing", str(candidate))

    # 0.2.2 安全决策：路径通过校验也不自动执行任意 Python 文件。
    return HooksFileDecision(False, "disabled_by_policy", str(candidate))
