"""Shared short-task extraction helpers for experience retrieval."""

from typing import Optional


def extract_short_task_for_retrieval(text: str, _env_name: Optional[str] = None) -> str:
    """
    Extract a compact task string from raw env observations or rendered prompts.

    Retrieval should compare task experiences against the task itself, not against a
    full prompt that also contains history, admissible actions, or injected
    memories.
    """
    if not text or not text.strip():
        return (text or "").strip()
    value = text.strip()

    goal_marker = "Your goal is to complete the following task:"
    idx_goal = value.find(goal_marker)
    if idx_goal != -1:
        start = idx_goal + len(goal_marker)
        for marker in ("\n\n====================", "\n\n##", "\n\n\n", "\n\n"):
            end = value.find(marker, start)
            if end != -1:
                return value[start:end].strip()
        return value[start:].strip()

    task_marker = "Your task is to:"
    idx_task = value.find(task_marker)
    if idx_task != -1:
        start = idx_task + len(task_marker)
        for marker in ("\n\n##", "\n\n"):
            end = value.find(marker, start)
            if end != -1:
                return value[start:end].strip()
        return value[start:].strip()

    if " [SEP] " in value and "Instruction:" in value:
        parts = value.split(" [SEP] ")
        for i, part in enumerate(parts):
            if part.strip() == "Instruction:" and i + 1 < len(parts):
                return parts[i + 1].strip()

    return value[:500]
