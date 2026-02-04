"""Prompt loading utilities for agent system prompts."""

from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent


def _load_prompt(name: str) -> str:
    """Load a prompt from a text file.

    Args:
        name: The prompt name (without .txt extension)

    Returns:
        The prompt content as a string
    """
    return (_PROMPTS_DIR / f"{name}.txt").read_text()


def get_orchestrator_prompt() -> str:
    """Get the orchestrator system prompt.

    Returns:
        The orchestrator system prompt
    """
    return _load_prompt("orchestrator")
