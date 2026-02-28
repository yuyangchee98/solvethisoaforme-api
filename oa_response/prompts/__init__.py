"""Prompt loading utilities for agent system prompts."""

from pathlib import Path

from claude_agent_sdk import AgentDefinition

_PROMPTS_DIR = Path(__file__).parent

# Subagent configuration
_SUBAGENT_DESCRIPTIONS: dict[str, str] = {
    "prior-art": (
        "Analyzes prior art rejections based on novelty (anticipation) and "
        "obviousness grounds. Invoke for any rejection where the examiner "
        "alleges the claims are anticipated by or obvious over cited references. "
        "Receives rejection text, claims, spec summary path, "
        "and prior art file paths. Returns per-claim analysis with "
        "argue/amend recommendation."
    ),
    "general-rejection": (
        "Analyzes non-prior-art rejections. Receives rejection text, claims, "
        "and spec file paths. Returns per-claim analysis with argue/amend "
        "recommendation."
    ),
}

_SUBAGENT_TOOLS: dict[str, list[str]] = {
    "prior-art": ["Read", "Grep", "Glob", "Bash", "Write"],
    "general-rejection": ["Read", "Grep", "Glob", "Bash", "Write"],
}

_SUBAGENT_MODELS: dict[str, str] = {
    "prior-art": "opus",
    "general-rejection": "opus",
}


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


def get_agent_definitions() -> dict[str, AgentDefinition]:
    """Get all subagent definitions for the orchestrator.

    Returns:
        Dict mapping agent names to their AgentDefinition objects.
        Agent names use kebab-case (e.g., "prior-art").
    """
    agents = {}
    for name, description in _SUBAGENT_DESCRIPTIONS.items():
        # Convert kebab-case to snake_case for file names
        file_name = name.replace("-", "_")
        agents[name] = AgentDefinition(
            description=description,
            prompt=_load_prompt(file_name),
            tools=_SUBAGENT_TOOLS.get(name),
            model=_SUBAGENT_MODELS.get(name),
        )
    return agents
