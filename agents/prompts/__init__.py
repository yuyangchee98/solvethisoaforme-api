"""Prompt loading utilities for agent system prompts."""

from pathlib import Path

from claude_agent_sdk import AgentDefinition

_PROMPTS_DIR = Path(__file__).parent

# Subagent configuration
_SUBAGENT_DESCRIPTIONS: dict[str, str] = {
    "prior-art-us": (
        "Analyzes US prior art rejections under 35 USC §102 (anticipation) and "
        "§103 (obviousness). Invoke for any §102 or §103 rejection from a US "
        "patent office action. Receives rejection text, claims, spec summary path, "
        "and prior art file paths. Returns per-claim analysis with "
        "argue/amend recommendation."
    ),
    # Future: "eligibility-us", "clarity-us"
}

_SUBAGENT_TOOLS: dict[str, list[str]] = {
    "prior-art-us": ["Read", "Grep", "Glob", "Bash", "Write"],
}

_SUBAGENT_MODELS: dict[str, str] = {
    "prior-art-us": "opus",
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
        Agent names use kebab-case (e.g., "prior-art-us").
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
