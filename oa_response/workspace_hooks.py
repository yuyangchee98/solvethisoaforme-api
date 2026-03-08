"""PreToolUse hook that enforces workspace path boundaries.

Prevents the agent from reading/writing files outside its session workspace.
"""

import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Tool input fields that contain file/directory paths
_PATH_FIELDS = {
    "Read": "file_path",
    "Write": "file_path",
    "Edit": "file_path",
    "Glob": "path",
    "Grep": "path",
}

# Subdirectory names that indicate a misrooted workspace path
_WORKSPACE_SUBDIRS = {"input", "rejections", "working", "prior_art_working"}


def create_workspace_guard(workspace: Path):
    """Return a PreToolUse hook callback that enforces workspace boundaries.

    - Paths inside workspace: pass through
    - Paths outside but containing a known subdir: rewrite to workspace
    - Paths completely outside: deny with helpful message
    - Relative paths: resolved against workspace before checking
    """
    workspace = workspace.resolve()

    async def _guard(hook_input, _tool_use_id, _ctx):
        tool_name = hook_input.get("tool_name", "")
        field = _PATH_FIELDS.get(tool_name)
        if field is None:
            return {}

        tool_input = hook_input.get("tool_input", {})
        raw_path = tool_input.get(field)
        if raw_path is None:
            # Glob/Grep path is optional (defaults to cwd)
            return {}

        target = Path(raw_path)
        if not target.is_absolute():
            target = workspace / target
        target = target.resolve()

        # Already inside workspace — allow
        try:
            target.relative_to(workspace)
            return {}
        except ValueError:
            pass

        # Try to salvage: look for a known subdir in the path parts
        parts = target.parts
        for i, part in enumerate(parts):
            if part in _WORKSPACE_SUBDIRS:
                # Reconstruct the tail from this subdir onward
                tail = Path(*parts[i:])
                corrected = workspace / tail
                log.warning(
                    "Workspace guard: rewriting %s → %s (tool=%s)",
                    raw_path,
                    corrected,
                    tool_name,
                )
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "allow",
                        "updatedInput": {
                            **tool_input,
                            field: str(corrected),
                        },
                    }
                }

        # Completely outside workspace — deny
        log.warning(
            "Workspace guard: DENIED %s outside workspace %s (tool=%s)",
            raw_path,
            workspace,
            tool_name,
        )
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    f"Path {raw_path} is outside your workspace. "
                    f"Use paths under {workspace} instead."
                ),
            }
        }

    return _guard
