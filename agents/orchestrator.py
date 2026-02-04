"""Orchestrator agent for patent prosecution assistance."""

import json
from pathlib import Path
from typing import AsyncIterator

from claude_agent_sdk import (
    ClaudeAgentOptions,
    query,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
)

from .prompts import get_orchestrator_prompt


async def run_orchestrator_turn(
    workspace: Path,
    conversation_history: list[dict],
    user_message: str,
) -> AsyncIterator[dict]:
    """Run one turn of the orchestrator agent.

    Yields SSE events as the agent processes the request.

    Args:
        workspace: Path to the session workspace directory
        conversation_history: Previous conversation messages
        user_message: The new user message to process

    Yields:
        Event dictionaries for SSE streaming:
        - {"type": "text", "content": "..."} - streaming text chunk
        - {"type": "tool_use", "tool": "...", "input": "..."} - tool invocation
        - {"type": "tool_result", "tool": "..."} - tool completed
        - {"type": "done"} - turn complete
        - {"type": "error", "message": "..."} - error occurred
    """
    # Build the prompt with conversation history
    prompt = _format_prompt(conversation_history, user_message)

    options = ClaudeAgentOptions(
        system_prompt=get_orchestrator_prompt(),
        cwd=str(workspace),
        allowed_tools=["Read", "Write", "Grep", "Glob", "Bash", "Task"],
        permission_mode="acceptEdits",
        max_turns=50,
    )

    try:
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        yield {"type": "text", "content": block.text}
                    elif isinstance(block, ToolUseBlock):
                        yield {
                            "type": "tool_use",
                            "tool": block.name,
                            "input": json.dumps(block.input)
                            if isinstance(block.input, dict)
                            else str(block.input),
                        }
                    elif isinstance(block, ToolResultBlock):
                        yield {"type": "tool_result", "tool": block.tool_use_id}

            elif isinstance(message, ResultMessage):
                yield {"type": "done"}

    except Exception as e:
        yield {"type": "error", "message": f"Agent error: {str(e)}"}


def _format_prompt(conversation_history: list[dict], user_message: str) -> str:
    """Format the conversation history and new message into a prompt.

    Args:
        conversation_history: Previous messages as list of {role, content}
        user_message: The new user message

    Returns:
        Formatted prompt string
    """
    parts = []

    # Include conversation history
    if conversation_history:
        parts.append("<conversation_history>")
        for msg in conversation_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<{role}>{content}</{role}>")
        parts.append("</conversation_history>")
        parts.append("")

    # Add the new user message
    parts.append(f"<user_message>{user_message}</user_message>")

    return "\n".join(parts)
