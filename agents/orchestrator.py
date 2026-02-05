"""Orchestrator agent for patent prosecution assistance."""

import json
import uuid
from pathlib import Path
from typing import AsyncIterator

from claude_agent_sdk import (
    ClaudeAgentOptions,
    query,
    AssistantMessage,
    ResultMessage,
    ToolResultBlock,
)
from claude_agent_sdk.types import StreamEvent

from .prompts import get_orchestrator_prompt


async def run_orchestrator_turn(
    workspace: Path,
    conversation_history: list[dict],
    user_message: str,
) -> AsyncIterator[dict]:
    """Run one turn of the orchestrator agent.

    Yields UI Message Stream Protocol events directly.

    Args:
        workspace: Path to the session workspace directory
        conversation_history: Previous conversation messages
        user_message: The new user message to process

    Yields:
        UI Message Stream Protocol event dicts:
        - {"type": "text-start", "id": "..."}
        - {"type": "text-delta", "id": "...", "delta": "..."}
        - {"type": "text-end", "id": "..."}
        - {"type": "tool-input-start", "toolCallId": "...", "toolName": "..."}
        - {"type": "tool-input-available", "toolCallId": "...", "toolName": "...", "input": {...}}
        - {"type": "tool-output-available", "toolCallId": "...", "output": "..."}
        - {"type": "error", "errorText": "..."}
        - {"type": "finish"}
    """
    # Build the prompt with conversation history
    prompt = _format_prompt(conversation_history, user_message)

    options = ClaudeAgentOptions(
        system_prompt=get_orchestrator_prompt(),
        cwd=str(workspace),
        allowed_tools=["Read", "Write", "Grep", "Glob", "Bash", "Task"],
        permission_mode="acceptEdits",
        max_turns=50,
        include_partial_messages=True,  # Enable token-level streaming
    )

    # Track text part state
    text_part_id = str(uuid.uuid4())
    text_started = False

    # Track current tool call
    current_tool_id: str | None = None
    current_tool_name: str | None = None
    current_tool_input_json = ""  # Accumulate input JSON delta

    try:
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, StreamEvent):
                event = message.event
                event_type = event.get("type")

                if event_type == "content_block_start":
                    content_block = event.get("content_block", {})
                    if content_block.get("type") == "tool_use":
                        # End text part if active before tool call
                        if text_started:
                            yield {"type": "text-end", "id": text_part_id}
                            text_started = False
                            text_part_id = str(uuid.uuid4())

                        # Tool call starting
                        current_tool_id = content_block.get("id") or f"call_{uuid.uuid4().hex[:8]}"
                        current_tool_name = content_block.get("name")
                        current_tool_input_json = ""

                        yield {
                            "type": "tool-input-start",
                            "toolCallId": current_tool_id,
                            "toolName": current_tool_name,
                        }

                elif event_type == "content_block_delta":
                    delta = event.get("delta", {})
                    delta_type = delta.get("type")

                    if delta_type == "text_delta":
                        text = delta.get("text", "")
                        if text:
                            # Start text part on first text chunk
                            if not text_started:
                                yield {"type": "text-start", "id": text_part_id}
                                text_started = True

                            yield {"type": "text-delta", "id": text_part_id, "delta": text}

                    elif delta_type == "input_json_delta":
                        # Accumulate tool input JSON
                        current_tool_input_json += delta.get("partial_json", "")

                elif event_type == "content_block_stop":
                    # Content block finished
                    if current_tool_id:
                        # Parse accumulated input and emit tool-input-available
                        try:
                            tool_input = json.loads(current_tool_input_json) if current_tool_input_json else {}
                        except json.JSONDecodeError:
                            tool_input = {"raw": current_tool_input_json}

                        yield {
                            "type": "tool-input-available",
                            "toolCallId": current_tool_id,
                            "toolName": current_tool_name,
                            "input": tool_input,
                        }
                        # Don't emit output here - wait for ToolResultBlock
                        current_tool_id = None
                        current_tool_name = None
                        current_tool_input_json = ""

            elif isinstance(message, AssistantMessage):
                # AssistantMessage contains complete content including tool results
                for block in message.content:
                    if isinstance(block, ToolResultBlock):
                        content = getattr(block, "content", "")
                        if isinstance(content, list):
                            content = " ".join(
                                str(c.get("text", c)) if isinstance(c, dict) else str(c)
                                for c in content
                            )
                        yield {
                            "type": "tool-output-available",
                            "toolCallId": block.tool_use_id,
                            "output": str(content)[:200],  # Truncate for UI
                        }

            elif isinstance(message, ResultMessage):
                # End text part if still active
                if text_started:
                    yield {"type": "text-end", "id": text_part_id}
                    text_started = False

                yield {"type": "finish"}

    except Exception as e:
        # End text part if active before error
        if text_started:
            yield {"type": "text-end", "id": text_part_id}

        yield {"type": "error", "errorText": f"Agent error: {str(e)}"}
        yield {"type": "finish"}


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
