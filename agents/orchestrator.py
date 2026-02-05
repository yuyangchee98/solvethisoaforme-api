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
) -> AsyncIterator[str]:
    """Run one turn of the orchestrator agent.

    Yields Vercel AI SDK Text Stream Protocol formatted strings with real-time streaming.

    Args:
        workspace: Path to the session workspace directory
        conversation_history: Previous conversation messages
        user_message: The new user message to process

    Yields:
        Vercel AI SDK Text Stream Protocol formatted strings:
        - '0:"text"\n' - text delta (type 0)
        - '9:{"toolCallId":"x","toolName":"Read","args":{}}\n' - tool call start
        - 'a:{"toolCallId":"x","result":"..."}\n' - tool result
        - 'd:{"finishReason":"stop"}\n' - finish
        - 'e:{"message":"..."}\n' - error
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

    # Track current tool call for accumulating input
    current_tool_id: str | None = None
    current_tool_name: str | None = None

    try:
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, StreamEvent):
                event = message.event
                event_type = event.get("type")

                if event_type == "content_block_start":
                    content_block = event.get("content_block", {})
                    if content_block.get("type") == "tool_use":
                        # Tool call starting
                        current_tool_id = content_block.get("id") or f"call_{uuid.uuid4().hex[:8]}"
                        current_tool_name = content_block.get("name")
                        tool_call = {
                            "toolCallId": current_tool_id,
                            "toolName": current_tool_name,
                            "args": {},
                        }
                        yield f"9:{json.dumps(tool_call)}\n"

                elif event_type == "content_block_delta":
                    delta = event.get("delta", {})
                    delta_type = delta.get("type")

                    if delta_type == "text_delta":
                        # Stream text chunk
                        text = delta.get("text", "")
                        if text:
                            yield f"0:{json.dumps(text)}\n"

                elif event_type == "content_block_stop":
                    # Content block finished - reset tracking
                    if current_tool_id:
                        current_tool_id = None
                        current_tool_name = None

            elif isinstance(message, AssistantMessage):
                # AssistantMessage contains complete content including tool results
                for block in message.content:
                    if isinstance(block, ToolResultBlock):
                        # Emit tool result
                        content = getattr(block, "content", "")
                        if isinstance(content, list):
                            content = " ".join(
                                str(c.get("text", c)) if isinstance(c, dict) else str(c)
                                for c in content
                            )
                        tool_result = {
                            "toolCallId": block.tool_use_id,
                            "result": str(content)[:200],  # Truncate for UI
                        }
                        yield f"a:{json.dumps(tool_result)}\n"

            elif isinstance(message, ResultMessage):
                # Agent finished all work
                yield 'd:{"finishReason":"stop"}\n'

    except Exception as e:
        # Type e: error
        error_data = {"message": f"Agent error: {str(e)}"}
        yield f"e:{json.dumps(error_data)}\n"


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
