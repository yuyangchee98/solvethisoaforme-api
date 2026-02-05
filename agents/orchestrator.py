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
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
)

from .prompts import get_orchestrator_prompt


async def run_orchestrator_turn(
    workspace: Path,
    conversation_history: list[dict],
    user_message: str,
) -> AsyncIterator[str]:
    """Run one turn of the orchestrator agent.

    Yields Vercel AI SDK Text Stream Protocol formatted strings.

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
    )

    try:
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        # Type 0: text delta
                        yield f"0:{json.dumps(block.text)}\n"
                    elif isinstance(block, ToolUseBlock):
                        # Type 9: tool call start
                        tool_call = {
                            "toolCallId": block.id or f"call_{uuid.uuid4().hex[:8]}",
                            "toolName": block.name,
                            "args": block.input if isinstance(block.input, dict) else {},
                        }
                        yield f"9:{json.dumps(tool_call)}\n"
                    elif isinstance(block, ToolResultBlock):
                        # Type a: tool result
                        content = getattr(block, "content", "")
                        # Handle content that might be a list of content blocks
                        if isinstance(content, list):
                            content = " ".join(
                                str(c.get("text", c)) if isinstance(c, dict) else str(c)
                                for c in content
                            )
                        tool_result = {
                            "toolCallId": block.tool_use_id,
                            "result": str(content),
                        }
                        yield f"a:{json.dumps(tool_result)}\n"

            elif isinstance(message, ResultMessage):
                # Type d: finish
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
