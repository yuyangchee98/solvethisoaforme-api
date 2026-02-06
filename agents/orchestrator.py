"""Orchestrator agent for patent prosecution assistance."""

import base64
import json
import uuid
from pathlib import Path
from typing import AsyncIterator, Any

from claude_agent_sdk import (
    ClaudeAgentOptions,
    query,
    AssistantMessage,
    UserMessage,
    ResultMessage,
    ToolResultBlock,
)
from claude_agent_sdk.types import StreamEvent

from .prompts import get_orchestrator_prompt, get_agent_definitions


# Supported image MIME types for Claude vision
IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}


def _build_message_content(
    user_message: str,
    uploaded_files: list[dict[str, Any]] | None,
) -> str | list[dict[str, Any]]:
    """Build Claude message content with native document/image blocks.

    Args:
        user_message: The user's text message
        uploaded_files: List of uploaded file metadata with:
            - filename: str
            - path: str
            - media_type: str
            - data: str (base64 encoded)

    Returns:
        Either a simple string (no files) or a list of content blocks
    """
    if not uploaded_files:
        return user_message

    content: list[dict[str, Any]] = []

    for file_info in uploaded_files:
        media_type = file_info["media_type"]
        filename = file_info["filename"]
        data = file_info["data"]

        if media_type == "application/pdf":
            # PDF: use native document block for visual understanding
            content.append({
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": data,
                }
            })
        elif media_type in IMAGE_MIME_TYPES:
            # Images: use native image block for vision
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": data,
                }
            })
        else:
            # Text files: include content inline
            try:
                text_content = base64.b64decode(data).decode("utf-8")
                content.append({
                    "type": "text",
                    "text": f"<file name=\"{filename}\">\n{text_content}\n</file>"
                })
            except (UnicodeDecodeError, ValueError):
                # Skip binary files that aren't PDF/image
                pass

    # Add user's text message at the end
    content.append({"type": "text", "text": user_message})

    return content


async def run_orchestrator_turn(
    workspace: Path,
    conversation_history: list[dict],
    user_message: str,
    uploaded_files: list[dict[str, Any]] | None = None,
) -> AsyncIterator[dict]:
    """Run one turn of the orchestrator agent.

    Yields UI Message Stream Protocol events directly.

    Args:
        workspace: Path to the session workspace directory
        conversation_history: Previous conversation messages
        user_message: The new user message to process
        uploaded_files: Optional list of uploaded file metadata for native content blocks

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
    # Build the message content with native file blocks if files are uploaded
    message_content = _build_message_content(user_message, uploaded_files)

    # Build the prompt with conversation history
    prompt = _format_prompt(conversation_history, message_content)

    options = ClaudeAgentOptions(
        system_prompt=get_orchestrator_prompt(),
        cwd=str(workspace),
        allowed_tools=["Read", "Write", "Grep", "Glob", "Bash", "Task"],
        permission_mode="acceptEdits",
        max_turns=50,
        include_partial_messages=True,  # Enable token-level streaming
        agents=get_agent_definitions(),  # Subagent definitions for Task tool
    )

    # Track text part state
    text_part_id = str(uuid.uuid4())
    text_started = False

    # Track current tool call
    current_tool_id: str | None = None
    current_tool_name: str | None = None
    current_tool_input_json = ""  # Accumulate input JSON delta

    # Determine prompt format based on whether we have content blocks
    if isinstance(prompt, list):
        # Use streaming mode for content blocks (files attached)
        async def prompt_stream():
            yield {
                "type": "user",
                "message": {"role": "user", "content": prompt},
                "parent_tool_use_id": None,
                "session_id": str(uuid.uuid4()),
            }
        prompt_input = prompt_stream()
    else:
        # Use simple string mode for text-only messages
        prompt_input = prompt

    try:
        async for message in query(prompt=prompt_input, options=options):

            # Handle UserMessage which contains tool results
            if isinstance(message, UserMessage):
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
                            "output": str(content),
                        }

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
                # AssistantMessage contains ToolUseBlock (handled via StreamEvent)
                pass

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


def _format_prompt(
    conversation_history: list[dict],
    user_message: str | list[dict[str, Any]],
) -> str | list[dict[str, Any]]:
    """Format the conversation history and new message into a prompt.

    Args:
        conversation_history: Previous messages as list of {role, content}
        user_message: The new user message (string or list of content blocks)

    Returns:
        Formatted prompt - string if user_message is string, otherwise
        prepends conversation history to content blocks
    """
    # Build conversation history prefix
    history_parts = []
    if conversation_history:
        history_parts.append("<conversation_history>")
        for msg in conversation_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            history_parts.append(f"<{role}>{content}</{role}>")
        history_parts.append("</conversation_history>")
        history_parts.append("")

    # If user_message is a string, return formatted string
    if isinstance(user_message, str):
        history_parts.append(f"<user_message>{user_message}</user_message>")
        return "\n".join(history_parts)

    # If user_message is a list of content blocks, prepend history as text block
    if history_parts:
        history_text = "\n".join(history_parts)
        # Find the last text block (which contains the actual user message)
        # and prepend the history to it
        result = []
        for i, block in enumerate(user_message):
            if block.get("type") == "text" and i == len(user_message) - 1:
                # This is the last text block - prepend history
                result.append({
                    "type": "text",
                    "text": f"{history_text}\n<user_message>{block['text']}</user_message>"
                })
            else:
                result.append(block)
        return result

    # No history, just wrap user message text in tags
    result = []
    for i, block in enumerate(user_message):
        if block.get("type") == "text" and i == len(user_message) - 1:
            result.append({
                "type": "text",
                "text": f"<user_message>{block['text']}</user_message>"
            })
        else:
            result.append(block)
    return result
