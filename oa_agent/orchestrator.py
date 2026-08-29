"""Orchestrator agent for patent prosecution assistance."""

import base64
import json
import logging
import time
import uuid
from pathlib import Path
from typing import AsyncIterator, Any

from claude_agent_sdk import (
    AssistantMessage,
    UserMessage,
    ResultMessage,
    ToolResultBlock,
    ToolUseBlock,
)
from claude_agent_sdk.types import StreamEvent

from .client_manager import AgentClientManager

log = logging.getLogger(__name__)


# Supported image MIME types for Claude vision
IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}


def build_message_content(
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
            # PDF: always send native document block for visual understanding
            content.append({
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": data,
                }
            })
            # If text was extracted, also note the extracted file path
            # so handler subagents can Read/Grep it
            if "extracted_text" in file_info:
                stem = Path(filename).stem
                content.append({
                    "type": "text",
                    "text": (
                        f"Text extracted from {filename} to input/{stem}.extracted.md. "
                        f"Handler subagents should Read/Grep that file rather than re-extracting."
                    ),
                })
            elif "processor_error" in file_info:
                content.append({
                    "type": "text",
                    "text": (
                        f"[Note] Automatic text extraction to .extracted.md failed for {filename}: "
                        f"{file_info['processor_error']} "
                        f"You can still read this PDF visually above — it is NOT unreadable. "
                        f"However, no .extracted.md file exists, so handler subagents cannot Grep it. "
                        f"If you need handlers to search this document, extract the text yourself into a file first."
                    ),
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
        elif "extracted_text" in file_info:
            # Processed documents (e.g. .docx): use extracted text
            content.append({
                "type": "text",
                "text": f"<file name=\"{filename}\">\n{file_info['extracted_text']}\n</file>"
            })
        else:
            # Text/other files: just notify, agent will Read from disk
            content.append({
                "type": "text",
                "text": f"File saved to input/{filename}",
            })

    # Add user's text message at the end
    content.append({"type": "text", "text": user_message})

    return content


async def stream_agent_response(
    client_manager: AgentClientManager,
    session_id: str,
    workspace: Path,
    message_content: str | list[dict],
) -> AsyncIterator[dict]:
    """Send message via persistent client, yield UI stream events.

    Translates SDK Message objects into the UI Message Stream Protocol
    events that the frontend expects (text-start, text-delta, text-end,
    tool-input-start, tool-input-available, tool-output-available, finish).
    """
    sid = session_id[:8]
    log.info("[%s] stream_agent_response: starting", sid)

    # Track text part state
    text_part_id = str(uuid.uuid4())
    text_started = False

    # Track current tool call
    current_tool_id: str | None = None
    current_tool_name: str | None = None
    current_tool_input_json = ""

    # Track announced tool IDs so we only emit outputs for tools the frontend knows about
    announced_tool_ids: set[str] = set()

    # Event flow tracking
    t0 = time.monotonic()
    event_counts: dict[str, int] = {}
    subagent_event_count = 0
    tool_calls_dispatched: list[str] = []

    try:
        async for message in client_manager.send_message(session_id, workspace, message_content):
            # Synthetic events injected by hooks (not SDK Message objects)
            if isinstance(message, dict) and message.get("_synthetic") == "compaction":
                if text_started:
                    yield {"type": "text-end", "id": text_part_id}
                    text_started = False
                    text_part_id = str(uuid.uuid4())
                yield {
                    "type": "compaction",
                    "id": str(uuid.uuid4()),
                    "trigger": message.get("trigger", "auto"),
                }
                continue

            parent_id = getattr(message, "parent_tool_use_id", None)

            # Count events by type
            msg_type = type(message).__name__
            event_counts[msg_type] = event_counts.get(msg_type, 0) + 1
            if parent_id is not None:
                subagent_event_count += 1

            # Handle UserMessage which contains tool results
            if isinstance(message, UserMessage):
                for block in message.content:
                    if isinstance(block, ToolResultBlock):
                        if block.tool_use_id not in announced_tool_ids:
                            continue
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
                # Skip subagent streaming events
                if parent_id is not None:
                    continue

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

                        announced_tool_ids.add(current_tool_id)
                        tool_calls_dispatched.append(current_tool_name or "unknown")
                        log.info("[%s] tool call #%d: %s", sid, len(tool_calls_dispatched), current_tool_name)
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
                            if not text_started:
                                yield {"type": "text-start", "id": text_part_id}
                                text_started = True
                            yield {"type": "text-delta", "id": text_part_id, "delta": text}

                    elif delta_type == "input_json_delta":
                        current_tool_input_json += delta.get("partial_json", "")

                elif event_type == "content_block_stop":
                    if current_tool_id:
                        try:
                            tool_input = json.loads(current_tool_input_json) if current_tool_input_json else {}
                        except json.JSONDecodeError:
                            log.warning("[%s] tool input JSON parse failed for %s (len=%d)", sid, current_tool_name, len(current_tool_input_json))
                            tool_input = {"raw": current_tool_input_json}

                        # Log subagent spawns and file operations with extra detail
                        if current_tool_name == "Task":
                            desc = tool_input.get("description", "")[:80] if isinstance(tool_input, dict) else ""
                            log.info("[%s] Task dispatched: %s", sid, desc)

                        yield {
                            "type": "tool-input-available",
                            "toolCallId": current_tool_id,
                            "toolName": current_tool_name,
                            "input": tool_input,
                        }
                        current_tool_id = None
                        current_tool_name = None
                        current_tool_input_json = ""

            elif isinstance(message, AssistantMessage):
                # Check for error field (rate_limit, server_error, etc.)
                msg_error = getattr(message, "error", None)
                if msg_error:
                    log.warning("[%s] AssistantMessage error: %s", sid, msg_error)

                if parent_id is not None:
                    # Subagent tool calls — announce as regular tool cards
                    for block in message.content:
                        if isinstance(block, ToolUseBlock):
                            if text_started:
                                yield {"type": "text-end", "id": text_part_id}
                                text_started = False
                                text_part_id = str(uuid.uuid4())

                            announced_tool_ids.add(block.id)
                            log.info("[%s] subagent(%s) tool: %s", sid, parent_id[:8], block.name)
                            yield {
                                "type": "tool-input-start",
                                "toolCallId": block.id,
                                "toolName": block.name,
                            }
                            yield {
                                "type": "tool-input-available",
                                "toolCallId": block.id,
                                "toolName": block.name,
                                "input": {**block.input, "_isSubagent": True},
                            }

            elif isinstance(message, ResultMessage):
                duration = time.monotonic() - t0
                is_error = getattr(message, "is_error", False)
                num_turns = getattr(message, "num_turns", None)
                duration_ms = getattr(message, "duration_ms", None)
                total_cost = getattr(message, "total_cost_usd", None)
                usage = getattr(message, "usage", None)
                log.info(
                    "[%s] ResultMessage is_error=%s num_turns=%s duration_ms=%s cost_usd=%s usage=%s events=%s subagent_events=%d tools=%s wall=%.1fs",
                    sid, is_error, num_turns, duration_ms, total_cost, usage,
                    event_counts, subagent_event_count, tool_calls_dispatched, duration,
                )
                if is_error:
                    result_text = getattr(message, "result", "")
                    log.error("[%s] ResultMessage error: %s", sid, str(result_text)[:500])

                if text_started:
                    yield {"type": "text-end", "id": text_part_id}
                    text_started = False

                yield {"type": "finish"}

    except Exception as e:
        if text_started:
            yield {"type": "text-end", "id": text_part_id}

        duration = time.monotonic() - t0
        log.error(
            "[%s] orchestrator error after events=%s subagent_events=%d tools=%s wall=%.1fs",
            sid, event_counts, subagent_event_count, tool_calls_dispatched, duration,
            exc_info=True,
        )
        yield {"type": "error", "errorText": f"Agent error: {str(e)}"}
        yield {"type": "finish"}
