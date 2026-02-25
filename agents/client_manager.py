"""Manages persistent ClaudeSDKClient instances per session.

Each session gets its own long-lived CLI subprocess that maintains
full conversation state, eliminating the need for XML history hacks.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import AsyncIterator

from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
from claude_agent_sdk._errors import CLIConnectionError
from claude_agent_sdk.types import Message, ResultMessage

from .prompts import get_orchestrator_prompt, get_agent_definitions
from .tools import create_patent_tools_server

log = logging.getLogger(__name__)

# Singleton
_client_manager: "AgentClientManager | None" = None


class _SessionState:
    """Holds a client and its long-lived message iterator for one session."""
    __slots__ = ("client", "messages")

    def __init__(self, client: ClaudeSDKClient, messages: AsyncIterator[Message]) -> None:
        self.client = client
        self.messages = messages


class AgentClientManager:
    """Manages persistent ClaudeSDKClient instances, one per session."""

    def __init__(self) -> None:
        self._sessions: dict[str, _SessionState] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._last_active: dict[str, float] = {}

    def _get_lock(self, session_id: str) -> asyncio.Lock:
        if session_id not in self._locks:
            self._locks[session_id] = asyncio.Lock()
        return self._locks[session_id]

    def _build_options(self, workspace: Path) -> ClaudeAgentOptions:
        patent_server = create_patent_tools_server(workspace)

        def _on_stderr(line: str) -> None:
            log.error("CLI stderr: %s", line.rstrip())

        return ClaudeAgentOptions(
            system_prompt=get_orchestrator_prompt(),
            cwd=str(workspace),
            allowed_tools=[
                "Read", "Write", "Grep", "Glob", "Bash", "Task",
                "mcp__patent-tools__FetchPatent",
            ],
            permission_mode="acceptEdits",
            max_turns=50,
            include_partial_messages=True,
            agents=get_agent_definitions(),
            mcp_servers={"patent-tools": patent_server},
            stderr=_on_stderr,
            max_buffer_size=50 * 1024 * 1024,
        )

    async def _create_session(
        self, session_id: str, workspace: Path
    ) -> _SessionState:
        """Create a new client + long-lived message iterator."""
        options = self._build_options(workspace)
        client = ClaudeSDKClient(options=options)
        await client.connect()
        # Create ONE message iterator that lives for the entire session.
        # This is critical: anyio MemoryObjectReceiveStream closes when
        # an async-for iterator over it is garbage-collected. Using
        # receive_response() per turn would create/destroy iterators and
        # kill the stream on the second turn.
        messages = client.receive_messages()
        state = _SessionState(client, messages)
        self._sessions[session_id] = state
        self._last_active[session_id] = time.monotonic()
        log.warning("Created new ClaudeSDKClient for session %s", session_id)
        return state

    async def _get_or_create(
        self, session_id: str, workspace: Path
    ) -> _SessionState:
        state = self._sessions.get(session_id)
        if state is not None:
            return state
        return await self._create_session(session_id, workspace)

    async def send_message(
        self,
        session_id: str,
        workspace: Path,
        content: str | list[dict],
    ) -> AsyncIterator[Message]:
        """Send a message and yield response messages.

        Acquires a per-session lock so concurrent requests are serialized.
        On CLIConnectionError, reconnects once and retries.
        """
        lock = self._get_lock(session_id)
        log.warning("[%s] send_message: acquiring lock", session_id[:8])
        async with lock:
            log.warning("[%s] send_message: lock acquired", session_id[:8])
            try:
                async for msg in self._do_send(session_id, workspace, content):
                    yield msg
            except CLIConnectionError:
                log.warning(
                    "CLIConnectionError for session %s — reconnecting",
                    session_id,
                )
                await self._force_disconnect(session_id)
                try:
                    async for msg in self._do_send(session_id, workspace, content):
                        yield msg
                except CLIConnectionError:
                    log.error(
                        "Retry also failed for session %s", session_id
                    )
                    await self._force_disconnect(session_id)
                    raise

    @staticmethod
    async def _wrap_as_prompt(content: str | list[dict]) -> AsyncIterator[dict]:
        """Wrap content blocks into the async iterable format client.query() expects."""
        yield {
            "type": "user",
            "message": {"role": "user", "content": content},
        }

    async def _do_send(
        self,
        session_id: str,
        workspace: Path,
        content: str | list[dict],
    ) -> AsyncIterator[Message]:
        """Send content to client and yield messages until ResultMessage."""
        state = await self._get_or_create(session_id, workspace)
        content_type = "str" if isinstance(content, str) else f"list[{len(content)} blocks]"
        log.warning("[%s] _do_send: calling client.query() with %s", session_id[:8], content_type)

        if isinstance(content, str):
            await state.client.query(content)
        else:
            await state.client.query(self._wrap_as_prompt(content))

        log.warning("[%s] _do_send: client.query() returned, reading from persistent iterator", session_id[:8])
        self._last_active[session_id] = time.monotonic()

        # Read from the session's long-lived iterator until ResultMessage.
        # This is equivalent to receive_response() but reuses the same
        # iterator across turns instead of creating a new one each time.
        msg_count = 0
        async for msg in state.messages:
            msg_count += 1
            log.warning("[%s] _do_send: msg #%d type=%s", session_id[:8], msg_count, type(msg).__name__)
            yield msg
            if isinstance(msg, ResultMessage):
                break

        log.warning("[%s] _do_send: turn complete after %d messages", session_id[:8], msg_count)
        self._last_active[session_id] = time.monotonic()

    async def disconnect(self, session_id: str) -> None:
        """Disconnect and remove a session's client."""
        await self._force_disconnect(session_id)

    async def _force_disconnect(self, session_id: str) -> None:
        state = self._sessions.pop(session_id, None)
        self._last_active.pop(session_id, None)
        if state is not None:
            try:
                await state.client.disconnect()
            except Exception:
                log.debug("Error disconnecting session %s", session_id, exc_info=True)

    async def cleanup_idle(self, max_idle_seconds: float = 300) -> None:
        """Disconnect clients that have been idle too long."""
        now = time.monotonic()
        idle_sessions = [
            sid
            for sid, last in self._last_active.items()
            if now - last > max_idle_seconds
        ]
        for sid in idle_sessions:
            log.warning("Disconnecting idle session %s", sid)
            await self._force_disconnect(sid)

    async def run_cleanup_loop(self, interval: float = 60) -> None:
        """Background loop that periodically cleans up idle clients."""
        while True:
            await asyncio.sleep(interval)
            try:
                await self.cleanup_idle()
            except Exception:
                log.warning("Cleanup loop error", exc_info=True)

    async def shutdown(self) -> None:
        """Disconnect all clients. Call on app shutdown."""
        session_ids = list(self._sessions.keys())
        for sid in session_ids:
            await self._force_disconnect(sid)
        self._locks.clear()
        log.warning("AgentClientManager shut down — %d clients cleaned up", len(session_ids))


def get_client_manager() -> AgentClientManager:
    """Return the singleton client manager."""
    global _client_manager
    if _client_manager is None:
        _client_manager = AgentClientManager()
    return _client_manager
