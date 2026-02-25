"""Manages persistent ClaudeSDKClient instances per session.

Each session gets its own long-lived CLI subprocess that maintains
full conversation state, eliminating the need for XML history hacks.

IMPORTANT: The Claude Agent SDK requires all operations on a client
to happen within the same asyncio Task (anyio cancel scope limitation).
Each session therefore gets a dedicated worker Task that owns the client.
HTTP request handlers communicate with the worker via asyncio Queues.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from pathlib import Path

from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
from claude_agent_sdk._errors import CLIConnectionError
from claude_agent_sdk.types import Message, ResultMessage

from .prompts import get_orchestrator_prompt, get_agent_definitions
from .tools import create_patent_tools_server

log = logging.getLogger(__name__)

# Sentinel that signals "turn is complete, no more messages"
_DONE = object()

# Singleton
_client_manager: "AgentClientManager | None" = None


class _SessionWorker:
    """A dedicated asyncio Task that owns a ClaudeSDKClient for one session.

    All SDK operations (connect, query, receive_response, disconnect) run
    inside this single Task, satisfying the anyio same-task requirement.
    HTTP handlers interact via asyncio Queues.
    """

    def __init__(self, session_id: str, options: ClaudeAgentOptions) -> None:
        self._session_id = session_id
        self._sid = session_id[:8]
        self._options = options
        # Input: (content, output_queue) pairs, or None to shutdown
        self._input: asyncio.Queue[tuple[str | list[dict], asyncio.Queue] | None] = (
            asyncio.Queue()
        )
        self._task: asyncio.Task | None = None
        self._ready = asyncio.Event()
        self._connect_error: BaseException | None = None

    async def start(self) -> None:
        """Start the worker Task and wait until the client is connected."""
        self._task = asyncio.create_task(self._run(), name=f"sdk-worker-{self._sid}")
        await self._ready.wait()
        if self._connect_error:
            raise self._connect_error

    # ------------------------------------------------------------------
    # Worker loop (runs in its own Task)
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        client = ClaudeSDKClient(options=self._options)
        try:
            await client.connect()
        except BaseException as e:
            self._connect_error = e
            self._ready.set()
            return

        self._ready.set()
        log.warning("[%s] worker: connected", self._sid)

        try:
            while True:
                item = await self._input.get()
                if item is None:
                    log.warning("[%s] worker: shutdown signal received", self._sid)
                    break

                content, output_q = item
                try:
                    await self._handle_turn(client, content, output_q)
                except CLIConnectionError:
                    log.warning("[%s] worker: CLIConnectionError, exiting", self._sid)
                    break
        finally:
            try:
                await client.disconnect()
            except Exception:
                log.debug("[%s] worker: error during disconnect", self._sid, exc_info=True)
            log.warning("[%s] worker: stopped", self._sid)

    async def _handle_turn(
        self,
        client: ClaudeSDKClient,
        content: str | list[dict],
        output_q: asyncio.Queue,
    ) -> None:
        """Execute one turn: query + stream response back via output_q."""
        try:
            if isinstance(content, str):
                await client.query(content)
            else:
                await client.query(self._make_prompt(content))

            async for msg in client.receive_response():
                await output_q.put(msg)

        except CLIConnectionError:
            # Propagate to caller, then re-raise to exit the worker loop.
            # finally block handles _DONE sentinel.
            await output_q.put(CLIConnectionError("CLI connection lost"))
            raise
        except Exception as e:
            log.error("[%s] worker: turn error: %s", self._sid, e, exc_info=True)
            await output_q.put(e)
        finally:
            await output_q.put(_DONE)

    @staticmethod
    async def _make_prompt(content: list[dict]) -> AsyncIterator[dict]:
        """Wrap list content blocks into the async iterable format query() expects."""
        yield {
            "type": "user",
            "message": {"role": "user", "content": content},
        }

    # ------------------------------------------------------------------
    # Public interface (called from HTTP handler Tasks)
    # ------------------------------------------------------------------

    async def send(self, content: str | list[dict]) -> AsyncIterator[Message]:
        """Send content and yield response Messages.

        This can be called from any asyncio Task — it communicates with the
        worker via Queues so the actual SDK calls stay in the worker Task.
        """
        if not self.alive:
            raise CLIConnectionError("Worker is not running")

        output_q: asyncio.Queue = asyncio.Queue()
        await self._input.put((content, output_q))

        while True:
            try:
                item = await asyncio.wait_for(output_q.get(), timeout=300)
            except asyncio.TimeoutError:
                raise CLIConnectionError("Worker did not respond within timeout")
            if item is _DONE:
                break
            if isinstance(item, BaseException):
                raise item
            yield item

    async def stop(self) -> None:
        """Ask the worker to shut down and wait for it to finish."""
        if self._task is None or self._task.done():
            return
        await self._input.put(None)
        try:
            await asyncio.wait_for(self._task, timeout=15)
        except asyncio.TimeoutError:
            log.warning("[%s] worker: timed out waiting for shutdown, cancelling", self._sid)
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    @property
    def alive(self) -> bool:
        return self._task is not None and not self._task.done()


class AgentClientManager:
    """Manages persistent ClaudeSDKClient instances, one per session.

    Each session's client runs inside a dedicated _SessionWorker Task
    to satisfy the anyio same-Task requirement for cancel scopes.
    """

    def __init__(self) -> None:
        self._workers: dict[str, _SessionWorker] = {}
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

    async def _get_or_create_worker(
        self, session_id: str, workspace: Path
    ) -> _SessionWorker:
        worker = self._workers.get(session_id)
        if worker is not None and worker.alive:
            return worker

        # Clean up dead worker if needed
        if worker is not None:
            log.warning("[%s] worker died, recreating", session_id[:8])
            self._workers.pop(session_id, None)

        options = self._build_options(workspace)
        worker = _SessionWorker(session_id, options)
        await worker.start()
        self._workers[session_id] = worker
        self._last_active[session_id] = time.monotonic()
        log.warning("Created worker for session %s", session_id)
        return worker

    async def send_message(
        self,
        session_id: str,
        workspace: Path,
        content: str | list[dict],
    ) -> AsyncIterator[Message]:
        """Send a message and yield response messages.

        Acquires a per-session lock so concurrent requests are serialized.
        On CLIConnectionError, recreates the worker and retries once.
        """
        lock = self._get_lock(session_id)
        async with lock:
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

    async def _do_send(
        self,
        session_id: str,
        workspace: Path,
        content: str | list[dict],
    ) -> AsyncIterator[Message]:
        """Send content to client and yield messages until ResultMessage."""
        worker = await self._get_or_create_worker(session_id, workspace)
        self._last_active[session_id] = time.monotonic()

        async for msg in worker.send(content):
            yield msg
        self._last_active[session_id] = time.monotonic()

    async def disconnect(self, session_id: str) -> None:
        """Disconnect and remove a session's client."""
        await self._force_disconnect(session_id)

    async def _force_disconnect(self, session_id: str) -> None:
        worker = self._workers.pop(session_id, None)
        self._last_active.pop(session_id, None)
        self._locks.pop(session_id, None)
        if worker is not None:
            try:
                await worker.stop()
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
            # Don't kill sessions that are actively processing a turn
            lock = self._locks.get(sid)
            if lock and lock.locked():
                continue
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
        session_ids = list(self._workers.keys())
        for sid in session_ids:
            await self._force_disconnect(sid)
        self._locks.clear()
        log.warning("AgentClientManager shut down — %d workers cleaned up", len(session_ids))


def get_client_manager() -> AgentClientManager:
    """Return the singleton client manager."""
    global _client_manager
    if _client_manager is None:
        _client_manager = AgentClientManager()
    return _client_manager
