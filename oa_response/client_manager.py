"""Manages persistent ClaudeSDKClient instances per session.

Each session gets its own long-lived CLI subprocess that maintains
full conversation state, eliminating the need for XML history hacks.

IMPORTANT: The Claude Agent SDK requires all operations on a client
to happen within the same asyncio Task (anyio cancel scope limitation).
Each session therefore gets a dedicated worker Task that owns the client.
HTTP request handlers communicate with the worker via asyncio Queues.
"""

import asyncio
import dataclasses
import logging
import time
from collections.abc import AsyncIterator
from pathlib import Path

from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, HookMatcher
from claude_agent_sdk._errors import CLIConnectionError
from claude_agent_sdk.types import Message, ResultMessage

from .prompts import get_orchestrator_prompt, get_agent_definitions
from .tools import create_patent_tools_server
from .workspace_hooks import create_workspace_guard

log = logging.getLogger(__name__)

# Sentinel that signals "turn is complete, no more messages"
_DONE = object()


def _describe_message(msg: object, worker: "_SessionWorker | None" = None) -> str:
    """Return a short diagnostic string describing an SDK message.

    Avoids logging payload data — only type names, tool names, and IDs.
    If worker is provided, tracks the current tool name being streamed.
    """
    from claude_agent_sdk import AssistantMessage, UserMessage, ResultMessage
    from claude_agent_sdk.types import StreamEvent

    parent = getattr(msg, "parent_tool_use_id", None)
    prefix = f"subagent({parent[:8]})" if parent else "root"

    if isinstance(msg, StreamEvent):
        evt = msg.event
        etype = evt.get("type", "?")
        if etype == "content_block_start":
            cb = evt.get("content_block", {})
            tool_name = cb.get("name", "")
            if tool_name and worker is not None:
                worker._current_streaming_tool = tool_name
            return f"{prefix}/StreamEvent/{etype}/{cb.get('type','?')}:{tool_name}"
        if etype == "content_block_delta":
            dt = evt.get("delta", {}).get("type", "?")
            tool_ctx = ""
            if dt == "input_json_delta" and worker is not None and worker._current_streaming_tool:
                tool_ctx = f"({worker._current_streaming_tool})"
            return f"{prefix}/StreamEvent/{etype}/{dt}{tool_ctx}"
        if etype == "content_block_stop" and worker is not None:
            worker._current_streaming_tool = ""
        return f"{prefix}/StreamEvent/{etype}"
    if isinstance(msg, AssistantMessage):
        tool_names = [
            b.name for b in getattr(msg, "content", [])
            if hasattr(b, "name")
        ]
        return f"{prefix}/AssistantMessage tools={tool_names}"
    if isinstance(msg, UserMessage):
        return f"{prefix}/UserMessage"
    if isinstance(msg, ResultMessage):
        return f"{prefix}/ResultMessage is_error={getattr(msg, 'is_error', '?')}"
    return f"{prefix}/{type(msg).__name__}"

# Singleton
_client_manager: "AgentClientManager | None" = None


class _SessionWorker:
    """A dedicated asyncio Task that owns a ClaudeSDKClient for one session.

    All SDK operations (connect, query, receive_response, disconnect) run
    inside this single Task, satisfying the anyio same-task requirement.
    HTTP handlers interact via asyncio Queues.
    """

    def __init__(self, session_id: str, options: ClaudeAgentOptions, workspace: Path) -> None:
        self._session_id = session_id
        self._sid = session_id[:8]
        self._options = options
        self._workspace = workspace
        # Input: (content, output_queue) pairs, or None to shutdown
        self._input: asyncio.Queue[tuple[str | list[dict], asyncio.Queue] | None] = (
            asyncio.Queue()
        )
        self._task: asyncio.Task | None = None
        self._ready = asyncio.Event()
        self._connect_error: BaseException | None = None
        # Set by _handle_turn so the PreCompact hook can inject events
        self._current_out_q: asyncio.Queue | None = None
        # Diagnostic: tracks last event seen by _handle_turn (readable from any task)
        self._last_event_info: str = "idle"
        self._last_event_time: float = 0.0
        self._turn_event_count: int = 0
        self._current_streaming_tool: str = ""  # tool name being streamed
        # Reference to the SDK client for subprocess diagnostics
        self._client: ClaudeSDKClient | None = None

    async def start(self) -> None:
        """Start the worker Task and wait until the client is connected."""
        self._task = asyncio.create_task(self._run(), name=f"sdk-worker-{self._sid}")
        await self._ready.wait()
        if self._connect_error:
            raise self._connect_error

    # ------------------------------------------------------------------
    # Worker loop (runs in its own Task)
    # ------------------------------------------------------------------

    def _inject_compaction_event(self, trigger: str) -> None:
        """Put a synthetic compaction marker onto the current output queue."""
        if self._current_out_q is not None:
            self._current_out_q.put_nowait(
                {"_synthetic": "compaction", "trigger": trigger}
            )
            log.warning("[%s] injected compaction event (trigger=%s)", self._sid, trigger)

    def _with_hooks(self) -> ClaudeAgentOptions:
        """Return options with PreCompact and PreToolUse hooks registered."""
        async def _on_precompact(hook_input, _tool_use_id, _ctx):
            self._inject_compaction_event(hook_input.get("trigger", "auto"))
            return {}

        guard = create_workspace_guard(self._workspace)

        existing_hooks = self._options.hooks or {}
        merged: dict = {**existing_hooks}
        merged["PreCompact"] = [
            HookMatcher(matcher=None, hooks=[_on_precompact]),
        ]
        merged["PreToolUse"] = [
            HookMatcher(matcher="Read|Write|Edit|Glob|Grep", hooks=[guard]),
        ]
        return dataclasses.replace(self._options, hooks=merged)

    async def _run(self) -> None:
        client = ClaudeSDKClient(options=self._with_hooks())
        self._client = client
        try:
            await client.connect()
        except BaseException as e:
            log.error("[%s] worker: connect failed", self._sid, exc_info=True)
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
                    log.error("[%s] worker: CLIConnectionError, exiting", self._sid, exc_info=True)
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
        content_type = "list" if isinstance(content, list) else "str"
        content_size = len(content) if isinstance(content, str) else len(content)
        log.info("[%s] turn start content_type=%s content_size=%d", self._sid, content_type, content_size)

        t0 = time.monotonic()
        event_count = 0
        self._turn_event_count = 0
        self._last_event_info = "querying"
        self._last_event_time = t0
        self._current_out_q = output_q
        try:
            if isinstance(content, str):
                await client.query(content)
            else:
                await client.query(self._make_prompt(content))

            self._last_event_info = "awaiting first event"
            self._last_event_time = time.monotonic()

            async for msg in client.receive_response():
                event_count += 1
                self._turn_event_count = event_count
                self._last_event_time = time.monotonic()
                self._last_event_info = _describe_message(msg, self)
                await output_q.put(msg)

            log.info("[%s] turn complete events=%d duration=%.1fs", self._sid, event_count, time.monotonic() - t0)

        except CLIConnectionError:
            log.error("[%s] turn crashed after events=%d duration=%.1fs last_event='%s'", self._sid, event_count, time.monotonic() - t0, self._last_event_info, exc_info=True)
            await output_q.put(CLIConnectionError("CLI connection lost"))
            raise
        except Exception as e:
            log.error("[%s] turn error after events=%d duration=%.1fs last_event='%s'", self._sid, event_count, time.monotonic() - t0, self._last_event_info, exc_info=True)
            await output_q.put(e)
        finally:
            self._current_out_q = None
            await output_q.put(_DONE)
            self._last_event_info = "idle"

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

    async def send(self, content: str | list[dict]) -> AsyncIterator[Message | dict]:
        """Send content and yield response Messages (or synthetic dicts from hooks).

        This can be called from any asyncio Task — it communicates with the
        worker via Queues so the actual SDK calls stay in the worker Task.
        """
        if not self.alive:
            raise CLIConnectionError("Worker is not running")

        output_q: asyncio.Queue = asyncio.Queue()
        await self._input.put((content, output_q))

        _HEARTBEAT_INTERVAL = 60  # log a heartbeat every 60s of silence
        _TOTAL_TIMEOUT = 300

        elapsed_silent = 0.0
        while True:
            chunk_timeout = min(_HEARTBEAT_INTERVAL, _TOTAL_TIMEOUT - elapsed_silent)
            try:
                item = await asyncio.wait_for(output_q.get(), timeout=chunk_timeout)
            except asyncio.TimeoutError:
                elapsed_silent += chunk_timeout
                diag = self._subprocess_diag()
                if elapsed_silent >= _TOTAL_TIMEOUT:
                    age = time.monotonic() - self._last_event_time if self._last_event_time else 0
                    log.error(
                        "[%s] send timeout after %.0fs output_q.qsize=%d alive=%s "
                        "worker_events=%d last_event='%s' last_event_age=%.1fs "
                        "subprocess=[%s]",
                        self._sid, elapsed_silent, output_q.qsize(), self.alive,
                        self._turn_event_count, self._last_event_info, age,
                        diag,
                    )
                    raise CLIConnectionError("Worker did not respond within timeout")
                # Heartbeat — worker is still alive but no events
                age = time.monotonic() - self._last_event_time if self._last_event_time else 0
                log.warning(
                    "[%s] send waiting %.0fs output_q.qsize=%d alive=%s "
                    "worker_events=%d last_event='%s' last_event_age=%.1fs "
                    "subprocess=[%s]",
                    self._sid, elapsed_silent, output_q.qsize(), self.alive,
                    self._turn_event_count, self._last_event_info, age,
                    diag,
                )
                continue
            # Got an item — reset silence timer
            elapsed_silent = 0.0
            if item is _DONE:
                break
            if isinstance(item, BaseException):
                raise item
            yield item

    async def stop(self) -> None:
        """Ask the worker to shut down and wait for it to finish."""
        if self._task is None or self._task.done():
            return
        log.info("[%s] worker: stop requested", self._sid)
        await self._input.put(None)
        try:
            await asyncio.wait_for(self._task, timeout=15)
            log.info("[%s] worker: stopped gracefully", self._sid)
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

    def _subprocess_diag(self) -> str:
        """Return subprocess diagnostic string for logging.

        Probes internal SDK attributes to check CLI process state.
        """
        client = self._client
        if client is None:
            return "no_client"
        transport = getattr(client, "_transport", None)
        if transport is None:
            return "no_transport"
        process = getattr(transport, "_process", None)
        if process is None:
            return "no_process"
        pid = getattr(process, "pid", None)
        returncode = getattr(process, "returncode", "?")
        ready = getattr(transport, "_ready", "?")
        stdin_open = getattr(transport, "_stdin_stream", None) is not None
        stdout_open = getattr(transport, "_stdout_stream", None) is not None
        return f"pid={pid} returncode={returncode} ready={ready} stdin={stdin_open} stdout={stdout_open}"


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

    def _build_options(self, workspace: Path, session_id: str = "") -> ClaudeAgentOptions:
        patent_server = create_patent_tools_server(workspace)
        sid = session_id[:8]

        def _on_stderr(line: str) -> None:
            log.error("[%s] CLI stderr: %s", sid, line.rstrip())

        return ClaudeAgentOptions(
            system_prompt=get_orchestrator_prompt(workspace),
            cwd=str(workspace),
            allowed_tools=[
                "Read", "Write", "Grep", "Glob", "Bash", "Task",
                "mcp__patent-tools__FetchPatent",
            ],
            permission_mode="acceptEdits",
            max_turns=50,
            include_partial_messages=True,
            agents=get_agent_definitions(workspace),
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
            task = worker._task
            exc = task.exception() if task and task.done() and not task.cancelled() else None
            log.warning(
                "[%s] worker died (cancelled=%s exception=%s), recreating",
                session_id[:8],
                task.cancelled() if task else "N/A",
                exc,
            )
            self._workers.pop(session_id, None)

        options = self._build_options(workspace, session_id)
        worker = _SessionWorker(session_id, options, workspace)
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
    ) -> AsyncIterator[Message | dict]:
        """Send a message and yield response messages (or synthetic hook dicts).

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
                    "[%s] CLIConnectionError — reconnecting; events from failed attempt were already streamed",
                    session_id[:8],
                    exc_info=True,
                )
                await self._force_disconnect(session_id)
                try:
                    async for msg in self._do_send(session_id, workspace, content):
                        yield msg
                except CLIConnectionError:
                    log.error(
                        "[%s] retry also failed", session_id[:8],
                        exc_info=True,
                    )
                    await self._force_disconnect(session_id)
                    raise

    async def _do_send(
        self,
        session_id: str,
        workspace: Path,
        content: str | list[dict],
    ) -> AsyncIterator[Message | dict]:
        """Send content to client and yield messages until ResultMessage."""
        worker = await self._get_or_create_worker(session_id, workspace)
        self._last_active[session_id] = time.monotonic()

        async for msg in worker.send(content):
            yield msg
        self._last_active[session_id] = time.monotonic()

    async def disconnect(self, session_id: str) -> None:
        """Disconnect and fully clean up a session (including its lock)."""
        await self._force_disconnect(session_id)
        self._locks.pop(session_id, None)

    async def _force_disconnect(self, session_id: str) -> None:
        """Stop the worker and remove session state.

        Does NOT remove the per-session lock — send_message may still hold it,
        and cleanup_idle relies on lock.locked() to skip active sessions.
        Locks are cleaned up by disconnect() and shutdown().
        """
        log.info("[%s] force disconnect", session_id[:8])
        worker = self._workers.pop(session_id, None)
        self._last_active.pop(session_id, None)
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
