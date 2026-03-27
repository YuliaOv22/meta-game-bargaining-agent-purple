"""A2A protocol executor that routes requests to Agent instances."""

import logging

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    TaskState,
    UnsupportedOperationError,
    InvalidRequestError,
)
from a2a.utils.errors import ServerError
from a2a.utils import (
    new_agent_text_message,
    new_task,
)

from agent import Agent

logger = logging.getLogger(__name__)

TERMINAL_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected,
}


class Executor(AgentExecutor):
    """Routes incoming A2A requests to per-context Agent instances."""

    def __init__(self):
        """Initialize with an empty agent registry."""
        self.agents: dict[str, Agent] = {}  # context_id to agent instance

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Handle an incoming A2A request: find or create an Agent and run it."""
        msg = context.message
        if not msg:
            raise ServerError(
                error=InvalidRequestError(message="Missing message in request")
            )

        task = context.current_task
        if task and task.status.state in TERMINAL_STATES:
            raise ServerError(
                error=InvalidRequestError(
                    message=f"Task {task.id} already processed (state: {task.status.state})"
                )
            )

        if not task:
            task = new_task(msg)
            await event_queue.enqueue_event(task)

        context_id = task.context_id
        agent = self.agents.get(context_id)
        if not agent:
            agent = Agent()
            self.agents[context_id] = agent

        updater = TaskUpdater(event_queue, task.id, context_id)

        await updater.start_work()
        try:
            await agent.run(msg, updater)
            try:
                terminal_reached = updater._terminal_state_reached
            except AttributeError:
                terminal_reached = False
            if not terminal_reached:
                await updater.complete()
        except Exception:
            logger.exception("Task %s failed with agent error", task.id)
            await updater.failed(
                new_agent_text_message(
                    "Internal agent error. Check server logs for details.",
                    context_id=context_id,
                    task_id=task.id,
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel is not supported."""
        raise ServerError(error=UnsupportedOperationError())
