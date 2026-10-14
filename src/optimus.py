import asyncio
import os

import numpy
import dspy
import mlflow

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from src.utils import stream_dspy_program, require_approval
from src.log import log_internal_event, log_error

from src.ctxseg import RCS
from src.deact import DeAct

class Optimus():
    """
    Adds an interface to RCS module.
    Support env vars for connecting to OpenAI compatible APIs, Mlflow and a single MCP server.
    """
    def __init__(self, approval: bool = False):
        lm = dspy.LM(
            os.environ["OPENAI_API_MODEL"],
            api_base=os.environ["OPENAI_API_BASE"],
            api_key=os.environ["OPENAI_API_KEY"],
            cache=False     # stored in ~/.dspy_cache
        )
        # dspy.configure(lm=self.lm, callbacks=[LoggingCallback()])      # set default provider locally, can override with dspy.context
        # dspy.configure(lm=lm, adapter=dspy.ChatAdapter(use_native_function_calling=True))
        dspy.configure(lm=lm, adapter=dspy.JSONAdapter())
        dspy.disable_litellm_logging()

        # Setup mlflow integration
        if os.environ.get("MLFLOW_API_BASE"):
            mlflow.set_tracking_uri(os.environ["MLFLOW_API_BASE"])
            mlflow.set_experiment("Optimus")
            mlflow.autolog()

        self.tasks = {}

        self.approval = approval

        # set of optimization rules. Acts as dspy dataset for LLM-powered optimizers.
        # self.constitution = {"Article 1": input("Define constitution > ")}
        # self.judge = dspy.ChainOfThought("constitution, trace -> verdict, approved: bool")

    async def pick(self, task_title: str):
        """Run an existing task with its dependencies as-is"""
        # Useful for cron jobs and manual user clicks from the console
        # WARNING: grow can pick so this method is implied by grow
        pass

    # TODO: rename 'solve' to 'grow' or 'explore' or 'feed'
    async def grow(self, task):
        """
        Consume a problem with RCS+MCP
        """
        try:
            async with streamable_http_client(os.environ["MCP_GATEWAY"]) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()

                    mcp_tools = [
                        dspy.Tool.from_mcp_tool(session, tool)
                        for tool in tools.tools
                    ]

                    if self.approval:
                        mcp_tools = [
                            dspy.Tool(
                                require_approval(tool.func, tool.name),
                                name=tool.name,
                                desc=tool.desc,
                                args=tool.args,
                                arg_types=tool.arg_types
                            )
                            for tool in mcp_tools
                        ]

                    # Tools have .name, .desc that can be used for discovery
                    log_internal_event(f"MCP TOOLS: {[tool.name for tool in mcp_tools]}")

                    agent = RCS(tools=mcp_tools, tasks=self.tasks)

                    # This streams tokens from the recursive ReAct calls inside RCS
                    # TODO: scale asyncio.create_task for parallel task executions
                    agent_task = asyncio.create_task(agent.acall(task=task))
                    try:
                        while True:
                            # chunk = await self.tokens.get()
                            chunk = await agent.tokens.get()
                            if chunk is None:
                                break
                            yield chunk
                        # Surface agent exceptions instead of losing time
                        await agent_task
                    finally:
                        if not agent_task.done():
                            agent_task.cancel()
                            # Let it observe CancelledError and run its own cleanup
                            await asyncio.gather(agent_task, return_exceptions=True)
                            log_error(f"Task {task} cancelled by client.")
        except Exception as e:
            log_error(f"MCP Tool error or timeout: {e}")
            raise e

    async def consume_tasks(self):
        """
        Main optimus loop function, with message queue streaming.
        """
        while True:
            task = await self.tasks.get()
            if task is None:
                return
            self.solve(task=task)

    async def optimize(self, task):
        """Optimize programs for a specific task"""
        # Use self.constitution as dataset
        # Wrap around GEPA to optimize RCS.explore programs
        # Consider if self.tasks can be used as part of dataset
        pass
