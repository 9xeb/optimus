import asyncio
import os
import json
import functools

import litellm
import dspy
import mlflow

from dspy.utils.callback import BaseCallback
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from src.utils import stream_dspy_program
from src.log import log_internal_event

from src.ctxseg import CtxSeg

class Optimus():
    """
    Adds an interface to CtxSeg module.
    Support env vars for connecting to OpenAI compatible APIs, Mlflow and a single MCP server.
    """
    def __init__(self):
        lm = dspy.LM(
            os.environ["OPENAI_API_MODEL"],
            api_base=os.environ["OPENAI_API_BASE"],
            api_key=os.environ["OPENAI_API_KEY"],
            cache=False     # stored in ~/.dspy_cache
        )
        # dspy.configure(lm=self.lm, callbacks=[LoggingCallback()])      # set default provider locally, can override with dspy.context
        dspy.configure(lm=lm)
        dspy.disable_litellm_logging()

        # Setup mlflow integration
        if os.environ.get("MLFLOW_API_BASE"):
            mlflow.set_tracking_uri(os.environ["MLFLOW_API_BASE"])
            mlflow.set_experiment("Optimus")
            mlflow.autolog()

        self.tasks = asyncio.Queue()
        # self.tasks_queue = asyncio.Queue()      # branches and leaves
        # self.questions = asyncio.Queue()

    async def solve(self, task):
        """
        Consume a problem with CtxSeg+MCP
        """
        async with streamablehttp_client(os.environ["MCP_GATEWAY"]) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()

                mcp_tools = [dspy.Tool.from_mcp_tool(session, tool) for tool in tools.tools]
                # Tools have .name, .desc that can be used for discovery
                log_internal_event(f"MCP TOOLS: {[tool.name for tool in mcp_tools]}")
                
                agent = CtxSeg(tools=mcp_tools)
                optimus_output = await stream_dspy_program(
                    agent,
                    task=task
                )
                # print(f"Traces: {agent.unclassified_traces}")
                # print(optimus_output.report)
        return optimus_output.report_tweet

    async def consume_tasks(self):
        """
        Main optimus loop function, with message queue streaming.
        """
        while True:
            task = await self.tasks.get()
            if task is None:
                return
            self.solve(task=task)
