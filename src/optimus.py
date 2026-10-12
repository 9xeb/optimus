import asyncio
import os

import numpy
import dspy
import mlflow

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from src.utils import stream_dspy_program, require_approval
from src.log import log_internal_event, log_error

from src.ctxseg import CtxSeg
from src.deact import DeAct

class Optimus():
    """
    Adds an interface to CtxSeg module.
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

        # self.tasks = asyncio.Queue()
        self.tasks = {}
        # self.tasks_queue = asyncio.Queue()      # branches and leaves
        # self.questions = asyncio.Queue()

        self.approval = approval

        # set of optimization rules. Acts as dspy dataset for LLM-powered optimizers.
        self.constitution = {"Article 1": input("Define constitution > ")}
        self.judge = dspy.ChainOfThought("constitution, trace -> verdict, approved: bool")


    async def solve(self, task):
        """
        Consume a problem with CtxSeg+MCP
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

                    agent = CtxSeg(tools=mcp_tools, tasks=self.tasks)
                    optimus_output = await stream_dspy_program(
                        agent,
                        task=task
                    )
                    print("CtxSeg is finished", flush=True)
                    # print(f"Traces: {agent.unclassified_traces}")
                    print(f"Report: {optimus_output.report}")
        except Exception as e:
            log_error(f"MCP Tool error or timeout: {e}")
            return "Optimus error: {e}"
        # log_internal_event(f"[OPTIMUS] Got report: {optimus_output.report}")
        return optimus_output.report

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
        # Wrap around GEPA to optimize CtxSeg.explore programs
        # Consider if self.tasks can be used as part of dataset
        pass
