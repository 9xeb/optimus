import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy
import dspy
import mlflow

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from src.utils import stream_dspy_program, require_approval, State, Task
from src.log import log_internal_event, log_error

from src.ctxseg import RCS
from src.deact import DeAct

class Optimus():
    """
    Adds an interface to RCS module.
    Support env vars for connecting to OpenAI compatible APIs, Mlflow and a single MCP server.
    """
    def __init__(self, filepath: str = None, approval: bool = False):
        lm = dspy.LM(
            os.environ["OPENAI_API_MODEL"],
            api_base=os.environ["OPENAI_API_BASE"],
            api_key=os.environ["OPENAI_API_KEY"],
            cache=False,     # stored in ~/.dspy_cache
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

        # Shareable async state between Backend, Optimus and RCS
        self.state = State(
            filepath=filepath,      # RCS needs it to update memory file after clean execution
            tasks={},               # RCS working memory
            focus=None,             # RCS internal state management, Frontend uses to link situations to task
            situation={},           # RCS produces, Backend consumes
            resolution=None,
            event=asyncio.Event(),  # RCS and Backend signal each other
            tokens=asyncio.Queue(), # Main stream of tokens out of RCS
            job=None                # Backend needs it to ensure consistency regardless of what Frontend does
        )
        # Populate tasks from file
        if filepath and Path(filepath).exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                tasks = json.loads(f.read())
                log_internal_event(f"Loaded tasks from file: {json.dumps(tasks, indent=4)}")
                self.state.tasks = tasks

        # self.tasks = self.state["tasks"]    # for easier access in RCS
        # self.tasks = {}
        # self.shared_data = {"event": asyncio.Event(), "question": None, "answer": None}

        self.approval = approval

        # set of optimization rules. Acts as dspy dataset for LLM-powered optimizers.
        # self.constitution = {"Article 1": input("Define constitution > ")}
        # self.judge = dspy.ChainOfThought("constitution, trace -> verdict, approved: bool")

    async def pick(self, task_title: str):
        """Run an existing task with its dependencies as-is"""
        # Useful for cron jobs and manual user clicks from the console
        # This uses the known tasks network to collect all reports without going through the reasoning that originally grew the tasks.
        # It is considerably cheaper, being just a recursive summarization of reports.
        pass

    # TODO: rename 'solve' to 'grow' or 'explore' or 'feed'
    async def grow(self, task):
        """
        Consume a problem with RCS+MCP
        """
        log_error(f"[OPTIMUS] Grow: {task}")
        try:
            async with streamable_http_client(os.environ["MCP_GATEWAY"]) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()

                    mcp_tools = [
                        dspy.Tool.from_mcp_tool(session, tool)
                        for tool in tools.tools
                        if tool.name != "sandbox_execute_code" # hide problematic tool
                    ]

                    if self.approval:
                        log_internal_event("[OPTIMUS] Enabling tool approval")
                        mcp_tools = [
                            dspy.Tool(
                                require_approval(self.state, tool.func, tool.name),
                                name=tool.name,
                                desc=tool.desc,
                                args=tool.args,
                                arg_types=tool.arg_types
                            )
                            for tool in mcp_tools
                        ]

                    # Tools have .name, .desc that can be used for discovery
                    log_internal_event(f"MCP TOOLS: {[tool.name for tool in mcp_tools]}")
                    # log_internal_event(f"MCP ARGS: {[{"name": tool.name, "args": tool.args} for tool in mcp_tools]}")

                    def get_tools():
                        """Get the list of available tools."""
                        list_of_tools = [tool.name for tool in mcp_tools]
                        log_internal_event(f"[OPTIMUS] Listing tools: {list_of_tools}")
                        return list_of_tools
                    def describe_tool(name: str):
                        """Describe the signature of a tool."""
                        match = [tool for tool in mcp_tools if tool.name == name]
                        log_internal_event(f"[OPTIMUS] Describing tool: {match}")
                        return match
                    @dataclass
                    class ToolCall:
                        """Tool call payload"""
                        name: str
                        args: dict
                    def run_tool(call: ToolCall):
                        """Run a tool, following its signature."""
                        log_internal_event(f"[OPTIMUS] Trying to call tool {call}")
                        try:
                            match = [tool for tool in mcp_tools if tool.name == call.name]
                            if len(match) > 0:
                                # If all requested args are in tool signature, proceed
                                if all([arg in list(match[0].args.keys()) for arg in list(call.args.keys())]):
                                    # Run the first tool that matches name
                                    # Await so it works both for async (with approval) and sync (no approval)
                                    return match[0].acall(**call.args)
                                else:
                                    return f"Error, tool call does not match signature. Tool signature: {describe_tool(call.name)}"
                            else:
                                return f"Error, tool does not exist. Available tools: {get_tools()}"
                        except Exception as e:
                            log_error(f"[OPTIMUS] Dynamic tool call error: {e}")

                    dynamic_tools = [
                        dspy.Tool(get_tools),
                        dspy.Tool(describe_tool),
                        dspy.Tool(run_tool)
                    ]
                    log_internal_event(f"[OPTIMUS] Dynamic tools: {dynamic_tools}")

                    # agent = RCS(tools=mcp_tools, tasks=self.tasks, shared_data=self.shared_data)
                    # agent = RCS(state=self.state, tools=mcp_tools)
                    agent = RCS(state=self.state, tools=dynamic_tools)

                    # This streams tokens from the recursive ReAct calls inside RCS
                    # TODO: scale asyncio.create_task for parallel task executions
                    agent_task = await agent.acall(task=task)
                    # try:
                    #     agent_task = await agent.acall(task=task)
                    #     # agent_task = asyncio.create_task(agent.acall(task=task))
                    #     # await agent_task
                    #     # await asyncio.gather(agent_task)
                    # except asyncio.CancelledError:
                    #     log_internal_event(f"[OPTIMUS] Cleaning up after client cancellation.")
                    #     # Clean up after cancellation
                    #     for task in self.state.tasks:
                    #         if task.get("state") == "awake":
                    #             task["state"] == "sleeping"
                    #     agent_task.cancel() # propagate cancellation
                    # return agent_task
                    # try:
                    #     while True:
                    #         # chunk = await self.tokens.get()
                    #         chunk = await agent.tokens.get()
                    #         if chunk is None:
                    #             break
                    #         yield chunk
                    #     # Surface agent exceptions instead of losing time
                    #     await agent_task
                    # finally:
                    #     if not agent_task.done():
                    #         agent_task.cancel()
                    #         # Let it observe CancelledError and run its own cleanup
                    #         await asyncio.gather(agent_task, return_exceptions=True)
                    #         log_error(f"Task {task} cancelled by client.")
                    #     else:
                    #         # Save to file
                    #         pass
        except Exception as e:
            log_error(f"MCP Tool error or timeout: {e}")
            raise e

    async def stop(self):
        """Stop a running job"""
        if not self.state.job.done():
            self.state.job.cancel()
            # Let it observe CancelledError and run its own cleanup
            await asyncio.gather(self.state.job, return_exceptions=True)
            log_error(f"Job cancelled by client.")

    # async def consume_tasks(self):
    #     """
    #     Main optimus loop function, with message queue streaming.
    #     """
    #     while True:
    #         task = await self.tasks.get()
    #         if task is None:
    #             return
    #         self.solve(task=task)

    # async def optimize(self, task):
    #     """Optimize programs for a specific task"""
    #     # Use self.constitution as dataset
    #     # Wrap around GEPA to optimize RCS.explore programs
    #     # Consider if self.tasks can be used as part of dataset
    #     pass
