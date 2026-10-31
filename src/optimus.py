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
            reasoning_effort="none",    # we let ReAct do the reasoning
            headers={"x-opencode-session": "optimus-agents"}, # necessary for OpenCode GO
            cache=False,     # stored in ~/.dspy_cache
        )
        # dspy.configure(lm=self.lm, callbacks=[LoggingCallback()])      # set default provider locally, can override with dspy.context
        # dspy.configure(lm=lm, adapter=dspy.ChatAdapter(use_native_function_calling=True))
        dspy.configure(lm=lm, adapter=dspy.JSONAdapter())
        # dspy.configure(lm=lm)
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
            qna={},                 # RCS long term memory
            parent=None,            # RCS internal state management, Frontend uses to link situations to task
            current=None,           # RCS internal state management, used for memory management
            discovered_tools=[],    # RCS manages when switching contexts
            approval=approval,      # RCS can ask for tool call and agents approval
            situation={},           # RCS produces, Backend consumes
            resolution=None,
            event=asyncio.Event(),  # RCS and Backend signal each other
            tokens=asyncio.Queue(), # Main stream of tokens out of RCS
            job=None                # Backend needs it to ensure consistency regardless of what Frontend does
        )
        # Load memories from file
        if self.state.filepath and Path(self.state.filepath).exists():
            log_internal_event(f"[Optimus] Retrieving memories from file: {self.state.filepath}")
            with open(self.state.filepath, 'r', encoding='utf-8') as f:
                memories = json.loads(f.read())
                # log_internal_event(f"Loaded tasks from file: {json.dumps(tasks, indent=4)}")
                self.state.tasks = memories.get("tasks")
                self.state.qna = memories.get("qna")
                log_internal_event(f"[Optimus] Task memories: {json.dumps(self.state.qna, indent=4)}")
                log_internal_event(f"[Optimus] QnA memories: {json.dumps(self.state.tasks, indent=4)}")

        # # Populate tasks from file
        # if filepath and Path(filepath).exists():
        #     with open(filepath, 'r', encoding='utf-8') as f:
        #         tasks = json.loads(f.read())
        #         log_internal_event(f"Loaded tasks from file: {json.dumps(tasks, indent=4)}")
        #         self.state.tasks = tasks

        # self.tasks = self.state["tasks"]    # for easier access in RCS
        # self.tasks = {}
        # self.shared_data = {"event": asyncio.Event(), "question": None, "answer": None}

        self.approval = approval

        # set of optimization rules. Acts as dspy dataset for LLM-powered optimizers.
        # self.constitution = {"Article 1": input("Define constitution > ")}
        # self.judge = dspy.ChainOfThought("constitution, trace -> verdict, approved: bool")

    # async def pick(self, task_title: str):
    #     """Run an existing task with its dependencies as-is"""
    #     # Useful for cron jobs and manual user clicks from the console
    #     # This uses the known tasks network to collect all reports without going through the reasoning that originally grew the tasks.
    #     # It is considerably cheaper, being just a recursive summarization of reports.            
    #     pass

    # TODO: rename 'solve' to 'grow' or 'explore' or 'feed'
    async def grow(self, task):
        """
        Consume a problem with RCS+MCP
        """
        # log_error(f"[OPTIMUS] Grow: {task}")
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

                    # def memory(title: str):
                    #     """Search memory for previously seen tasks"""
                    #     if self.state.tasks.get(title):
                    #         return self.state.tasks[title]
                    #     else:
                    #         return f"Error, the requested memory does not exist. Available memories: {list(self.state.tasks.keys())}"
                    def list_tools():
                        """Get the list of available tools."""
                        list_of_tools = [{"name": tool.name, "args": list(tool.args.keys())} for tool in mcp_tools]
                        log_internal_event(f"[OPTIMUS] Listing tools: {list_of_tools}")
                        return list_of_tools
                    def describe_tools(names: list[str]):
                        # """Describe the signature of a tool"""
                        """Get descriptions of the most promising tools, by tool name."""
                        match = [tool for tool in mcp_tools if tool.name in names]
                        log_internal_event(f"[OPTIMUS] Discovering tools for this session: {match}")
                        self.state.discovered_tools += names
                        return match
                    @dataclass
                    class ToolCall:
                        """Tool call payload"""
                        name: str
                        args: dict
                    async def run_tools(calls: list[ToolCall]):
                        """
                        Run a list of tools by name.
                        Run as many tool calls as possible in a single pass.
                        """
                        # Try to batch as many calls as you can, as that is very efficient.
                        list_flag = False   # track if tools already listed to avoid spam
                        output = []
                        for call in calls:
                            log_internal_event(f"[OPTIMUS] Trying to call tool {call}")
                            log_internal_event(f"[OPTIMUS] Discovered tools: {self.state.discovered_tools}")
                            try:
                                match = [tool for tool in mcp_tools if tool.name == call.name]
                                if len(match) > 0:
                                    # If all requested args are in tool signature, proceed
                                    if all([arg in list(match[0].args.keys()) for arg in list(call.args.keys())]):
                                        # Run the first tool that matches name
                                        # Await so it works both for async (with approval) and sync (no approval)
                                        tool_output = await match[0].acall(**call.args)
                                        log_internal_event(f"[OPTIMUS] Tool {call.name} output: {tool_output}")
                                        log_error(f"[OPTIMUS] discovered tools: {self.state.discovered_tools}")
                                        if call.name in self.state.discovered_tools:
                                            log_internal_event(f"[OPTIMUS] Tool already discovered: {call.name}. Skipping piggybacking.")
                                            # return tool_output
                                            # output += [{"call": {"name": call.name, "args": call.args}, "output": tool_output}]
                                            output += [{"call": {"name": call.name, "args": call.args}, "output": tool_output}]
                                        else:
                                            log_internal_event(f"[OPTIMUS] Piggybacking tool description after running tool: {call.name}.")
                                            # return {
                                            #     "tool_description": describe_tools([call.name]),
                                            #     "tool_call_output": tool_output if len(tool_output) > 0 else "Tool returned empty output"
                                            # }
                                            output += [
                                                {
                                                    "call": {"name": call.name, "args": call.args},
                                                    "signature": describe_tools([call.name]),
                                                    # "signature": describe_tools([call.name]) if (call.name in self.state.discovered_tools) else f"See previous {call.name} description.",
                                                    "output": tool_output
                                                    # "output": tool_output if len(tool_output.output) > 0 else "Tool returned empty output",
                                                }
                                            ]
                                    else:
                                        log_internal_event(f"[OPTIMUS] Tool call violates signature")
                                        # return f"Error, tool call: [{call}] does not match signature. Tool signature: {describe_tools([call.name])}"
                                        # return f"Error, tool call: [{call}] does not match signature. Tool signature: {describe_tools([call.name])}"
                                        output += [
                                            {
                                                "call": {"name": call.name, "args": call.args},
                                                "signature": describe_tools([call.name]) if call.name in self.state.discovered_tools else f"See previous {call.name} description.",
                                                "error": f"Error: wrong arguments for tool {call.name}."
                                            }
                                        ]
                                else:
                                    # return f"Error, tool does not exist. Available tools: {list_tools()}"
                                    output += [
                                        {
                                            "call": {"name": call.name, "args": call.args},
                                            "error": f"Error, tool name {call.name} does not exist."
                                        } | {"available_tools": list_tools()} if not list_flag else {}
                                    ]
                                    list_flag = True
                            except Exception as e:
                                log_error(f"[OPTIMUS] Critical dynamic tool call error: {e}")
                                output += [
                                    {
                                        "call": {"name": call.name, "args": call.args},
                                        "signature": describe_tools([call.name]) if call.name in self.state.discovered_tools else f"See previous {call.name} description.",
                                        "error": e
                                    }
                                ]
                        # log_internal_event(f"[OPTIMUS] Tool calls: {output}")
                        return output
                        
                    dynamic_tools = [
                        # dspy.Tool(memory),
                        dspy.Tool(list_tools),
                        dspy.Tool(describe_tools),
                        dspy.Tool(run_tools)
                    ]
                    log_internal_event(f"[OPTIMUS] Dynamic tools: {dynamic_tools}")

                    # agent = RCS(tools=mcp_tools, tasks=self.tasks, shared_data=self.shared_data)
                    # agent = RCS(state=self.state, tools=mcp_tools)
                    agent = RCS(
                        state=self.state,
                        tools=dynamic_tools,
                        info={}
                    )

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
