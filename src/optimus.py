import asyncio
from dataclasses import dataclass, field
import json
import os
from pathlib import Path

import numpy
import dspy
import mlflow

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from src.utils import require_approval, State, Task
from src.log import log_internal_event, log_error

from src.ctxseg import RCS
# from src.deact import DeAct

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
            reasoning_effort="none",                          # we let ReAct do the reasoning
            headers={"x-opencode-session": "optimus-agents"}, # necessary for OpenCode GO
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

        # Shared data between async Backend <-> Optimus <-> RCS
        self.state = State(
            filepath=filepath,      # RCS needs it to update memory file after clean execution
            tasks={},               # RCS working memory
            qna={},                 # RCS long term memory, a.k.a examples
            parent=None,            # RCS parent task cursor, Frontend uses to graphically link situations to task
            current=None,           # RCS current task cursor
            discovered_tools=[],    # RCS tracks discovered tools, clears with new task
            approval=approval,      # Instructs RCS to for approvals or not
            justification=None,     # RCS enriches tool calls with motive, user approves via Backend
            situation={},           # RCS generates situations for user
            resolution=None,        # Situation feedback from user via Backend
            event=asyncio.Event(),  # RCS and Backend signal each other when events happen
            tokens=asyncio.Queue(), # Main stream of all tokens out of RCS
            job=None                # Backend ensures consistency against incoming requests
        )
        # Load memories from file
        if self.state.filepath and Path(self.state.filepath).exists():
            log_internal_event(f"[Optimus] Retrieving memories from file: {self.state.filepath}")
            with open(self.state.filepath, 'r', encoding='utf-8') as f:
                memories = json.loads(f.read())
                self.state.tasks = memories.get("tasks")
                self.state.qna = memories.get("qna")
                log_internal_event(f"[Optimus] Known Tasks: {json.dumps(self.state.tasks, indent=4)}")
                log_internal_event(f"[Optimus] Known Examples: {json.dumps(self.state.qna, indent=4)}")

        self.approval = approval

    async def grow(self, task):
        """
        Grow the memories while solving the tasks in the meantime
        This is the section that retrieves and wraps MCP tools around approval gates
        """
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

                    gated_mcp_tools = [
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

                    def list_tools():
                        """Get the list of available tools."""
                        list_of_tools = [{"name": tool.name, "args": list(tool.args.keys())} for tool in mcp_tools]
                        log_internal_event(f"[OPTIMUS] Listing tools: {list_of_tools}")
                        return list_of_tools
                    def describe_tools(names: list[str]):
                        """Get descriptions of the most promising tools, by tool name."""
                        match = [tool for tool in mcp_tools if tool.name in names]
                        log_internal_event(f"[OPTIMUS] Discovered tools: {match}")
                        self.state.discovered_tools += names
                        return match
                    @dataclass
                    class ToolCall:
                        """Tool call payload"""
                        goal: str = field(metadata={"description": "What the call is meant to achieve."})
                        name: str
                        args: dict
                        readonly: bool = field(metadata={"description": "True if the call is a read only operation. False otherwise."})
                    async def run_tools(calls: list[ToolCall]):
                        """
                        Run a list of tools by name.
                        Run as many tool calls as possible in a single pass.
                        """
                        list_flag = False   # track if tools already listed to avoid spam in the same batch
                        output = []
                        for call in calls:
                            log_internal_event(f"[OPTIMUS] Trying to call tool {call}")
                            log_internal_event(f"[OPTIMUS] Discovered tools: {self.state.discovered_tools}")
                            try:
                                # 1. Find matching tools by name (and their approval-gated counterparts)
                                match = [tool for tool in mcp_tools if tool.name == call.name]
                                gated_match = [tool for tool in gated_mcp_tools if tool.name == call.name]
                                if len(match) > 0:
                                    # Run the first maching tool
                                    # Check if all args exist in signature
                                    if all([arg in list(match[0].args.keys()) for arg in list(call.args.keys())]):
                                        if call.readonly:
                                            # Direct call
                                            log_internal_event(f"[OPTIMUS] Tool call [{call.name}] advertised as read-only.")
                                            # Await calls so it works both for async and sync
                                            tool_output = await match[0].acall(**call.args)
                                        else:
                                            # Wait for user approval
                                            log_internal_event(f"[OPTIMUS] Tool call [{call.name}] advertised as not read-only. Requiring approval.")
                                            self.state.justification = call.goal  # register the justification for user interface
                                            tool_output = await gated_match[0].acall(**call.args)
                                        log_internal_event(f"[OPTIMUS] Tool {call.name} output: {tool_output}")

                                        # Update tool discovery with piggybacking
                                        if call.name in self.state.discovered_tools:
                                            log_internal_event(f"[OPTIMUS] Tool already discovered: {call.name}. Skipping piggybacking.")
                                            output += [{"call": {"name": call.name, "args": call.args}, "output": tool_output}]
                                        else:
                                            log_internal_event(f"[OPTIMUS] Piggybacking tool description after running tool: {call.name}.")
                                            output += [
                                                {
                                                    "call": {"name": call.name, "args": call.args},
                                                    "signature": describe_tools([call.name]),
                                                    "output": tool_output
                                                }
                                            ]
                                    else:
                                        # Get here if LLM gave broken args
                                        log_internal_event(f"[OPTIMUS] Tool call violates signature")
                                        output += [
                                            {
                                                "call": {"name": call.name, "args": call.args},
                                                "signature": describe_tools([call.name]) if call.name in self.state.discovered_tools else f"See previous {call.name} description.",
                                                "error": f"Error: wrong arguments for tool {call.name}."
                                            }
                                        ]
                                else:
                                    # Get here if LLM gave broken tool name
                                    output += [
                                        {
                                            "call": {"name": call.name, "args": call.args},
                                            "error": f"Error, tool name {call.name} does not exist."
                                        } | {"available_tools": list_tools()} if not list_flag else {}
                                    ]
                                    list_flag = True
                            except Exception as e:
                                # Land any unexpected exceptions here for the LLM to observe
                                log_error(f"[OPTIMUS] Critical dynamic tool call error: {e}")
                                output += [
                                    {
                                        "call": {"name": call.name, "args": call.args},
                                        "signature": describe_tools([call.name]) if call.name in self.state.discovered_tools else f"See previous {call.name} description.",
                                        "error": e
                                    }
                                ]
                        # LLM observes good calls and bad calls
                        return output
                        
                    dynamic_tools = [
                        dspy.Tool(list_tools),
                        dspy.Tool(describe_tools),
                        dspy.Tool(run_tools)
                    ]
                    log_internal_event(f"[OPTIMUS] Dynamic tools: {dynamic_tools}")
                    # This is the entrypoint of the repeated context segmentation
                    agent = RCS(
                        state=self.state,
                        tools=dynamic_tools,
                        info={}
                    )

                    # Starts the async RCS task
                    agent_task = await agent.acall(task=task)
        except Exception as e:
            log_error(f"Optimus critical error: {e}")
            raise e

    async def stop(self):
        """Stop a running job"""
        if not self.state.job.done():
            self.state.job.cancel()
            # Let it observe CancelledError and run its own cleanup
            await asyncio.gather(self.state.job, return_exceptions=True)
            log_error(f"Job cancelled by client.")
