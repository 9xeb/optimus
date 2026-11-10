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
            filepath=filepath,        # RCS needs it to update memory file after clean execution
            tasks={},                 # RCS working memory
            qna={},                   # RCS long term memory, a.k.a examples
            parent=None,              # RCS parent task cursor, Frontend uses to graphically link situations to task
            current=None,             # RCS current task cursor
            discovered_tools=[],      # RCS tracks discovered tools, clears with new task
            approval=approval,        # Instructs RCS to for approvals or not
            justification=None,       # RCS enriches tool calls with motive, user approves via Backend
            situation={},             # RCS generates situations for user
            resolution=None,          # Situation feedback from user via Backend
            event=asyncio.Event(),    # RCS and Backend signal each other when events happen
            tokens=asyncio.Queue(),   # Main stream of all tokens out of RCS
            actions=asyncio.Queue(),  # High level actions being performed by RCS
            trajectory=[],            # The current execution trajectory, as a list of action strings
            job=None                  # Backend ensures consistency against incoming requests
        )
        # Load memories from file
        if self.state.filepath and Path(self.state.filepath).exists():
            log_internal_event(f"[Optimus] Retrieving memories from file: {self.state.filepath}")
            with open(self.state.filepath, 'r', encoding='utf-8') as f:
                memories = json.loads(f.read())
                self.state.tasks = memories.get("tasks")
                self.state.qna = memories.get("qna")
                log_internal_event(f"[Optimus] Known Tasks: {json.dumps(self.state.tasks, indent=4)}")
                # log_internal_event(f"[Optimus] Known Examples: {json.dumps(self.state.qna, indent=4)}")

        self.approval = approval

    async def grow(self, task: str):
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

                    async def list_tools():
                        """Get the list of available tools."""
                        list_of_tools = [
                            {
                                "name": tool.name,
                                "args": list(tool.args.keys())
                            }
                            for tool in mcp_tools
                        ] + [{"disclaimer": "The tools above must run through the `run_tools` tool."}]
                        log_internal_event(f"[OPTIMUS] Listing tools: {list_of_tools}")
                        await self.state.actions.put({"task": self.state.current.title, "data": "Listing tools..."})
                        return list_of_tools
                    async def describe_tools(names: list[str]):
                        """Get descriptions of the most promising tools, by tool name."""
                        match = [tool for tool in mcp_tools if tool.name in names]
                        log_internal_event(f"[OPTIMUS] Discovered tools: {match}")
                        # await self.state.actions.put({"task": self.state.current.title, "data": "Describing tools..."})
                        self.state.discovered_tools += names
                        return match
                    @dataclass
                    class ToolCall:
                        """Tool call payload"""
                        goal: str = field(metadata={"description": "What the call is meant to achieve."})
                        change: bool = field(metadata={"description": "True if the call makes a potentially breaking change. False otherwise."})
                        name: str
                        args: dict
                        # readonly: bool = field(metadata={"description": "True if the call is a read only operation. False otherwise."})
                        # affects_user: bool = field(metadata={"description": "True if the call makes changes to the property of the user. False otherwise."})
                        # requires_approval: bool = field(metadata={"description": "True if the user should approve this call first. False otherwise."})
                    async def run_tools(calls: list[ToolCall]):
                        """
                        Run a list of tools by name.
                        When iterating, submit the single largest list of calls possible.
                        """
                        # Split as many tool calls as possible in multiple items in the call list.
                        # Split repetitive or looping operations into multiple items in the calls list.
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
                                        # Stream call goal
                                        await self.state.actions.put({"task": self.state.current.title, "data": call.goal})
                                        if not call.change:
                                            # Direct call if read only and previously reviewed task
                                            log_internal_event(f"[OPTIMUS] Tool call [{call.name}] advertised as read-only.")
                                            # Await calls so it works both for async and sync
                                            tool_output = await match[0].acall(**call.args)
                                            # Add to trajectory without approval
                                            # self.state.trajectory += [call.goal]
                                            self.state.tasks[self.state.parent.title]["trajectory"] += [call.goal]
                                        else:
                                            # Wait for user approval
                                            log_internal_event(f"[OPTIMUS] Tool call [{call.name}] advertised as a change. Requiring approval.")
                                            self.state.justification = call.goal  # register the justification for user interface
                                            tool_output = await gated_match[0].acall(**call.args)
                                        log_internal_event(f"[OPTIMUS] Tool [{call.name}] output -> {tool_output[:50]}...")

                                        # Update tool discovery with piggybacking
                                        if call.name in self.state.discovered_tools:
                                            log_internal_event(f"[OPTIMUS] [{call.name}] tool already discovered. Skipping piggybacking.")
                                            output += [{"call": {"name": call.name, "args": call.args}, "output": tool_output}]
                                        else:
                                            log_internal_event(f"[OPTIMUS] Piggybacking [{call.name}] description after running for the first time in this session.")
                                            output += [
                                                {
                                                    "call": {"name": call.name, "args": call.args},
                                                    "signature": await describe_tools([call.name]),
                                                    "output": tool_output
                                                }
                                            ]
                                    else:
                                        # Get here if LLM gave broken args
                                        log_internal_event(f"[OPTIMUS] Tool call [{call.name}] violates signature")
                                        output += [
                                            {
                                                "call": {"name": call.name, "args": call.args},
                                                "error": f"Error: wrong arguments for tool {call.name}. Either you are calling the wrong tool or you are providing the wrong arguments.",
                                                # "signature": describe_tools([call.name]),
                                            }
                                        ]
                                else:
                                    # Get here if LLM gave broken tool name
                                    output += [
                                        {
                                            "call": {"name": call.name, "args": call.args},
                                            "error": f"Error, tool name {call.name} does not exist."
                                        } | {"available_tools": await list_tools()} if not list_flag else {}
                                    ]
                                    list_flag = True
                            except Exception as e:
                                # Land any unexpected exceptions here for the LLM to observe
                                log_error(f"[OPTIMUS] Critical dynamic tool call error: {e}")
                                output += [
                                    {
                                        "call": {"name": call.name, "args": call.args},
                                        "signature": await describe_tools([call.name]) if call.name in self.state.discovered_tools else f"See previous {call.name} description.",
                                        "error": e
                                    }
                                ]

                        log_internal_event(f"[OPTIMUS] Current trajectory: {self.state.tasks[self.state.parent.title]["trajectory"]}")
                        # LLM observes good calls and bad calls
                        output += [{"reminder": "Always follow up "}]
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
        except* Exception as eg:
            if isinstance(eg, BaseExceptionGroup):
                for e in eg.exceptions:
                    # logger.exception("real error", exc_info=e)
                    log_error(f"Optimus critical error: {e}")
            else:
                log_error(f"Optimus critical error: {eg}")
            # signal failing tasks
            for title,task in self.state.tasks.items():
                if task["state"] == "awake":
                    task["state"] = "failing"
            raise e

    def prune(self, task: str):
        """Remove a root task"""
        if task not in list(self.state.tasks.keys()):
            log_internal_event(f"[OPTIMUS] Tried to delete non-existing task [{task}]")
            return f"Task [{task}] does not exist."
        else:
            # Remove all references to task as a dependency
            for _,v in self.state.tasks.items():
                v["dependencies"] = [dep for dep in v["dependencies"] if dep != task]
            # Remove task
            del self.state.tasks[task]
            log_internal_event(f"[OPTIMUS] Removed all references to [{task}]")
            return

    def graft(self, task: str, deps: list[str], tip: str = None):
        """Reconnect task with new dependencies"""
        if task not in list(self.state.tasks.keys()):
            return f"Task [{task}] does not exist."
        elif any(dep not in list(self.state.tasks.keys()) for dep in deps):
            return f"At least one dependency in {deps} does not exist."
        else:
            log_internal_event(f"[OPTIMUS] Grafting [{task}] with new dependencies: {deps}")
            self.state.tasks[task]["dependencies"] = deps

            # Add tip (will be consumed at the next run of the task)
            if tip is not None:
                self.state.tasks[task]["tips"] += [tip]
                self.state.tasks[task]["tier"] = 0
            return

    def plant(self, task: str, deps: list[str]):
        """Create a new task, or update its dependencies"""
        # Ensure sane defaults for task
        if not self.state.tasks.get(task):
            self.state.tasks[task] = {
                "task": task,
                "kind": "root",
                "state": "sleeping",
                "tier": 1,  # starts at tier one to avoid refining nothing
                "ttl": 60,  # 60 minutes by default
                "tips": [],
                "qna": [],
                "trajectory": []
            }
            self.graft(task, deps)
            return
        else:
            return f"Task [{task}] already exists"

    async def stop(self):
        """Stop a running job"""
        if not self.state.job.done():
            self.state.job.cancel()
            # Let it observe CancelledError and run its own cleanup
            await asyncio.gather(self.state.job, return_exceptions=True)
            self.state.job = None
            self.state.situation = {}
            self.state.tokens = asyncio.Queue()
            for title,task in self.state.tasks.items():
                if task.get("state") == "awake":
                    task["state"] = "sleeping"
            log_internal_event("[OPTIMUS] Job cancelled.")
