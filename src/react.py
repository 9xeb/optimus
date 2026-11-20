# Reimplementation of a dspy.ReAct because the new one in 3.5.0:
# 1. is structurally unfeasible for smaller models
# 2. hard to streamify
# 3. kind of broke everything nice thing I had built around the previous one

# All the good stuff moved here: full control over dynamic information between steps, tool discovery, soft recoveries (no agent restarts)

import json
import dspy
from dataclasses import asdict
from src.utils import State, ToolCall, update_action_metrics, require_approval
from src.log import log_internal_event, log_error

class OAgent(dspy.Module):
    """Optimus-native agent, similar to Old-style (3.4) ReAct loop, but with full control"""
    def __init__(self, state: State, tools: list[dspy.Tool], discovery: bool = True, max_iters: int = 10):
        self.state = state
        self.tools = tools
        self.metatools = [
            dspy.Tool(self.list_tools),
            dspy.Tool(self.describe_tools),
            dspy.Tool(self.run_tools),
            dspy.Tool(self.finish)
        ] if discovery else self.tools
        self.working_memory = []
        self.dataset = []

        self.action_signature = "working_memory, tools, task -> tool_call: ToolCall"
        self.act = dspy.Predict(self.action_signature)

    async def aforward(self, task, **kwargs):
        """Agent loop"""
        output = dspy.Prediction(answer="")
        try:
            while True:
                # Choose one tool
                log_internal_event(f"[OAgent] Choosing a tool among [{[tool.name for tool in self.metatools]}]")
                action = await self.act.acall(
                    working_memory=self.working_memory,
                    # tools=self.tools + [self.finish],
                    tools=[
                        {"name": tool.name, "desc": tool.desc, "args": tool.args}
                        for tool in self.metatools
                    ],
                    task=task
                )
                # log_error(f"[OAgent] I chose: {action.tool_call}")
                call = action.tool_call
                # Find matching tool
                tool_output = None
                for tool in self.metatools:
                    log_internal_event(f"[OAgent] Matching [{call.name}] with [{tool.name}]")
                    if tool.name == call.name:
                        tool_output = await tool.acall(**call.args)
                if not tool_output:
                    log_error(f"[OAgent] Tool [{call.name}] not found, letting agent observe the mistake.")
                    self.working_memory += [f"You must wrap `{call.name}` with `run_tools`."]
                else:
                    log_internal_event(f"[OAgent] Saving [{call.name}] output to working memory")
                    self.working_memory += [asdict(call) | {"output": tool_output}]
                if call.name == "finish":
                    output = dspy.Prediction(answer=tool_output)
                    break
        except Exception as e:
            log_error(f"[OAgent] Critical error: {e}")
        return output


    def finish(self, answer: str, tips: list[str] = None):
        """
        Confirm the completion of the task.

        Args:
            answer: the answer to the task
            tips: if any, some tips to avoid repeating mistakes from in this run, in future iterations of this task.
        """
        if tips is not None:
            self.dataset += [tips]
        return answer
        
    async def list_tools(self):
        """Get the list of available tools."""
        update_action_metrics(state=self.state, action_count=1)
        list_of_tools = [
            {
                "name": tool.name,
                "args": list(tool.args.keys())
            }
            for tool in self.tools
        ] + [{"reminder": "Remember to wrap tools around `run_tools`."}]
        log_internal_event(f"[OPTIMUS] Listing tools: {list_of_tools}")
        await self.state.actions.put({"task": self.state.current.title, "data": "Listing tools..."})
        return list_of_tools
    async def describe_tools(self, names: list[str]):
        """Get descriptions of the most promising tools, by tool name."""
        update_action_metrics(state=self.state, action_count=1)
        match = [tool for tool in self.tools if tool.name in names]
        log_internal_event(f"[OPTIMUS] Discovered tools: {match}")
        # await self.state.actions.put({"task": self.state.current.title, "data": "Describing tools..."})
        self.state.discovered_tools += names
        return match
    async def run_tools(self, calls: list[ToolCall]):
        """
        Run a list of tools by name.
        When iterating, submit the single largest list of calls possible.
        """
        update_action_metrics(state=self.state, action_count=len(calls))
        # Split as many tool calls as possible in multiple items in the call list.
        # Split repetitive or looping operations into multiple items in the calls list.
        list_flag = False   # track if tools already listed to avoid spam in the same batch
        output = []
        for call in calls:
            log_internal_event(f"[OPTIMUS] Trying to call tool {call}")
            log_internal_event(f"[OPTIMUS] Discovered tools: {self.state.discovered_tools}")
            try:
                # 1. Find matching tools by name (and their approval-gated counterparts)
                match = [tool for tool in self.tools if tool.name == call.name]
                # gated_match = [tool for tool in gated_mcp_tools if tool.name == call.name]
                if len(match) > 0:
                    # Run the first matching tool
                    # Check if all args exist in signature
                    if all([arg in list(match[0].args.keys()) for arg in list(call.args.keys())]):
                        # Stream call goal
                        await self.state.actions.put({"task": self.state.current.title, "data": call.rationale})
                        if not call.risky:
                            # Direct call if read only and previously reviewed task
                            log_internal_event(f"[OPTIMUS] Tool call [{call.name}] advertised as read-only.")
                            # Await calls so it works both for async and sync
                            tool_output = await match[0].acall(**call.args)
                            # Add to trajectory without approval
                            # self.state.trajectory += [call.rationale]
                            self.state.tasks[self.state.parent.title]["trajectory"] += [call.rationale]
                            self.state.tasks[self.state.current.title]["trajectory"] += [call.rationale]
                        else:
                            # Wait for user approval
                            log_internal_event(f"[OPTIMUS] Tool call [{call.name}] advertised as a change. Requiring approval.")
                            self.state.justification = call.rationale  # register the justification for user interface
                            gated_tool = dspy.Tool(
                                require_approval(self.state, match[0], match[0].name),
                                name=match[0].name,
                                desc=match[0].desc,
                                args=match[0].args,
                                arg_types=match[0].arg_types
                            )
                            tool_output = await gated_tool.acall(**call.args)
                            if not tool_output:
                                return f"Tool call denied by user. Reason: {self.state.resolution}"
                            else:
                                self.state.tasks[self.state.parent.title]["trajectory"] += [call.rationale]
                                self.state.tasks[self.state.current.title]["trajectory"] += [call.rationale]
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
                                    "signature": await self.describe_tools([call.name]),
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
                        } | {"available_tools": await self.list_tools()} if not list_flag else {}
                    ]
                    list_flag = True
            except Exception as e:
                # Land any unexpected exceptions here for the LLM to observe
                log_error(f"[OPTIMUS] Critical dynamic tool call error: {e}")
                output += [
                    {
                        "call": {"name": call.name, "args": call.args},
                        "signature": await self.describe_tools([call.name]) if call.name in self.state.discovered_tools else f"See previous {call.name} description.",
                        "error": e
                    }
                ]
        log_internal_event(f"[OPTIMUS] Current trajectory: {self.state.tasks[self.state.parent.title]["trajectory"]}")
        # LLM observes good calls and bad calls
        return output
