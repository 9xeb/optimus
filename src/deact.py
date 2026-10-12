# Decide + Act = DeAct
# Uses decision model to lift as much as possible away from LLM
# Cheaper to train if it misbehaves.

import base64
import requests
import dspy
from dspy.experimental import Noul, Choice, Score

from src.utils import stream_dspy_program
from src.log import log_error, log_internal_event
from src.decide import Choice, Choose

class DeAct(dspy.Module):
    """
    DeAct: Decide and Act. Brings decision models to the agentic loop.
    Calls LLMs as little as possible, currently for filling tool call arguments.
    """
    def __init__(self, tools: list[dspy.Tool] = None):
        self.external_tools = tools if tools is not None else []
        self.external_tools += [
            dspy.Tool(
                self.finish,
                name="finish",
                desc="Signal that the task has been completed"
            )
        ]
        self.named_external_tools = {
            tool.name: tool
            for tool in self.external_tools
        }

        self.tool_choice = Choice(
            name="tool",
            instructions="Which tool should be called next?",
            criteria={
                tool.name: tool.desc
                for tool in self.external_tools
            }
        )

        # tool_choices = [(tool.name, tool.desc) for tool in self.external_tools]
        # log_internal_event(f"{tool_choices}")
        # Toolcall = Choice[tuple(tool_choices)]
        # class DecideTool(dspy.Signature):
        #     """Given the task and the history, decide the best next tool to call."""
        #     task: str = dspy.InputField(desc="The task to complete")
        #     history: str = dspy.InputField(desc="The history so far.")

        #     # Available tools are embedded in output field
        #     tool_call: Toolcall = dspy.OutputField(desc="Which tool should be called next?")

        # self.decide = dspy.Predict(DecideTool)

        self.decide = Choose()
        self.fill = dspy.ChainOfThought("task, history, chosen_tool -> tool_args")
        self.report = dspy.ChainOfThought("task, history -> report")

        self.finish_flag = False

    def forward(self, task, **kwargs):
        """
        Combine decision model and LLM for an efficient agent loop
        """
        self.finish_flag = False
        history = []
        while not self.finish_flag:
            log_internal_event(f"Deciding with history: {history}")
            decision = self.decide(
                state={"task": task, "history": history},
                choices=[self.tool_choice]
            )
            # decision = self.decide(
            #     task=task,
            #     history=history,
            # )
            tool_output = self.route_tool(task=task, history=history, tool_name=decision.tool_call)
            history += [
                {"kind": "tool_call", "name": decision.tool_call, "output": tool_output}
            ]
        return self.report(task=task, history=history).report

    async def aforward(self, task, **kwargs):
        """
        Combine decision model and LLM for an efficient agent loop
        """
        self.finish_flag = False
        history = []
        while not self.finish_flag:
            decision = await stream_dspy_program(
                self.decide,
                task=task,
                history=history,
            )
            tool_output = self.route_tool(task=task, history=history, tool_name=decision.tool_call)
            history += [
                {"kind": "tool_call", "name": decision.tool_call, "output": tool_output}
            ]
        return await stream_dspy_program(self.report, task=task, history=history).report

    def route_tool(self, task: str, history: list, tool_name: str):
        log_internal_event(f"Decision is asking for tool {tool_name}.")
        tool = self.named_external_tools.get(tool_name)
        if tool is None:
            # Tool does not exist
            log_error(f"Tool {tool_name} does not exist.")
            return f"Tool {tool_name} does not exist."
        if tool.args is None or len(tool.args.items()) == 0:
            # Tool exists and expects no arguments
            log_internal_event(f"Tool {tool_name} exists and is parameter-less.")
            tool_output = tool()
        else:
            # Tool exists and expects some arguments (to be filled with LLM)
            log_internal_event(f"Tool {tool_name} exists and parameters must be filled.")
            tool_not_working = True
            while tool_not_working:
                tool_args = self.fill(task=task, history=history, chosen_tool=tool).tool_args
                try:
                    tool_output = tool(**tool_args)
                    tool_not_working = False
                except Exception as e:
                    log_error(f"Tool Call Error: {e}")
                    history += [
                        {"kind": "tool_call_error", "name": tool_name, "error": e}
                    ]

        log_internal_event(f"Tool {tool_name} output is: {tool_output}")
        return tool_output

    def finish(self):
        """Signal that the task has been completed"""
        log_internal_event(f"Decision is asking to finish.")
        self.finish_flag = True
        return True
