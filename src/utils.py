import asyncio
from dataclasses import dataclass, field
import numpy
import dspy
import json
import time
from colorama import Fore, Style

from src.log import log_internal_event, log_error

@dataclass
class Task:
    """A payload containing a task and its dependencies"""
    title: str = field(metadata={"description": "Title of the task. Must include the name of the entities involved, to avoid ambiguities with other, similar tasks applied to other entities."})
    task: str
    kind: str = field(metadata={"description": "The kind of task. Only kind 'simple' is allowed."})
    ttl: int = field(metadata={"description": "How long a report from this task should be considered valid, in minutes."})

    # # # # # dependencies field allows the LLM to write coherent architectures
    # # # # dependencies: list[str] = field(metadata={"description": "Titles of previous tasks that this task would benefit from knowing about."})
    # # # # def __post_init__(self):
    # # # #     if self.dependencies is None:
    # # # #         self.dependencies = []

@dataclass
class Metrics:
    """Model metrics"""
    last_action_time: float = None
    actions: int = 0
    apm: float = 0
    # prompt_tokens: int = 0
    # completion_tokens: int = 0
    # prompt_tps: int = 0
    # completion_tps: int = 0
@dataclass
class State:
    """The state of an Optimus instance"""
    filepath: str   # where to persist memory
    tasks: dict[str, Task]
    qna: dict
    parent: str  # the parent task
    current: str # the current task
    discovered_tools: list[str] # used to track discovered tools in a context
    approval: bool      # whether to ask for tool and agents approval
    justification: str  # the current justification for the situation
    situation: dict
    resolution: str
    event: asyncio.Event
    tokens: asyncio.Queue
    actions: asyncio.Queue
    trajectory: list
    job: asyncio.Task
    error: str          # last error
    metrics: Metrics    # track RCS metrics

# # # # class RecoveringReAct(dspy.ReAct):  
# # # #     def __init__(self, signature, tools, max_iters=20):  
# # # #         super().__init__(signature, tools, max_iters)  
  
# # # #         class _RecoveringPredict(dspy.Predict):  
# # # #             def __init__(self, inner):  
# # # #                 super().__init__(inner.signature)  
# # # #                 # Adopt inner's state wholesale: lm, demos, config,  
# # # #                 # fields, stage, traces, ...  
# # # #                 self.__dict__.update(inner.__dict__)  
  
# # # #             def _recover(self, err):  
# # # #                 return dspy.Prediction(  
# # # #                     next_thought=f"Output failed to parse: {err}",  
# # # #                     tool_calls=dspy.ToolCalls.from_dict_list(  
# # # #                         [{"name": "__format_error__", "args": {}}]  
# # # #                     ),  
# # # #                 )  
  
# # # #             def forward(self, **kwargs):  
# # # #                 try:  
# # # #                     return super().forward(**kwargs)  
# # # #                 except (AdapterParseError, ValueError) as err:  
# # # #                     return self._recover(err)  
  
# # # #             async def aforward(self, **kwargs):  
# # # #                 try:  
# # # #                     return await super().aforward(**kwargs)  
# # # #                 except (AdapterParseError, ValueError) as err:  
# # # #                     return self._recover(err)  
  
# # # #         self.react = _RecoveringPredict(self.react)

def update_action_metrics(state: State, action_count: int):
    # Update action metrics
    state.metrics.actions += action_count
    if state.metrics.last_action_time is not None:
        # avg between current apm and new apm
        state.metrics.apm = (
            state.metrics.apm + (
               action_count / (time.time() - state.metrics.last_action_time)
            )
        ) / 2.0
    state.metrics.last_action_time = time.time()


def prettify_chunk(chunk: dspy.streaming.StreamResponse):
    """Wrap a chunk with colors"""
    if chunk.signature_field_name == "next_thought":
        return Fore.GREEN + chunk.chunk
    elif "tool" in chunk.signature_field_name:
        return Fore.YELLOW + chunk.chunk
    else:
        return Style.RESET_ALL + chunk.chunk

async def stream_dspy_program(queues: list[asyncio.Queue], module: dspy.Module, **kwargs):
    """
    Output streaming util for DSPy programs.
    Args:
        module: DSPy module to run
        **kwargs: list of kwargs corresponding to the signature inputs
    
    Return:
        a Prediction object containing the signature outputs
    """
    # Wrap module around streamer
    streamer = dspy.streamify(
        program=module,
        is_async_program=True,
        stream_listeners=[
            dspy.streaming.StreamListener(
                signature_field_name=output_field_key,
                predict=predictor,
                predict_name=name,
                allow_reuse=True,   # ReAct calls its predictor repeatedly
            )
            for name, predictor in module.named_predictors()
            for output_field_key in predictor.signature.output_fields.keys()
        ]
    )
    # Run program=module+arguments and stream output fields
    try:
        prediction = dspy.Prediction(answer="")
        async for chunk in streamer(**kwargs):          # pass signature values to module inside streamer
            if isinstance(chunk, dspy.streaming.StreamResponse):
                # when multi-output, chunk.signature_field_name is the name of the output that the chunk belongs to
                pretty_chunk = prettify_chunk(chunk)
                print(pretty_chunk, end="", flush=True)
                # print(pretty_chunk, flush=True)
                # print(type(chunk).__name__, time.monotonic(), flush=True)
                for queue in queues:
                    await queue.put({"task": kwargs["task"]["title"], "data": chunk})
            elif isinstance(chunk, dspy.Prediction):      # the last chunk with final response is a Prediction
                prediction = chunk
    except* AttributeError as e:
        # except* unwraps any exceptions raised under BaseExceptionGroup, which happens with dspy.RLM
        log_error(f"Streaming error: {e}")
        pass    # skip current bug where some dspy.RLM predictors have broken chunks
    return prediction

def require_approval(state: State, func, name):
    """Simple wrapper for human approval of tool call"""
    async def wrapped(**kwargs):
        state.situation = {
            "type": "tool-approval",
            "parent": state.current.title,
            "tool": name,
            "args": kwargs,
            "justification": state.justification,
            # "trajectory": state.trajectory          # trajectory so far
            "trajectory": state.tasks[state.parent.title]["trajectory"]
        }
        log_internal_event(f"[SITUATION] {state.situation}")
        await state.event.wait()
        state.situation = {}
        state.event.clear()
        if not state.resolution:    # No feedback from user = approved
            # state.trajectory += [state.justification]
            state.tasks[state.parent.title]["trajectory"] += [state.justification]
            state.tasks[state.current.title]["trajectory"] += [state.justification]
            return await func(**kwargs)
        else:
            log_error(f"Tool '{name}' call denied: {kwargs}. Reason: {state.resolution}")
            return f"Tool call denied by user. Reason: {state.resolution}"
    return wrapped
