import asyncio
from dataclasses import dataclass, field
import numpy
import dspy
import json
from colorama import Fore, Style

from src.log import log_internal_event, log_error

@dataclass
class Task:
    """A payload containing a task and its dependencies"""
    title: str = field(metadata={"description": "Title of the task. Must include the name of the entities involved, to avoid ambiguities with other, similar tasks applied to other entities."})
    task: str
    kind: str = field(metadata={"description": "The kind of task. Only kind 'simple' is allowed."})
    ttl: int = field(metadata={"description": "How long a report from this task should be considered valid, in minutes."})

    # dependencies field allows the LLM to write coherent architectures
    dependencies: list[str] = field(metadata={"description": "Titles of previous tasks that this task would benefit from knowing about."})
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class State:
    """The state of an Optimus instance"""
    filepath: str   # where to persist memory
    tasks: list[Task]
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
    job: asyncio.Task

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
        prediction = ""
        async for chunk in streamer(**kwargs):          # pass signature values to module inside streamer
            if isinstance(chunk, dspy.streaming.StreamResponse):
                # when multi-output, chunk.signature_field_name is the name of the output that the chunk belongs to
                for queue in queues:
                    pretty_chunk = prettify_chunk(chunk)
                    print(pretty_chunk, end="", flush=True)
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
            "justification": state.justification
        }
        log_internal_event(f"[SITUATION] {state.situation}")
        await state.event.wait()
        state.situation = {}
        state.event.clear()
        if not state.resolution:    # No feedback from user
            return await func(**kwargs)
        else:
            log_error(f"Tool '{name}' call denied: {kwargs}. Reason: {state.resolution}")
            return f"Tool call denied by user. Reason: {state.resolution}"
    return wrapped
