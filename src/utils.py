import asyncio
from dataclasses import dataclass, field
import numpy
import dspy
import json
from colorama import Fore, Style
# from json_repair import repair_json
from anytree import Node, RenderTree

from src.log import log_internal_event, log_error

# This simplified view goes to the LLM
@dataclass
class Task:
    """A payload containing a task and its dependencies"""
    title: str = field(metadata={"description": "Title of the task. Must include the name of the entities involved, to avoid ambiguities with other, similar tasks applied to other entities."})
    task: str
    # kind: str = field(metadata={"description": "The kind of task between: simple, complex. Use simple when the task may not be broken down any further."})
    # kind: str = field(metadata={"description": "The kind of task between: simple, complex. Use simple when the task never breaks down into more tasks."})
    # kind: str = field(metadata={"description": "The kind of task between: simple, complex. Use simple if the task could be done in just a few steps."})
    kind: str = field(metadata={"description": "The kind of task. Only kind 'simple' is allowed."})
    ttl: int = field(metadata={"description": "How long a report from this task should be considered valid, in minutes."})
    # volatility: str = field(metadata={"description": "How often the results of the task may change in the future, between: low, medium, high"})
    # domain: str = field(metadata={"description": "Where the task can be performed between: browser, shell, ide"})

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
    # log_internal_event("Beginning streaming")
    try:
        prediction = ""
        async for chunk in streamer(**kwargs):          # pass signature values to module inside streamer
            if isinstance(chunk, dspy.streaming.StreamResponse):
                # when multi-output, chunk.signature_field_name is the name of the output that the chunk belongs to
                # print(f"{chunk.chunk}", end="", flush=True)
                for queue in queues:
                    pretty_chunk = prettify_chunk(chunk)
                    print(pretty_chunk, end="", flush=True)
                    await queue.put({"task": kwargs["task"]["title"], "data": chunk})
                    # await asyncio.sleep(0)
                # yield f"{chunk.chunk}"
                # await self.message_queue.put(chunk.chunk)
            elif isinstance(chunk, dspy.Prediction):      # the last chunk with final response is a Prediction
                prediction = chunk
    except* AttributeError as e:
        # except* unwraps any exceptions raised under BaseExceptionGroup, which happens with dspy.RLM
        log_error(f"Streaming error: {e}")
        pass    # skip current bug where some dspy.RLM predictors have broken chunks
    return prediction

def render_optimus_tree(root: Node) -> str:
    """
    Render and anytree tree starting from a root optimus node (a.k.a. contains a prediction key).

    Args:
        root: the anytree Node instance to start from
    
    Return:
        a rendering of the tree
    """
    try:
        print("################# RENDER ##############")
        render = ""
        for pre, _, node in RenderTree(root):
            payload = {
                "task_id": node.id,
                "task": node.name,
                # "completed": node.prediction is not None
                # "response": node.prediction.report if node.prediction is not None else "Not yet available."
            }
            print(f"{pre}{payload}")
            render += f"{pre}{payload}\n"
        print("#######################################")
    except Exception as e:
        log_error(f"{e}")
    return render

def require_approval(state: State, func, name):
    """Simple wrapper for human approval of tool call"""
    async def wrapped(**kwargs):
        state.situation = {
            "type": "tool-approval",
            "parent": state.focus.title,
            "tool": name,
            "args": kwargs
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
            # raise PermissionError(f"Tool '{name}' call denied: {kwargs}")
        # print("\n############# APPROVAL REQUIRED ##############")
        # if input(f"{name} {json.dumps(kwargs, indent=4)} ? (y/N) > ").lower() in ['y', 'yes']:
        #     return func(**kwargs)
        # else:
        #     raise PermissionError(f"Tool '{name}' call denied: {kwargs}")
    return wrapped
    
# def count_tokens(new_messages):
#     """
#     Return estimated token count from a list of Agent messages

#     Args:
#         new_messages: a list of ModelMessage
    
#     Returns:
#         an integer with the token count
#     """
#     estimate = _estimate_usage(new_messages)
#     return estimate.input_tokens + estimate.output_tokens

# def extract_execution_path(new_messages):
#     """
#     Extract tool call execution path in pretty format

#     Args:
#         new_messages: list of ModelMessage

#     Returns:
#         a structured execution path 
#     """
#     execution_path = {}
#     for message in new_messages:
#         for part in message.parts:
#             if part.part_kind == "tool-call":
#                 execution_path[part.tool_call_id] = {"tool": part.tool_name, "args": part.args}
#             elif part.part_kind == "tool-return":
#                 execution_path[part.tool_call_id] = execution_path[part.tool_call_id] | {"status": part.content}
#     return execution_path

# def initialize_openai_client(model_string: str, temperature: float = 0.0):
#     """
#     Translate model strings into the appropriate Pydantic AI Model, while preserving the original model string for GEPA
#     """
#     # Extract provider and model name
#     model_provider = model_string.split('/')[0]
#     model_name = model_string.split('/', 1)[1]
#     # log_internal_event(f"Initializing model: {model_provider}/{model_name}")
#     if model_provider == "openai":
#         # Override openai provider to support custom endpoints
#         provider = OpenAIProvider(
#             base_url=os.environ["OPENAI_API_BASE"],
#             api_key=os.environ["OPENAI_API_KEY"]
#         )
#         # Construct client with exceptions to manage reasoning effort for models that support it
#         try:
#             client = OpenAIChatModel(
#                 model_name=model_name,
#                 provider=provider,
#                 # control model parameters
#                 settings=OpenAIChatModelSettings(
#                     openai_reasoning_effort="low",
#                     temperature=temperature,
#                     frequency_penalty=1.5,       # discourage repetitive tokens
#                     # max_tokens=self.context_size
#                 ),
#             )
#         except Exception:
#             log_internal_event("OpenAI Chat Model - Falling back to safe non-thinking settings")
#             client = OpenAIChatModel(
#                 model_name=model_name,
#                 provider=provider,
#                 # control model parameters
#                 settings=OpenAIChatModelSettings(
#                     # openai_reasoning_effort="low",
#                     temperature=temperature,
#                     frequency_penalty=1.5,       # discourage repetitive tokens
#                     # max_tokens=self.context_size
#                 ),
#             )
#     else:
#         raise ValueError(f"Unsupported model provider: {model_provider}")
#     return client

# def extract_first_json(text):
#     """
#     Extract the first available JSON in a free form string

#     Args:
#         text: free form string

#     Returns
#         Extracted JSON
#     """
#     # Extract between the first '{' and the last '}'
#     first_bracket = "{"+"{".join(text.split('{')[1:])
#     last_bracket = "}".join(first_bracket.split('}')[:-1])+"}"
#     return last_bracket

# def stabilize_json(unstable_string: str, expected_keys: list = None) -> dict:
#     """
#     Stabilize a given input JSON, usually from an LLM output.
#     Several methods coming from json and json5 modules are tried until one works.
#     If all fails, one exception of either AssertionError, IndexError or TypeError is raised
#     Args:
#         unstable_string: input string containing a JSON string to stablize
#         expected_keys: list of expected keys the final dictionary must have
#     Raises
#         AssertionError if one of the expected keys is not present in the final dictionary
#         IndexError if string splitting fails to find a JSON inside curly brackets
#         TypeError if both json5 and json fail to parse
#     Returns:
#         The final stable dictionary
#     """
#     # Grab the first JSON that is found in the string
#     stable_json = extract_first_json(unstable_string)
#     repaired_json = repair_json(stable_json)
#     try:
#         stable_dict = json5.loads(stable_json)
#     except ValueError:
#         stable_dict = json.loads(stable_json)
#     # Ensure all expected_keys are present
#     for key in expected_keys:
#         assert key in stable_dict.keys()
#     return stable_dict
