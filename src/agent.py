import hashlib
import json
import pickle
import os
from typing import Any, Optional, Self
from collections.abc import Callable
from dataclasses import dataclass

from pydantic_ai import Agent, UsageLimits, ModelMessage, RunContext, ToolDefinition, Tool, SkipToolExecution
from pydantic_ai.capabilities import Hooks, Thinking
from pydantic_ai._utils import disable_threads
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.mcp import load_mcp_servers

from src.utils import initialize_openai_client
from src.log import log_tool_request, log_error, log_internal_event, log_request, log_part, log_tool_response

@dataclass
class AgentDeps:
    log_prefix: str
    verbose: bool
    # tool approval function takes (ctx, tool_name, args) -> bool. Defaults to returning always true
    tool_approval_function: Optional[Callable[[RunContext[Self], str, dict[str, Any]], bool]] = None
    # lambda tool_name, args: True

async def default_tool_approval_function(ctx: RunContext[AgentDeps], tool_name: str, args: dict[str, Any]) -> bool:
    """
    Simple tool approval via stdio
    """
    log_tool_request(f"{ctx.deps.log_prefix} - REQUESTING HUMAN APPROVAL - {tool_name}({args})")
    try:
        response = input("Approve or refuse? [y/N] ")
        return response.lower() == "y" or response.lower() == "yes"
    except Exception as e:
        log_error(f"{e}")
        log_tool_request(f"{ctx.deps.log_prefix} - HUMAN UNREACHABLE - TOOL REFUSED - {tool_name}({args})")
        return False

hooks = Hooks()     # Initialize hooks container
# Register tool execution logging hooks  
@hooks.on.before_tool_execute  
async def ensure_tool_approval(  
    ctx: RunContext[AgentDeps],   
    *,   
    call: ToolCallPart,   
    tool_def: ToolDefinition,   
    args: dict[str, Any]  
) -> dict[str, Any]:
    try:
        if not tool_def.metadata.get("read_only"):
            log_internal_event(f"{ctx.deps.log_prefix} - TOOL APPROVAL REQUIRED - {call.tool_name}({args})")
            # Ask for human confirmation using the provided approval function
            user_approved = await ctx.deps.tool_approval_function(ctx, call.tool_name, args)
        else:
            # Tool approval not required, approving by default
            user_approved = True
            # log_internal_event(f"{ctx.deps.log_prefix} - TOOL APPROVAL NOT REQUIRED - {call.tool_name}({args})")
    except Exception as e:
        log_error(f"{ctx.deps.log_prefix} - before_tool_execute hook error: {e}")
        user_approved = False
    
    if user_approved:
        log_tool_request(f"{ctx.deps.log_prefix} - TOOL CALL - {call.tool_name}({args})")
    #     log_internal_event(f"{ctx.deps.log_prefix} - TOOL APPROVED - {call.tool_name}({args})")
    if not user_approved:
        log_error(f"{ctx.deps.log_prefix} - TOOL REFUSED - {call.tool_name}({args})")
        # Skip execution and return a custom message  
        raise SkipToolExecution(result=f"Tool {call.tool_name} was not approved")
    return args

@hooks.on.after_tool_execute  
async def log_tool_result(  
    ctx: RunContext[AgentDeps],  
    *,  
    call: ToolCallPart,  
    tool_def: ToolDefinition,  
    args: dict[str, Any],  
    result: Any,  
) -> Any:
    try:  
        # Your logging logic here
        # if ctx.deps.verbose:
        print_output_limit = 200
        # agent_output_limit = 5000
        # truncated_result = f"{result[:agent_output_limit+1]} {"[... truncated due to exceeding output size limits. Try filtering or grepping.]" if len(result) > agent_output_limit else ""}"
        log_tool_response(f'{ctx.deps.log_prefix}{call.tool_name}\n{result[:print_output_limit]}{" [... truncated]" if len(result) > print_output_limit else ""}')
    except Exception as e:  
        # Don't let exceptions propagate - they cause retries  
        log_error(f"{ctx.deps.log_prefix} - after_tool_execute hook error: {e}")  
    # agent_output_limit = 5000
    # truncated_result = f"{result[:agent_output_limit+1]} {"[... truncated due to exceeding output size limits. Try filtering or grepping.]" if len(result) > agent_output_limit else ""}"
    return result

# Model request logging  
@hooks.on.before_model_request  
async def log_before_request(ctx: RunContext[AgentDeps], request_context):
    try:
        if ctx.deps.verbose:
            history_size = len(request_context.messages)
            # ctx.tool_manager.tools provides deep info about tools and available arguments
            tools = [key for key in ctx.tool_manager.tools.keys()]
            # log_request(f"- HOOK - {request_context.messages[0].parts[0].content}")
            log_payload = {
                "content": request_context.messages[0].parts[0].content,
                "history_size": history_size,
                "context_messages": request_context.messages,
                "tools": tools
            }
            log_request(f"{ctx.deps.log_prefix} - {log_payload}")
        # else:
        #     log_request(f"{ctx.deps.log_prefix}")
    except Exception as e:
        log_error(f"{ctx.deps.log_prefix} - before_model_request hook error: {e}")
    return request_context

@hooks.on.after_model_request  
async def log_after_response(ctx: RunContext[AgentDeps], *, request_context, response):
    try:
        if ctx.deps.verbose:
            # Response can include tool calls, so we log them here instead of @hooks.on.before_tool_call
            for response_part in response.parts:
                log_part(
                    prefix=ctx.deps.log_prefix,
                    part=response_part
                )
        # else:
        #     # Only log the last message (if present)
        #     if len(response.parts) > 0:
        #         log_part(
        #             prefix=ctx.deps.log_prefix,
        #             part=response.parts[-1]
        #         )
    except Exception as e:
        log_error(f"{ctx.deps.log_prefix} - after_model_request hook error: {e}")
    return response

# Error handling
@hooks.on.tool_execute_error
async def log_tool_error(ctx: RunContext[AgentDeps], *, call, tool_def, args, error):
    try:
        log_error(f"{ctx.deps.log_prefix} - Tool {call.tool_name} failed: {error} (automatic retry)")
    except Exception as e:
        log_error(f"{ctx.deps.log_prefix} - tool_execute_error hook error: {e}")
    raise error  # Re-raise to maintain normal retry flow


class AgentWrapper:
    def __init__(self, lm: str, thinking: bool = True, debug: bool = False):
        self.client = initialize_openai_client(lm)
        self.thinking = thinking
        self.deps = AgentDeps(
            log_prefix="OPTIMUS -",
            verbose=debug,
            tool_approval_function=default_tool_approval_function
        )

        self.mcp_path = os.environ["HOME"]+'/'+".optimus/mcp.json"
        self.mcp_toolsets = load_mcp_servers(self.mcp_path) if os.path.exists(self.mcp_path) else []
        # self.mcp_toolsets = load_mcp_servers(os.environ["MCP_CONFIG"]) if os.environ.get("MCP_CONFIG") else []

    async def async_step(
        self,
        task: str,
        response_format: str,
        user_prompt: str,
        tools: list[Tool] = None,
        history: list[ModelMessage] = None,
        stream: bool = False
    ):
        """
        Args:
            prefix: a prefix to add when logging

        Returns:
            current_history: the new messages generated right now
            result_output: the last message in the new messages
        """

        tools = tools if tools is not None else []
        # Loop through prompts until it stops calling tools and gives a final answer

        agent = Agent(
            model=self.client,
            system_prompt=task,
            tools=tools,
            toolsets=self.mcp_toolsets,     # Additional tools provided by mcp.json
            capabilities=[
                hooks,
                Thinking(effort=self.thinking)
            ],
            deps_type=AgentDeps
        )

        try:
            async with agent.run_stream(
                user_prompt=f"""
                {user_prompt}
                {response_format}
                """,
                message_history=history,
                deps=self.deps,
                usage_limits=UsageLimits(tool_calls_limit=20)
            ) as result:
                if stream:
                    async for text in result.stream_text():
                        print(text)
                # stream finished here, collect results
                output = await result.get_output()
                return result.new_messages(), output
        except ModelHTTPError as e:
            # Fall back to empty response
            log_error(f"Model HTTP Error: {e}")
            return [], ""


    def step(
        self,
        task: str,
        response_format: str,
        user_prompt: str,
        tools: list[Tool] = None,
        history: list[ModelMessage] = None,
    ):
        """
        Args:
            prefix: a prefix to add when logging

        Returns:
            current_history: the new messages generated right now
            result_output: the last message in the new messages
        """

        tools = tools if tools is not None else []
        # log_error(f"Available tools: {tools}")
        # Loop through prompts until it stops calling tools and gives a final answer
        agent = Agent(
            model=self.client,
            system_prompt=task,
            tools=tools,
            toolsets=self.mcp_toolsets,     # Additional tools provided by mcp_config.json
            capabilities=[
                hooks,
                Thinking(effort=self.thinking)
            ],
            deps_type=AgentDeps
        )

        attempts=0
        while True:
            try:
                with disable_threads():     # disabling threads is necessary to have agent calls inside an agent tool
                    try:
                        result = agent.run_sync(
                            user_prompt=f"""
                            {user_prompt}
                            {response_format}
                            """,
                            message_history=history,
                            deps=self.deps,
                            usage_limits=UsageLimits(tool_calls_limit=20)
                        )
                    except ModelHTTPError as e:
                        # Fall back to empty response
                        log_error(f"Model HTTP Error: {e}")
                        return [], ""
                    break
            except Exception as e:
                attempts+=1
                if attempts>3:
                    raise e
                continue
        return result.new_messages(), result.output

    def simulate(self, prompt: str):
        """
        Simulate execution path of a certain prompt.
        Useful for assessing quality of prompt or generating examples for few shot agents.

        Args:
            prompt: string containing instructions for an AI Agent
        """
        return self.step(
            task="Simulate an execution path with a list of tool calls to solve the user's problem.",
            response_format="OUTPUT FORMAT\nBrief bullet point list of tool calls with possible scenarios. Single paragraph.",
            user_prompt=prompt,
            # tools=[],     # these are read by AgentWrapper class from ~/.optimus/mcp.json
        )
    
    def judge(self, premise: str, proposal: str):
        """
        Rate how well a proposed string incorporates information contained in a premise string.

        Args:
            premise: a string containing a premise in natural language. Facts, statements, information.
            proposal: a string containing a proposal to test againts the premise. Instructions, execution paths, procedures.
        """
        return self.step(
            task="""
            # YOUR ROLE
            You are the Judge. 
            Rate how well the PROPOSAL respects the PREMISE. A valid PROPOSAL must consider the PREMISE.
            """,
            response_format="""
            # YOUR OUTPUT FORMAT
            Output format in JSON:
            {{"critique": "... (how the PROPOSAL violates the PREMISE)", "score": "between 0 and 100"}}
            Return ONLY valid JSON.
            Escape all quotes inside string values.
            Escape all backslashes.
            Do not include markdown fences.
            """,
            user_prompt=f"""

            # PROPOSAL
            {proposal}

            # PREMISE
            {premise}
            """,
            tools=[
                # Tool(ask_human_expert, takes_ctx=False, metadata={'read_only': True})
            ],
            # history=evaluation_new_messages,
        )

    def merge(self, current: str, inbound: str):
        """
        Merge information contained inside inbound string with current string

        Args:
            current: a string containing current information
            inbound: a string containing new information to merge
        """
        return self.step(
            task="""
            # YOUR ROLE
            You are the Information Merger.
            Given a CURRENT text and an INBOUND text, create a new text that incorporates information from INBOUND inside CURRENT.
            """,
            reponse_format="",
            user_prompt=f"""

            # CURRENT
            {current}

            # INBOUND
            {inbound}
            """,
            tools=[
                # Tool(ask_human_expert, takes_ctx=False, metadata={'read_only': True})
            ],
            # history=evaluation_new_messages,
        )


    # def list_tool_calls(self, messages: list[ModelMessage]):
    #     tool_calls = [
    #         {
    #             "name": part.tool_name,
    #             "args": json.loads(part.args)
    #         }
    #         for message in messages
    #         for part in message.parts
    #         if part.part_kind == "tool-call"
    #     ]
    #     return tool_calls