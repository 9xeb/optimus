import argparse
import os
import asyncio
import json
import functools

import litellm
import dspy
import mlflow

from dspy.utils.callback import BaseCallback
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from src.utils import stream_dspy_program
from src.log import log_internal_event

from src.ctxseg import CtxSeg

lm = dspy.LM(
    os.environ["OPENAI_API_MODEL"],
    api_base=os.environ["OPENAI_API_BASE"],
    api_key=os.environ["OPENAI_API_KEY"],
    cache=False     # stored in ~/.dspy_cache
)
# dspy.configure(lm=self.lm, callbacks=[LoggingCallback()])      # set default provider locally, can override with dspy.context
dspy.configure(lm=lm)
dspy.disable_litellm_logging()

parser = argparse.ArgumentParser(
    prog="optimus",
    description="Automatic optimization of text artifacts according to predefined criteria"
)
# run_mode = parser.add_mutually_exclusive_group(required=True)
# run_mode.add_argument('-f', '--file', help="source file")
# parser.add_argument('-d', '--dir', help='Where to save the prompt files', required=False)
# parser.add_argument('-c', '--criteria', help='criteria', action='append')
# parser.add_argument('-l', '--learn', action='store_true', help='enable learn mode')
args = parser.parse_args()

# MCP stuff

# Setup mlflow integration
# mlflow.set_tracking_uri(os.environ["MLFLOW_API_BASE"])
# mlflow.set_experiment("OptimusV2")
# mlflow.autolog()
# Setup dspy LM


async def main():
    """
    Main optimus loop function, with message queue streaming.
    """

    while True:
        # Wrap program call around MCP session
        async with streamablehttp_client(os.environ["MCP_GATEWAY"]) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()

                mcp_tools = [dspy.Tool.from_mcp_tool(session, tool) for tool in tools.tools]
                # Tools have .name, .desc that can be used for discovery
                log_internal_event(f"MCP TOOLS: {[tool.name for tool in mcp_tools]}")
                prompt = input("> ")

                agent = CtxSeg(tools=mcp_tools)
                optimus_output = await stream_dspy_program(
                    agent,
                    task=prompt
                )
                print(f"Traces: {agent.unclassified_traces}")
                print(optimus_output.report)

    # await asyncio.gather(
    #     optimus.listen()
    # )

asyncio.run(main())

