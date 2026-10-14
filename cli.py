import asyncio
import argparse
# import dspy

from src.optimus import Optimus
from src.utils import prettify_chunk

parser = argparse.ArgumentParser(
    prog="optimus",
    description="Automatic optimization of text artifacts according to predefined criteria"
)
# run_mode = parser.add_mutually_exclusive_group(required=True)
# run_mode.add_argument('-f', '--file', help="source file")
# parser.add_argument('-d', '--dir', help='Where to save the prompt files', required=False)
# parser.add_argument('-c', '--criteria', help='criteria', action='append')
# parser.add_argument('-l', '--learn', action='store_true', help='enable learn mode')
parser.add_argument('-a', '--approval', action='store_true', help='Enable tool approval')
args = parser.parse_args()

# TODO: build a static binary for this cli
# TODO: operate CLI in two modes: local or remote.
#   Local is for testing out Optimus or single users.
#   Remote attaches to an Optimus backend for team work.

async def main():
    """
    Connects Optimus with User Interface
    """
    optimus = Optimus(approval=args.approval)
    while True:
        prompt = input("> ")
        async for chunk in optimus.grow(prompt):
            print(prettify_chunk(chunk["data"]), end="", flush=True)

asyncio.run(main())
