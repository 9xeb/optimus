import asyncio
import argparse

from src.optimus import Optimus

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

async def main():
    """
    Connects Optimus with User Interface
    """
    while True:
        optimus = Optimus(approval=args.approval)
        prompt = input("> ")
        print(await optimus.solve(prompt))
    # await asyncio.gather(
    #     optimus.consume_tasks()
    # )

asyncio.run(main())