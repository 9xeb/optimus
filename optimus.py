import argparse

from src.gepa import GepaWrapper

# model_string = 'openai/unsloth/gemma-4-E4B-it-GGUF:Q4_K_M'

parser = argparse.ArgumentParser(
    prog="optimus",
    description="Automatic optimization of text artifacts according to predefined criteria"
)
# run_mode = parser.add_mutually_exclusive_group(required=True)
# run_mode.add_argument('-f', '--file', help="source file")
parser.add_argument('-m', '--model', help='Model name (openai/path/to/model)', required=True)
# parser.add_argument('-c', '--criteria', help='criteria', action='append')
# parser.add_argument('-d', '--debug', action='store_true', help='enable debug output')
# parser.add_argument('-s', '--scope', help='scope to call', required=True)
args = parser.parse_args()

optimus = GepaWrapper(
    model_string=args.model,
    objective=input("GOAL > "),
    # debug=True
)
optimized_artifact = optimus.optimize()
print("#########################################")
print(optimized_artifact)
print("#########################################")
