import argparse
from src.gepa import GepaWrapper

parser = argparse.ArgumentParser(
    prog="optimus",
    description="Automatic optimization of text artifacts according to predefined criteria"
)
# run_mode = parser.add_mutually_exclusive_group(required=True)
# run_mode.add_argument('-f', '--file', help="source file")
# parser.add_argument('-m', '--model', help='Model name (openai/path/to/model)', required=True)
parser.add_argument('-f', '--file', help='Where to save the optimized artifact', required=False)
# parser.add_argument('-c', '--criteria', help='criteria', action='append')
# parser.add_argument('-d', '--debug', action='store_true', help='enable debug output')
# parser.add_argument('-s', '--scope', help='scope to call', required=True)
args = parser.parse_args()

if args.file:
    try:
        with open(args.file, 'r', encoding="utf-8") as f:
            seed = f.read()
    except FileNotFoundError:
        seed = None
else:
    seed = None
optimus = GepaWrapper(
    # model_string=args.model,
    objective=input("GOAL > "),
    seed=seed if seed and len(seed) > 0 else None
    # debug=True
)
optimized_artifact = optimus.optimize()
print("#########################################")
print(optimized_artifact)
print("#########################################")
print(f"Tokens spent on evaluations: {optimus.token_count}")
if args.file:
    with open(args.file, 'w', encoding="utf-8") as f:
        print(optimized_artifact, file=f)
