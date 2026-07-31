import argparse
# from alive_progress import alive_it, alive_bar
from src.gepa import GepaWrapper


# import time
# import random
# # bar = alive_it(
# #     it=[1,2,3,4,5,6,7,8,9,0],
# #     total=100,
# #     manual=True
# # )
# with alive_bar(100, manual=True) as bar:
#     for i in [1,2,3,4,5,6,7,8,9,0]:
#         bar.text(f"Number is {i}")
#         time.sleep(2)
#         bar(random.random())

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
    objective=input(""),
    seed=seed if seed and len(seed) > 0 else None
    # debug=True
)
best_artifact, best_score = optimus.optimize()
print("#########################################")
print(best_artifact)
print("#########################################")
print(f"Tokens spent on evaluations: {optimus.token_count}")
print(f"Best score: {best_score}/100.0")
if args.file:
    with open(args.file, 'w', encoding="utf-8") as f:
        print(best_artifact, file=f)
