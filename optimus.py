import argparse
import os
import asyncio
from src.gepa import GepaWrapper
from src.agent import AgentWrapper
from src.utils import stabilize_json
from src.log import log_internal_event

parser = argparse.ArgumentParser(
    prog="optimus",
    description="Automatic optimization of text artifacts according to predefined criteria"
)
# run_mode = parser.add_mutually_exclusive_group(required=True)
# run_mode.add_argument('-f', '--file', help="source file")
# parser.add_argument('-f', '--file', help='Where to save the optimized artifact', required=False)
# parser.add_argument('-d', '--dir', help='Where to save the prompt files', required=False)
# parser.add_argument('-c', '--criteria', help='criteria', action='append')
# parser.add_argument('-d', '--debug', action='store_true', help='enable debug output')
# parser.add_argument('-s', '--scope', help='scope to call', required=True)
args = parser.parse_args()

class Optimus(GepaWrapper):
    """
    Main class for the Optimus project.
    It has three methods: learn, apply and find_seed.
    Together, they allow Optimus to:
        - optimize prompts
        - store them in prompt files under ~/.optimus
        - deploy agents with prompts
    """

    def __init__(self, debug: bool = False):
        super().__init__(debug=debug)
        self.seeds_dir = os.environ["HOME"]+'/'+'.optimus/prompts'

    # async def chat(self):
    #     def list_modules():
    #         pass
    #     def make_module():
    #         pass
    #     def run_module():
    #         pass
    #     history = []
    #     with open(os.environ["HOME"]+'/'+'.optimus/SYSTEM.md', 'r', encoding="utf-8") as f:
    #         system = f.read()
    #     agent = AgentWrapper(lm=self.model_string)
    #     while True:
    #         _, _ = await agent.async_step(
    #                 task=system,
    #                 response_format="",
    #                 user_prompt=input("> "),
    #                 stream=True,
    #                 tools=[
    #                     "list_modules",
    #                     "run_module",
    #                     "make_module"
    #                 ]
    #             )

    def apply(self, objective: str):
        """
        Treats learned prompts as composable memory fragments.
        Prompts are recalled, combined and passed to agents for instruction following.
        Alternatively, stop if none of the seeds are good.

        Args:
            objective: task to complete
        """
        # TBD
        pass

    def learn(self, objective: str):
        """
        Run an optimus loop for a given objective.
        1. Recalls most similar known prompt or defines a new one
        2. Optimizes prompt
        3. Stores the optimized prompt to ~/.optimus/prompts

        Args:
            objective: a string for steering the known prompts
        """

        # 1. Find the most relevant known prompt to use as seed, or get a new one
        seed_filename, seed = self.find_seed(objective)

        # 2. Optimize seed with GEPA*
        optimized_seed, optimized_seed_score = self.optimize(
            objective=objective,
            seed={"name": seed_filename, "prompt": seed}
        )

        # 3. Store optimized seed to prompt file under ~/.optimus/prompts
        with open(self.seeds_dir+'/'+seed_filename, 'w', encoding="utf-8") as f:
            log_internal_event(f"[OPTIMUS] Saving to {self.seeds_dir}/{seed_filename}.")
            print(optimized_seed, file=f)
        
        return optimized_seed, optimized_seed_score

    def find_seed(self, objective: str):
        """
        Find the most relevant known prompt file to use as seed, or make a new one.
        In practice, optimized seeds stored as prompt files are used as memory fragments.

        Args:
            objective: user input to steer seed search
        """

        if not os.path.exists(self.seeds_dir):
            os.makedirs(self.seeds_dir)

        # 1. Build list of seed previews from prompt files
        preview_window = 100
        seeds_preview = []
        for filename in os.listdir(self.seeds_dir):
            with open(self.seeds_dir+'/'+filename, 'r', encoding="utf-8") as file:
                seeds_preview += [
                    {"name": filename, "preview": file.read()[:preview_window]+"..."}
                ]
        log_internal_event(f"[OPTIMUS] Seeds preview: {seeds_preview}")

        # 2. Pick a suitable seed or start from scratch with a new one
        agent = AgentWrapper(lm=self.model_string)
        _, seed_choice = agent.step(
            task=f"""
            # LIST OF AVAILABLE MEMORY FRAGMENTS
            {seeds_preview}

            # TASK
            Tell which memory fragment is suitable for incorporating the user request. Or propose a new memory fragment.
            """,
            response_format="""
            Output format in JSON:
            {{"explanation": "...", "name": "... (must be as specific as possible to maximize memory efficiency)"}}
            Return ONLY valid JSON.
            Escape all quotes inside string values.
            Escape all backslashes.
            Do not include markdown fences.
            """,
            # {{"explanation": "...", "name": "... (broad topic to include a domain of operations)"}}
            user_prompt=objective
        )
        # 3. Stabilize json
        stable_feedback = stabilize_json(
            unstable_string = seed_choice,
            expected_keys = ["name", "explanation"]
        )

        # 4. Retrieve seed from file or fallback to empty seed
        log_internal_event(f"[OPTIMUS] Choosing seed {stable_feedback}")
        try:
            with open(self.seeds_dir+'/'+stable_feedback["name"], 'r', encoding="utf-8") as f:
                seed = f.read()
        except FileNotFoundError:
            seed = None

        return stable_feedback["name"], seed if seed and len(seed) > 0 else None


# Main loop
optimus = Optimus()
while True:
    best_artifact, best_score = optimus.learn(objective=input("> "))
    print("#########################################")
    print(best_artifact)
    print("#########################################")
    print(f"Tokens spent on evaluations: {optimus.token_count}")
    print(f"Best score: {best_score}/100.0")
