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
    def __init__(self, debug: bool = False):
        super().__init__(debug=debug)
        self.seeds_dir = os.environ["HOME"]+'/'+'.optimus/prompts'

    async def chat(self):
        def list_modules():
            pass
        def make_module():
            pass
        def run_module():
            pass
        history = []
        with open(os.environ["HOME"]+'/'+'.optimus/SYSTEM.md', 'r', encoding="utf-8") as f:
            system = f.read()
        agent = AgentWrapper(lm=self.model_string)
        while True:
            _, _ = await agent.async_step(
                    task=system,
                    response_format="",
                    user_prompt=input("> "),
                    stream=True,
                    tools=[
                        "list_modules",
                        "run_module",
                        "make_module"
                    ]
                )

    def apply(self, objective: str):
        """
        Apply the available seeds to achieve the objective. Stop if none of the seeds are good.
        Recalls and merges multiple seeds, synergizes with very fragmented seeds in learn mode.

        Args:
            objective: task to complete
        """
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

        # 1. Find the most relevant known prompt to use as seed
        seed_filename, seed = self.find_seed(objective)
        # 2. Optimize seed with gepa
        optimized_seed, optimized_seed_score = self.optimize(
            objective=objective,
            # seed={"name": "test", "prompt": None},
            seed={"name": seed_filename, "prompt": seed}
        )

        # 3. Store optimized seed
        with open(self.seeds_dir+'/'+seed_filename, 'w', encoding="utf-8") as f:
            log_internal_event(f"[OPTIMUS] Saving to {self.seeds_dir}/{seed_filename}.")
            print(optimized_seed, file=f)
        
        return optimized_seed, optimized_seed_score

    def find_seed(self, objective: str):
        """
        Lazy find the most relevant known prompt file to use as seed
        """

        if not os.path.exists(self.seeds_dir):
            os.makedirs(self.seeds_dir)

        # 1. Build list of seeds previews
        preview_window = 100
        seeds_preview = []
        for filename in os.listdir(self.seeds_dir):
            with open(self.seeds_dir+'/'+filename, 'r', encoding="utf-8") as file:
                seeds_preview += [
                    {"name": filename, "preview": file.read()[:preview_window]+"..."}
                ]
        log_internal_event(f"[OPTIMUS] Seeds preview: {seeds_preview}")

        # 2. Choose one seed
        # agent = AgentWrapper(lm=self.model_string)
        # _, seed_choice = agent.step(
        #     task=f"""
        #     # LIST OF AVAILABLE PROMPT FILES
        #     {seeds_preview}

        #     # TASK
        #     Tell if a prompt file name is 100% suitable for routing the user request to. If not, give a new name.
        #     """,
        #     # Choose the most fitting prompt file name to route the user request to.
        #     # If there aren't any routable prompts for the request, make a new prompt file name that you would add to the list.
        #     response_format="""
        #     Output format in JSON:
        #     {{"explanation": "...", "name": "... (must be as specific as possible to maximize prompt efficiency)"}}
        #     Return ONLY valid JSON.
        #     Escape all quotes inside string values.
        #     Escape all backslashes.
        #     Do not include markdown fences.
        #     """,
        #     # {{"explanation": "...", "name": "... (broad topic to include a domain of operations)"}}
        #     # {{"name": "... (must be as specific as possible to maximize prompt efficiency)", "explanation": "..."}}
        #     user_prompt=objective
        # )

        agent = AgentWrapper(lm=self.model_string)
        _, seed_choice = agent.step(
            task=f"""
            # LIST OF AVAILABLE MEMORY FRAGMENTS
            {seeds_preview}

            # TASK
            Tell which memory fragment is suitable for incorporating the user request. Or propose a new memory fragment.
            """,
            # Choose the most fitting prompt file name to route the user request to.
            # If there aren't any routable prompts for the request, make a new prompt file name that you would add to the list.
            response_format="""
            Output format in JSON:
            {{"explanation": "...", "name": "... (must be as specific as possible to maximize memory efficiency)"}}
            Return ONLY valid JSON.
            Escape all quotes inside string values.
            Escape all backslashes.
            Do not include markdown fences.
            """,
            # {{"explanation": "...", "name": "... (broad topic to include a domain of operations)"}}
            # {{"name": "... (must be as specific as possible to maximize prompt efficiency)", "explanation": "..."}}
            user_prompt=objective
        )


        # 3. Stabilize json
        stable_feedback = stabilize_json(
            unstable_string = seed_choice,
            expected_keys = ["name", "explanation"]
        )

        # 4. Get seed or fallback to empty seed
        log_internal_event(f"[OPTIMUS] Choosing seed {stable_feedback}")

        try:
            with open(self.seeds_dir+'/'+stable_feedback["name"], 'r', encoding="utf-8") as f:
                seed = f.read()
        except FileNotFoundError:
            seed = None

        return stable_feedback["name"], seed if seed and len(seed) > 0 else None


# 2. Optimize seed
optimus = Optimus()
while True:
    best_artifact, best_score = optimus.learn(objective=input("> "))
    print("#########################################")
    print(best_artifact)
    print("#########################################")
    print(f"Tokens spent on evaluations: {optimus.token_count}")
    print(f"Best score: {best_score}/100.0")
