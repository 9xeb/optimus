import argparse
import os
import asyncio
from src.gepa import GepaWrapper
from src.agent import AgentWrapper
from src.utils import stabilize_json
from src.log import log_internal_event

class Optimus(GepaWrapper):
    """
    Main class for the Optimus project.
    It has three methods: learn, apply and find_fragment.
    Together, they allow Optimus to:
        - optimize prompts
        - store them in prompt files under ~/.optimus
        - deploy agents with prompts
    """

    def __init__(self, debug: bool = False):
        super().__init__(debug=debug)
        self.fragments_dir = os.environ["HOME"]+'/'+'.optimus/prompts'

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
        Alternatively, stop if none of the fragments are good.

        Args:
            objective: task to complete
        """
        # TBD
        _, fragment = self.find_fragment(objective)
        if fragment is None:
            log_error("No memory fragments found")
            return None
        agent = AgentWrapper(enable_mcp_toolsets=True)
        _, response = agent.async_step(
            task=fragment,
            response_format="",
            user_prompt=objective
        )
        return response

    def learn(self, objective: str):
        """
        Run an optimus loop for a given objective.
        1. Recalls most similar known prompt or defines a new one
        2. Optimizes prompt
        3. Stores the optimized prompt to ~/.optimus/prompts

        Args:
            objective: a string for steering the known prompts
        """

        # 1. Find the most relevant known prompt to use as fragment, or get a new one
        fragment_filename, fragment = self.find_fragment(objective)

        # 2. Optimize fragment with GEPA*
        optimized_fragment, optimized_fragment_score = self.optimize(
            objective=objective,
            seed={"name": fragment_filename, "prompt": fragment}
        )

        # 3. Store optimized fragment to prompt file under ~/.optimus/prompts
        with open(self.fragments_dir+'/'+fragment_filename, 'w', encoding="utf-8") as f:
            log_internal_event(f"[OPTIMUS] Saving to {self.fragments_dir}/{fragment_filename}.")
            print(optimized_fragment, file=f)
        
        return optimized_fragment, optimized_fragment_score

    def find_fragment(self, objective: str) -> (str, str):
        """
        Find the most relevant known prompt file to use as fragment, or make a new one.
        In practice, optimized fragments stored as prompt files are used as memory fragments.

        Args:
            objective: user input to steer fragment search

        Returns:
            a tuple containing: the name of the memory fragment, the state of the fragment if existing (else None)
        """

        if not os.path.exists(self.fragments_dir):
            os.makedirs(self.fragments_dir)

        # 1. Build list of fragment previews from prompt files
        preview_window = 100
        fragments_preview = []
        for filename in os.listdir(self.fragments_dir):
            with open(self.fragments_dir+'/'+filename, 'r', encoding="utf-8") as file:
                fragments_preview += [
                    {"name": filename, "preview": file.read()[:preview_window]+"..."}
                ]
        log_internal_event(f"[OPTIMUS] fragments preview: {[fragment["name"] for fragment in fragments_preview]}")

        # 2. Pick a suitable fragment or start from scratch with a new one
        agent = AgentWrapper()
        _, fragment_choice = agent.step(
            task=f"""
            # LIST OF AVAILABLE MEMORY FRAGMENTS
            {fragments_preview}

            # TASK
            Tell which memory fragment is suitable for incorporating the user request. Or propose to create a new memory fragment name if none of the available match.
            Fragments must be as specific as possible. The more fragments, the better.
            """,
            response_format="""
            Output format in JSON:
            {{"explanation": "...", "name": "..."}}
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
            unstable_string = fragment_choice,
            expected_keys = ["name", "explanation"]
        )

        # 4. Retrieve fragment from file or fallback to empty fragment
        log_internal_event(f"[OPTIMUS] Choosing fragment {stable_feedback["name"]} - {stable_feedback["explanation"]}")
        try:
            with open(self.fragments_dir+'/'+stable_feedback["name"], 'r', encoding="utf-8") as f:
                fragment = f.read()
        except FileNotFoundError:
            fragment = None

        # log_internal_event(fragment)
        return stable_feedback["name"].replace('-', '_'), fragment if fragment and len(fragment) > 0 else None

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
parser.add_argument('-c', '--chat', action='store_true', help='enable chat mode')
# parser.add_argument('-s', '--scope', help='scope to call', required=True)
args = parser.parse_args()

optimus = Optimus()
if args.learn:
    # Main loop in optimus mode
    while True:
        best_artifact, best_score = optimus.learn(objective=input("> "))
        print("#########################################")
        print(best_artifact)
        print("#########################################")
        print(f"Tokens spent on evaluations: {optimus.token_count}")
        print(f"Best score: {best_score}/100.0")
else:
    result = optimus.apply(objective=input("> "))
    print(f"{result}")
# else:
#     # Main loop in tool debug chat loop
#     chat_agent = AgentWrapper(
#         thinking=True,
#         debug=False
#     )
#     while True:
#         _ = asyncio.run(chat_agent.chat(input("> ")))
