import argparse
import os
import asyncio

from pydantic_ai.models.function import _estimate_usage

from src.gepa import GepaWrapper
from src.agent import AgentWrapper
from src.utils import stabilize_json
from src.log import log_internal_event, log_error, log_response

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
        self.token_count = 0

    # Alternative to GepaWrapper's optimize(), implements explorer-worker pattern
    def explore(self, objective: str, seed: str):
        """
        Explorer strategy.
        The philosophy behind this is: LLMs are much more capable than they look. Stop getting in their way. They just need believable tools.
        Also, pre-planning is limited to predictable scenarios like coding, not general-purpose agentic tasks.

        There are two actors in here: Explorer and Agent
        Explorer keeps track of high level strategic results and runs strategies by calling Agent as a tool.
        Agent has two builtin tools to confirm success or give up, plus all the tool to access its world.
        When Agent is run, it starts with an empty context and produces a recap as a result back to Explorer.
        Problem stays the same all the time. Strategy changes.
        Agent is the only one with actual access to tools, and is incentivized to give up quickly if it meets a dead end.
        Explorer is a higher level agent, that can sustain complex problems because its context grows slowly.
        This process should be able to sustain long horizon problems.
        """
        # find -> merge(seed,objective) -> simulate -> judge -> merge
        self.async_token_usage = 0
        async def run_strategy(strategy: str):
            """
            Send instructions to an agent.
            Provide an unexplored course of action to solve the problem.

            Args:
                strategy: a strategy to put to the test
            """
            worker_agent = AgentWrapper(name="WORKER", enable_mcp_toolsets=True)
            new_messages, response = await worker_agent.async_step(
                task=strategy,
                response_format="",
                user_prompt=objective
            )
            token_usage = _estimate_usage(new_messages)
            self.token_count += token_usage.input_tokens + token_usage.output_tokens
            return response

        # def give_up():
        #     """
        #     Final fallback tool. Use this exclusively to terminate the search and indicate failure to find new suggestions.
        #     You must call this ONLY as an absolute last resort, strictly after all other available tools, strategies, and reasoning paths have been exhaustively attempted and have conclusively failed.
        #     """
        #     return "Give-up acknowledged. Provide a full bullet point recap of what you have done and why you stopped."

        # Main Explorer Agent
        explorer_agent = AgentWrapper(name="EXPLORER")
        new_messages, feedback = explorer_agent.step(
            task=f"""
            Run natural-language strategies until you solve the problem.
            When a strategy fails, provide a new, unexplored course of action.

            # INITIAL STRATEGY
            {seed}
            """,
            response_format="",
            user_prompt=f"{objective}",
        )
        log_response(f"{feedback}")
        explorer_token_usage = _estimate_usage(new_messages)
        self.token_count += explorer_token_usage.input_tokens + explorer_token_usage.output_tokens
        return new_messages, feedback

    # def apply(self, objective: str):
    #     """
    #     Treats learned prompts as composable memory fragments.
    #     System Prompts are recalled, combined and passed to agents for instruction following.
    #     Alternatively, stop if none of the fragments are good.

    #     Args:
    #         objective: task to complete
    #     """
    #     # 1. Find relevant memory fragment
    #     _, fragment = self.find_fragment(objective)
    #     if fragment is None:
    #         log_error("No memory fragments found")
    #         return None
    #     # 2. Run an agent with memory fragment and objective
    #     agent = AgentWrapper(enable_mcp_toolsets=True)
    #     _, response = agent.step(
    #         task=fragment,
    #         response_format="",
    #         user_prompt=objective
    #     )
    #     return response

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
        
        # 4. Run the fragment
        # agent = AgentWrapper(name="AGENT", enable_mcp_toolsets=True)
        # _, response = agent.step(
        #     task=optimized_fragment,
        #     response_format="",
        #     user_prompt=objective
        # )
        # log_response(f"AGENT - {response}")

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

        # if learn:
        #     # Learn mode. Creating and updating memories. Maximum fragmentation
        #     memory_operation = """
        #     Tell which memory fragment is suitable for incorporating the user request. Or propose to create a new memory fragment name if none of the available match.
        #     Fragments must be as specific as possible. The more fragments, the better.
        #     """
        # else:
        #     # Apply mode. Recalling memories. Maximum consolidation
        #     memory_operation = """

        #     """

        # 2. Pick a suitable fragment or start from scratch with a new one
        agent = AgentWrapper(name="MEMORY")
        _, fragment_choice = agent.step(
            task=f"""
            # LIST OF AVAILABLE MEMORY FRAGMENTS
            {fragments_preview}

            # TASK
            Tell which memory fragment is suitable for incorporating the user request. Or propose to create a new memory fragment name if none of the available match.
            Information must be consolidated in large fragments to cluster knowledge by broad domain. The fewer memory fragments, the better.
            """,
            # Tell which memory fragment is suitable for incorporating the user request. Or propose to create a new memory fragment name if none of the available match.
            # Fragments must be as specific as possible. The more fragments, the better.
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
# parser.add_argument('-c', '--chat', action='store_true', help='enable chat mode')
# parser.add_argument('-l', '--learn', action='store_true', help='enable learn mode')
# parser.add_argument('-s', '--scope', help='scope to call', required=True)
args = parser.parse_args()

optimus = Optimus()
# if args.learn:
# Main loop in optimus mode
while True:
    best_artifact, best_score = optimus.learn(objective=input("> "))
    print("#########################################")
    print(best_artifact)
    print("#########################################")
    print(f"Tokens spent on evaluations: {optimus.token_count}")
    print(f"Best score: {best_score}/100.0")
# else:
#     result = optimus.apply(objective=input("> "))
#     print(f"{result}")
# else:
#     # Main loop in tool debug chat loop
#     chat_agent = AgentWrapper(
#         thinking=True,
#         debug=False
#     )
#     while True:
#         _ = asyncio.run(chat_agent.chat(input("> ")))
