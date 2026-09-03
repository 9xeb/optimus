import argparse
import os
import asyncio
import json

import litellm
import dspy
import mlflow
from alive_progress import alive_bar
from pydantic_ai.models.function import _estimate_usage

from src.gepa import GepaWrapper
from src.agent import AgentWrapper
from src.utils import TITLES, stabilize_json, extract_execution_path, count_tokens
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

    def __init__(self, message_queue: asyncio.Queue, debug: bool = False):
        super().__init__(debug=debug)
        self.fragments_dir = os.environ["HOME"]+'/'+'.optimus/prompts'
        self.token_count = 0
        # self.message_queue = message_queue
        self.queue = asyncio.Queue()
        self.message_queue = asyncio.Queue()
        self.memories = {}

        self.chat_agent = AgentWrapper(name="CHAT", message_queue=self.message_queue)

        # Setup mlflow integration
        mlflow.set_tracking_uri(os.environ["MLFLOW_API_BASE"])
        mlflow.set_experiment("Optimus")
        mlflow.autolog()

        # Setup dspy LM
        self.lm = dspy.LM(
            os.environ["OPENAI_API_MODEL"],
            api_base=os.environ["OPENAI_API_BASE"],
            api_key=os.environ["OPENAI_API_KEY"],
            cache=False     # stored in ~/.dspy_cache
        )
        dspy.configure(lm=self.lm)      # set default provider locally, can override with dspy.context



    """
    Optimus optimizes agentic DSPy programs, starting from a set of primitives.
    Primitives: 
        - signature strings
        - dspy.ChainOfThought is a module for pure, tool-less text generation
        - dspy.ReAct is a module for agentic operations
        - dspy.RLM is a module for navigating large contexts (e.g. entire optimus memory)
        - dspy.GEPA generally optimizes prompts
        - stream_program() runs any module with signatures and streams outputs
    """

    async def live(self):
        """
        Main optimus loop, can run 24/7 and:
            - explores your environment
            - asks questions
            - solves problems
            - hums optimizations when there are no problems
        Live data structures make this possible:
            - memories
            - problem pool (or queue)
        With time, it becomes a valuable co-worker in your digital space.
        """
        pass

    async def problem_polling(self):
        """
        Poll for problems to solve in the problems queue.
        When a problem comes up, lock all other coroutines and focus on it.
        """
        pass
    async def enviroment_exploration(self):
        """
        Curiously explore the environment using read only tools:
            - memorize useful tool peculiarities
            - find answers to common useful questions, worthy of being remembered in the given domain
        """
        pass
    async def program_optimization(self):
        """
        Optimize programs/runbooks, through runnable code or automated prompt engineering.
        """
        pass
    async def memory_management(self):
        """
        DSPy powered memory canvas loop.
        """
        merger = dspy.ChainOfThought(signature="old_truth:str,new_truth:str -> merged_truth:str")
        recaller = dspy.ChainOfThought(signature="memory_names:list[str],query:str -> most_relevant_memory_name:str | None")
        # generator = dspy.ChainOfThought(signature="objective:str -> short_knowledge_paragraph:str, title:str")
        # generator = dspy.ChainOfThought(signature="prompt:str -> expanded_prompt:str, title:str")
        # generator = dspy.ChainOfThought(signature="prompt:str -> runbook:str, title:str")
        def ask(question: str) -> str:
            """
            Dispel any doubt regarding:
                - environment-specific knowledge that the user might have.
                - possible information the user has that would make things much simpler.

            Args:
                question: The question you have

            Return:
                An answer to your question
            """
            print("#################### QUESTION #####################")
            print(question)
            answer = input("Answer > ")
            print("#########################################")
            return answer
        generator = dspy.ReAct(signature="prompt:str -> runbook:str, title:str", tools=[ask])

        memories = {}
        while True:
            prompt = input("> ")
            # Recall memory fragment
            recalled = await self.run_program(
                recaller,
                memory_names=[name for name, _ in memories.items()],
                query=prompt
            )
            log_error(f"{recalled.most_relevant_memory_name}")
            if recalled.most_relevant_memory_name not in memories.keys():
                # Generate new memory fragment if none exists
                generator_output = await self.run_program(
                    generator,
                    prompt=prompt
                )
                memories = memories | {generator_output.title: generator_output.runbook}
            else:
                # Update memory fragment
                merger_output = await self.run_program(
                    merger,
                    old_truth=memories[recalled.most_relevant_memory_name],
                    new_truth=prompt
                )
                memories = merger_output.merged_truth

            print("\n############### MEMORIES ####################\n")
            print(f"{[name for name, _ in memories.items()]}")
            print("\n####################################\n")

    async def recall(self, query: str):
        """
        Use RLM to explore the memory and find everything related to the query.
        It is a READ ONLY operation.
        """
        recaller = dspy.RLM(signature="context:str,query:str -> truth:str")
        return await self.run_program(module=recaller, context=self.memory, query=query)
    async def store(self):
        """
        When information is not present in memory, generate it
        """
        dspy.RLM(signature="memory_feedback:str,query:str -> ")
        pass
    async def explore(self):
        pass
    async def execute(self):
        pass
    async def improve(self):
        pass

    async def run_program(self, module: dspy.Module, **kwargs):
        """
        Run a DSPy program, stream tokens and return Prediction
        Only supports ChainOfThought module.

        Args:
            module: DSPy module to run
            **kwargs: list of kwargs corresponding to the signature inputs
        
        Return
            a Prediction object containing the signature outputs
        """
        # Wrap module around streamer
        streamer = dspy.streamify(
            program=module,
            stream_listeners=[    # recursively extract all output field names regardless of the module type
                dspy.streaming.StreamListener(signature_field_name=output_field_key)
                for _, (name, predictor) in enumerate(module.named_predictors())
                for output_field_key in predictor.signature.output_fields.keys()
            ]
        )

        # Run program=module+arguments and stream output fields
        try:
            prediction = ""
            async for chunk in streamer(**kwargs):          # pass signature values to module inside streamer
                if isinstance(chunk, dspy.streaming.StreamResponse):
                    # when multi-output, chunk.signature_field_name is the name of the output that the chunk belongs to
                    print(f"{chunk.chunk}", end="", flush=True)
                    # await self.message_queue.put(chunk.chunk)
                elif isinstance(chunk, dspy.Prediction):      # the last chunk with final response is a Prediction
                    prediction = chunk
        except AttributeError as e:
            pass    # skip current bug where some dspy.RLM predictors have broken chunks
        return prediction

    async def apply(self, instructions: str, objective: str):
        """
        Treats learned prompts as composable memory fragments.
        System Prompts are recalled, combined and passed to agents for instruction following.
        Alternatively, stop if none of the fragments are good.

        Args:
            objective: task to complete
        
        Returns:
            a string with the response
        """
        # 1. Run an agent with instructions and objective
        # self.progress_bar.title(TITLES["agent"])
        agent = AgentWrapper(name="AGENT", enable_mcp_toolsets=True, message_queue=self.message_queue)
        new_messages, response = await agent.act(
            system_prompt=instructions,
            prompt=objective
        )

        # 2. Extract tool call execution path for HITL review
        return new_messages, response

    async def learn(self, instructions: str, objective: str):
        """
        Run an optimus loop for a given objective.
        1. Recalls most similar known prompt or defines a new one
        2. Optimizes prompt
        3. Stores the optimized prompt to ~/.optimus/prompts

        Args:
            objective: a string for steering the known prompts
        """

        # # 1. Find the most relevant known prompt to use as fragment, or get a new one
        # fragment_filename, fragment = self.find_fragment(objective)

        # 2. Optimize fragment with GEPA* if new fragment, else simple merge
        # optimized_fragment, _ = self.optimize(
        #     objective=objective,
        #     seed=instructions,
        #     message_queue=self.message_queue
        # )
        optimized_fragment, _ = await self.optimize_async(
            objective=objective,
            seed=instructions,
            message_queue=self.message_queue
        )
        return optimized_fragment

        # 3. Act upon fragment with HITL
        # self.apply(instructions=optimized_fragment, objective=objective)

        # 4. learn() again if HITL is bad

        # 5. remember() if HITL is good

    # def memorize(self, fragment_name: str, fragment: str):
    #     """
    #     Store memory fragment to file at ~/.optimus/prompts

    #     Args:
    #         fragment_name: name of the file
    #         fragment: content of the memory fragment
    #     """
    #     # 3. Store optimized fragment to prompt file under ~/.optimus/prompts
    #     with open(self.fragments_dir+'/'+fragment_name, 'w', encoding="utf-8") as f:
    #         log_internal_event(f"[OPTIMUS] Saving to {self.fragments_dir}/{fragment_name}.")
    #         print(fragment, file=f)

    async def find_fragment(self, objective: str) -> (str, str):
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

        # 1. Retrieve list of memory fragments
        fragments_preview = self.list_fragments()
        log_internal_event(f"[OPTIMUS] Memory Fragments: {[fragment["name"] for fragment in fragments_preview]}")

        # 2. Pick a suitable fragment or start from scratch with a new one
        agent = AgentWrapper(name="MEMORY", message_queue=self.message_queue)
        _, fragment_choice = await agent.async_step(
            task=f"""
            # LIST OF AVAILABLE MEMORY FRAGMENTS
            {fragments_preview}

            # TASK
            Tell which memory fragment is suitable for incorporating the user request. Or propose to create a new memory fragment name if none of the available match.
            Prefer consolidating information in fragments by type of task for AI Agents.
            """,
            # Prefer consolidating information into the same fragment, unless the user request is for a totally unrelated domain. The fewer memory fragments, the better.

            # Tell which memory fragment is suitable for incorporating the user request. Or propose to create a new memory fragment name if none of the available match.
            # Fragments must be as specific as possible. The more fragments, the better.
            response_format="""
            Output format in JSON:
            {{"explanation": "... (why the memory fragment was chosen)", "name": "..."}}
            Return ONLY valid JSON.
            Escape all quotes inside string values.
            Escape all backslashes.
            Do not include markdown fences.
            """,
            # {{"explanation": "...", "name": "... (broad topic to include a domain of operations)"}}
            user_prompt=objective,
            stream=True
        )
        # 3. Stabilize json
        stable_feedback = stabilize_json(
            unstable_string = fragment_choice,
            expected_keys = ["name", "explanation"]
        )

        # 4. Retrieve fragment from file or fallback to empty fragment
        # log_internal_event(f"[OPTIMUS] Proposing fragment {stable_feedback["name"]} - {stable_feedback["explanation"]}")
        try:
            with open(self.fragments_dir+'/'+stable_feedback["name"], 'r', encoding="utf-8") as f:
                fragment = f.read()
        except FileNotFoundError:
            fragment = None

        # log_internal_event(fragment)
        return stable_feedback["name"].replace('-', '_'), fragment if fragment and len(fragment) > 0 else None

    def list_fragments(self):
        """
        List memory fragments from ~/.optimus/prompts in preview mode
        """
        preview_window = 100
        fragments_preview = []
        for filename in os.listdir(self.fragments_dir):
            with open(self.fragments_dir+'/'+filename, 'r', encoding="utf-8") as file:
                fragments_preview += [
                    # {"name": filename, "preview": file.read()}
                    {"name": filename, "preview": file.read()[:preview_window]+"..."}
                ]
        return fragments_preview

    def hitl(self, prompt: str, report: str = None) -> str:
        """
        Human in the loop implementation.
        It works by expecting suggestions by human to fix execution path.
        If no suggestion is provided (i.e. human just pressed ENTER), the check is passed

        Args:
            prompt: a short text to present to the human
            history: the latest agent message history
            response: a string containing the response
        """

        # Prepare execution path recap
        # if len(history) > 0:
        #     recap_agent = AgentWrapper(name="RECAP")
        #     new_messages, response = recap_agent.recap(history=history)
        #     print("########## EXECUTION PATH ###############")
            # print(json.dumps(extract_execution_path(new_messages), indent=4))
        # print("########## RECAP #####################")
        # print(response)
        # print("#########################################")

        # In the future, human in the loop will happen on side channel (Telegram, Discord, ecc...)
        if report is not None:
            print("########## RECAP #####################")
            print(report)
            print("#########################################")

        # with self.progress_bar.pause():
        human_in_the_loop = input(f"{prompt} > ")
        return human_in_the_loop

    # async def optimus_loop(self, queue: asyncio.Queue):
    #     """
    #     Producer of optimus messages
    #     """
    #     optimus = Optimus(message_queue=queue)
    #     # VARIANT: after hitl(), apply()+binary_hitl() to check if current fragment is enough to predict outcome, else learn()
    #     while True:
    #         query = optimus.hitl(prompt="Give Optimus a Task")   # user provides a task
    #         fragment_name, fragment = await optimus.find_fragment(objective=query)
    #         # SHORT PATH: Can infer without learning?
    #         if fragment is not None:
    #             new_messages, response = await optimus.apply(instructions=fragment, objective=query)
    #             # await asyncio.gather(optimus.stream(), optimus.apply())
    #             remark = optimus.hitl(prompt="Tell Optimus how to do better (Empty ENTER if satisfied)")
    #         else:
    #             remark = " "    # whitespace to trigger LONG PATH when new fragment is requested
    #         # LONG PATH if SHORT PATH triggered a non-empty human remark
    #         while len(query) > 0 and len(remark) > 0:   # LONG PATH: Must learn fragment
    #                 fragment = await optimus.learn(instructions=fragment, objective=query+remark)
    #                 # new_messages, response = optimus.apply(instructions=fragment, objective=query+remark)
    #                 remark = optimus.hitl(prompt="Tell Optimus how to do better (Empty ENTER if satisfied)")
    #         # FINAL STEP: memorize new fragment
    #         optimus.memorize(fragment_name=fragment_name, fragment=fragment)
    #     await queue.put(None)

    # async def recall(self, query: str):
    #     """
    #     Recall memory fragments and queue them
    #     """
    #     fragment_name, fragment = await self.find_fragment(objective=query)
    #     return await self.draft(objective=query, draft=fragment)
    #     # await self.queue.put(
    #     #     {
    #     #         "coroutine": "draft",
    #     #         "params": {
    #     #             "objective": query,
    #     #             "draft": fragment,
    #     #             "draft_name": fragment_name
    #     #         }
    #     #     }
    #     # )

    async def draft(self, objective: str, draft: str, fix: str = ""):
        """
        Attach to queue and draft the next prompt when a feedback arrives
        """
        # CONCISION_CLAUSE = """
        # # INSTRUCTIONS OUTPUT CONSTRAINTS
        # AI Agent instructions must be concise to the extreme. Break syntax and grammar, only pure operational focus.
        # """

        # Core logic: merge old draft with objective+fix
        reflection_agent = AgentWrapper(
            name="REFLECTION",
            enable_mcp_toolsets=True,
            simulated=True,
            message_queue=self.message_queue
        )
        merger_new_messages, draft = await reflection_agent.merge(
            current=draft,
            inbound=objective+"\n"+fix,    # Fix reminds of the objective: "generate AI agent instructions"
            # output_format="AI Agent instructions to solve the class of problems incorporating the merged information" + CONCISION_CLAUSE
            # output_format="Python code with reference implementation and placeholder tool calls to solve the class of problems incorporating the merged information"
            output_format="Human-readable pseudo-code instructions with reference implementation and relevant placeholder tool calls to solve the class of problems incorporating the merged information"
            # output_format="Python code referencing available tools to implement reference solution the class of problems incorporating the merged information"
            # output_format="Bash script referencing available tools to implement reference solution the class of problems incorporating the merged information"
        )

        self.token_count += count_tokens(merger_new_messages)
        # return await self.eval(objective=objective, draft=draft)
        # await self.queue.put(
        #     {
        #         "kind": "draft",
        #         "content": {
        #             "objective": objective,
        #             "draft": draft,
        #             "draft_name": msg["content"]["draft_name"]
        #         }
        #     }
        # )

        # Temporary override where draft is recap and eval is skipped
        return await self.review(objective=objective, draft=draft, recap=draft)

    async def eval(self, objective: str, draft: str):
        """
        Attach to queue and evaluate draft when it arrives
        """
        # 2. Eval prompt draft (this actually acts on the environment)
        # TODO: replace this with outsourcing to opencode or pi
        eval_agent = AgentWrapper(name="AGENT", enable_mcp_toolsets=True, message_queue=self.message_queue)
        system_prompt = f"""
        # REFERENCE INSTRUCTIONS
        {draft}

        # TASK
        Follow the reference instructions 
        """
        eval_new_messages, eval_response = await eval_agent.act(system_prompt=draft, prompt=objective)
        self.token_count += count_tokens(eval_new_messages)
        return await self.review(objective=objective, draft=draft, recap=eval_response)
        # await self.queue.put(
        #     {
        #         "kind": "eval",
        #         "params": {
        #             "objective": objective,
        #             "draft": draft,
        #             "recap": eval_response
        #         }
        #     }
        # )

    async def review(self, objective: str, draft: str, recap: str):
        """
        Attach to queue and review incoming evaluations
        """
        # 4. Judge recap vs raw_objective (without concision clause)
        judge_agent = AgentWrapper(
            name="JUDGE",
            enable_mcp_toolsets=True,
            message_queue=self.message_queue,
            simulated=True,
            debug=self.debug
        )
        judge_new_messages, feedback = await judge_agent.judge_execution_path(
            execution_path=recap,
            objective=objective
        )
        self.token_count += count_tokens(judge_new_messages)
        fix = feedback["fix"]
        score = float(feedback["score"])

        if score < 90:
            # Send to fix the draft
            return await self.draft(objective=objective, draft=draft, fix=fix)
        else:
            # Send for human review
            return await self.human_review(objective=str, draft=draft, report=recap)

    async def human_review(self, objective: str, draft: str, report: str):
        """
        Attach to queue and call human when machine review is done
        """
        # Call human for review
        if report is not None:
            print("########## RECAP #####################")
            print(report)
            print("#########################################")
        human_in_the_loop = input(f"Suggest fix ()> ")
        if len(human_in_the_loop) == 0:
            return await self.memorize(fragment_name="", fragment=draft)
            # await self.queue.put({"kind": "memory", "content": {"draft": draft}})
        else:
            return await self.draft(objective=objective, draft=draft, fix=human_in_the_loop)
            # await self.queue.put(
            #     {
            #         "kind": "review",
            #         "params": {
            #             "objective": objective,
            #             "draft": draft,
            #             "fix": human_in_the_loop
            #         }
            #     }
            # )

    async def memorize(self, fragment_name: str, fragment: str):
        """
        Store memory fragment to file at ~/.optimus/prompts

        Args:
            fragment_name: name of the file
            fragment: content of the memory fragment
        """
        # 3. Store optimized fragment to prompt file under ~/.optimus/prompts
        with open(self.fragments_dir+'/'+fragment_name, 'w', encoding="utf-8") as f:
            log_internal_event(f"[OPTIMUS] Saving to {self.fragments_dir}/{fragment_name}.")
            print(fragment, file=f)
        await self.queue(None)              # signal optimus coroutines to quit
        await self.message_queue.put(None)  # signal message streaming to quit

    async def chat(self):
        """
        Async chat loop
        """
        query = input("> ")
        # log_internal_event(f"History: {len(self.chat_agent.history)}")
        if query == "/new":
            self.chat_agent.clear_chat()
        else:
            _ = await self.chat_agent.chat(
                system_prompt="""
                Your task is to perform requirements analysis with the user.
                Grill the user until you converge to ready-to-deploy AI Agents instructions.
                """,
                # Work through all possible ambiguities with the user until you converge to indisputable requirements for both of you.
                prompt=query
            )
        return await self.chat()

    async def route(self):
        """
        Consume optimus queue and route to async coroutine
        """
        query = input("> ")

        # Static path (active human active bot): recall -> (re)draft -> memorize
        fragment_name, fragment = await self.find_fragment(objective=query)
        new_fragment = await self.draft(objective=query, draft=fragment)
        await self.memorize(fragment_name=fragment_name, fragment=new_fragment)

        # Static path (passive human active bot): analyize_requirements -> draft+grill loop -> memorize

        # TODO: dynamic path, self-programmable by optimus
        # ...

        # while True:
        #     msg = await self.queue.get()
        #     if msg is None:
        #         return
        #     kind = msg["kind"]
        #     params = msg["params"]
        #     if self.coroutines.get(kind):
        #         log_internal_event(f"Calling coroutine {kind} with params {params}")
        #         await self.coroutines[kind](**params)
        #     else:
        #         raise ValueError(f"Unknown coroutine: {kind}")

    async def stream(self):
        """
        Consumer of agent streaming messages
        """
        while True:
            text = await self.message_queue.get()
            if text is None:
                break
            print(f"{text}", end="", flush=True)
        # print("Consumer is done")



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

async def main():
    """
    Main optimus loop function, with message queue streaming.
    """
    queue = asyncio.Queue()
    optimus = Optimus(message_queue=queue)
    await asyncio.gather(
        # optimus.route(),
        # optimus.chat(),
        # optimus.test(),
        optimus.memory_management(),
        # optimus.stream()
    )

    # # with alive_bar(
    # #     total=100,
    # #     dual_line=True,
    # #     manual=True,
    # #     title_length=max([len(TITLES[title]) for title in TITLES])
    # # ) as progress_bar:
    # queue = asyncio.Queue()

    # async def optimus_loop(queue: asyncio.Queue):
    #     """
    #     Producer of optimus messages
    #     """
    #     optimus = Optimus(message_queue=queue)
    #     # VARIANT: after hitl(), apply()+binary_hitl() to check if current fragment is enough to predict outcome, else learn()
    #     while True:
    #         query = optimus.hitl(prompt="Give Optimus a Task")   # user provides a task
    #         fragment_name, fragment = await optimus.find_fragment(objective=query)
    #         # SHORT PATH: Can infer without learning?
    #         if fragment is not None:
    #             new_messages, response = await optimus.apply(instructions=fragment, objective=query)
    #             # await asyncio.gather(optimus.stream(), optimus.apply())
    #             remark = optimus.hitl(prompt="Tell Optimus how to do better (Empty ENTER if satisfied)")
    #         else:
    #             remark = " "    # whitespace to trigger LONG PATH when new fragment is requested
    #         # LONG PATH if SHORT PATH triggered a non-empty human remark
    #         while len(query) > 0 and len(remark) > 0:   # LONG PATH: Must learn fragment
    #                 fragment = await optimus.learn(instructions=fragment, objective=query+remark)
    #                 # new_messages, response = optimus.apply(instructions=fragment, objective=query+remark)
    #                 remark = optimus.hitl(prompt="Tell Optimus how to do better (Empty ENTER if satisfied)")
    #         # FINAL STEP: memorize new fragment
    #         optimus.memorize(fragment_name=fragment_name, fragment=fragment)
    #     await queue.put(None)

    # async def optimus_stream(queue: asyncio.Queue):
    #     """
    #     Consumer of optimus messages
    #     """
    #     while True:
    #         text = await queue.get()
    #         if text is None:
    #             break
    #         print(f"{text}", end="", flush=True)
    #     print("Consumer is done")

    # await asyncio.gather(optimus_stream(queue), optimus_loop(queue))

asyncio.run(main())
# ORIGINAL, SIMPLIFIED VERSION
# while True:
#     # user_input = input("> ").lstrip()

#     # 0. find memory fragment
#     # 1. IF no memory fragment, learn(), ELSE apply()
#     # 2. AFTER learn() ALWAYS apply()
#     # 3. AFTER bad apply() ALWAYS learn()
#     # 4. AFTER good apply() ALWAYS memorize()
#     # 5. AFTER memorize() ALWAYS back to sleep
#     # VARIANT: after hitl(), apply()+binary_hitl() to check if current fragment is enough to predict outcome, else learn()
#     human_in_the_loop = optimus.hitl(history=[], response="")
#     fragment_name, fragment = optimus.find_fragment(human_in_the_loop)
#     while fragment is None or len(human_in_the_loop) > 0:
#         fragment = optimus.learn(instructions=fragment, objective=human_in_the_loop)
#         new_messages, response = optimus.apply(instructions=fragment, objective=human_in_the_loop)
#         human_in_the_loop = optimus.hitl(history=new_messages, response=response)
#     optimus.memorize(fragment_name=fragment_name, fragment=fragment)

# def openai_wrapper():
#     from fastapi import FastAPI
#     from fastapi_openai_compat import CompletionResult, MessageParam, create_chat_completion_router
#     from collections.abc import Generator

#     def list_models() -> list[str]:
#         return ["local"]
    
#     def run_completion(model: str, messages: list[MessageParam], body: dict) -> CompletionResult:
#         last_msg = messages[-1]["content"]

#         if body.get("stream", False):
#             def stream() -> Generator[str, None, None]:
#                 for word in last_msg.split():
#                     yield word + " "
#             return stream()

#         return ChatCompletion(
#             id="resp-1",
#             object="chat.completion",
#             created=int(time.time()),
#             model=model,
#             choices=[
#                 Choice(
#                     index=0,
#                     message=Message(role="assistant", content="Hello!"),
#                     finish_reason="stop",
#                 )
#             ],
#             usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
#         )
#         # return f"You said: {last_msg}"

#     app = FastAPI()
#     router = create_chat_completion_router(
#         list_models=list_models,
#         run_completion=run_completion,
#     )
#     app.include_router(router)