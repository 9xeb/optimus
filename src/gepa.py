import json
import logging
import os

from gepa.optimize_anything import optimize_anything, GEPAConfig, EngineConfig, ReflectionConfig, TrackingConfig
from gepa.utils import NoImprovementStopper, ScoreThresholdStopper
from pydantic_ai import Tool
# from pydantic_ai.models.function import _estimate_usage
from alive_progress import alive_bar

from src.agent import AgentWrapper
from src.utils import TITLES, count_tokens, initialize_openai_client, stabilize_json
from src.log import log_error, log_internal_event, log_request, log_response, log_tool_request, log_tool_response

# Suppress verbose LiteLLM logging
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)

class GepaWrapper:
    """
    Class that incorporates GEPA prompt optimization methods.
    It is fundamentally a three agent system where:
        1. Reflection refines LLM instruction candidates from feedback
        2. Evaluator simulates execution paths with LLM instructions
        3. Judge comments on execution paths and sends back feedback to Reflection
    The system goes on until a plateau is reached (did not increase feedback score after N tries), or a score of 100 is reached.
    """
    def __init__(self, progress_bar, debug: bool = False):
        self.progress_bar = progress_bar
        # self.concision_clause = "(Avoid repetitions and verbose instructions. Minimize additional assumptions. Do not include information that you would be able to infer naturally.)"
        self.concision_clause = """
        # INSTRUCTIONS OUTPUT CONSTRAINTS
        AI Agent instructions must be concise to the extreme. Break syntax and grammar, only pure operational focus.
        """
        # AI Agent instructions must be concise to the extreme. Break syntax and grammar, only pure operational pseudo-code.

        # self.concision_clause = "(Minimize output token usage while maintaining maximum information density. Convey the same amount of information in the most dense language.)"
        # self.concision_clause = "(Convey the same amount of information in the most dense language possible.)"
        # Your AI Agent instructions must be efficient. Cut all filler, keep substance.
        #     - Drop articles (a, an, the), filler (just, really, basically, actually).
        #     - Drop pleasantries (sure, certainly, happy to).
        #     - No hedging. Fragments fine. Short synonyms.
        self.model_string = os.environ["OPENAI_API_MODEL"]
        self.agent_cache = {}
        self.debug = debug
        self.client = initialize_openai_client(self.model_string)
        self.token_count = 0
        self.best_score = 0
        self.max_context_segment = 0

    # def steer(self, criteria: list[str]):
    def optimize(self, objective: str, seed: str):
        """
        Given an objective and a starting prompt draft, try to improve the draft.
        Improvement is guided by GEPA's optimize_anything evolutionary algorithm, specialized for AI agent instructions.

        Args:
            objective: a string to steer the evolution
            seed: a dictionary containing the title and content of the starting prompt draft

        Returns:
            A tuple containing the improved draft and its score.
        """
        raw_objective = objective
        objective = objective + ".\n" + self.concision_clause   # override global objective with concision clause
        self.best_score = 0

        self.progress_bar.title(TITLES["optimus"])
        def propose_configuration(candidate, reflective_dataset = None, components_to_update = None, *, metadata=None) -> dict[str, str]:
            """
            Custom reflection callback. The only way to exert control over what comes out of the candidates.
            """
            try:
                new_candidate = {}
                # log_error(f"Components: {components_to_update}")
                # reflective_dataset["current_candidate"]
                for component in components_to_update:
                    new_candidate[component] = ""
                    # log_error(f"COMPONENT: {component}")
                    for record in reflective_dataset[component]:
                        # log_error(f"Record: {record}")
                        # log_error(f"{candidate}")
                        # objective = record["objective"]
                        fix = record.get("fix_this_in_instructions")

                        # 1. Merge previous candidate with fix from judge
                        reflection_agent = AgentWrapper(name="REFLECTION")
                        self.progress_bar.title(TITLES["reflection"])
                        merger_new_messages, merged_candidate = reflection_agent.merge( # merge old candidate with fix
                            current=candidate["current_candidate"],
                            inbound=objective+"\n"+fix    # Fix reminds of the objective: "generate AI agent instructions"
                        )
                        self.token_count += count_tokens(merger_new_messages)
                        new_candidate[component] += merged_candidate

                        # # # 2. Simulated execution path on candidate+fix is the new candidate, so it must be done in reflection instead of evaluation
                        # self.progress_bar.title(TITLES["evaluator"])
                        # evaluator_agent = AgentWrapper(
                        #     name="EVALUATOR",
                        #     enable_mcp_toolsets=True,
                        #     # simulated=True
                        # )
                        # simulation_new_messages, simulation = evaluator_agent.simulate(merged_candidate)
                        # self.token_count += count_tokens(simulation_new_messages)

                        # 3. Compact simulated execution path
                        # self.progress_bar.title(TITLES["compactor"])
                        # compactor_agent = AgentWrapper(name="COMPACTOR")
                        # compaction_new_messages, compacted = compactor_agent.compact(merged_candidate)
                        # self.token_count += count_tokens(compaction_new_messages)

                        # merges += compacted
            except Exception as e:
                log_error(f"REFLECTION ERROR - {e}")
                return {"current_candidate": "Generic instructions."}
            return new_candidate

        def evaluate_configuration(candidate: str):
            """
            Evaluation callback used by GEPA to assess and score candidates proposed by the Reflection.
            Candidate prompts are simulated in execution paths by the Evaluator, and the execution is judged and scored.

            Args:
                candidate: a string coming from the Reflection

            Returns:
                A score between 0 and 100, and the Judge's feedback to steer the Reflection.
            """
            # if self.debug:
            #     log_request(f"[REFLECTION] - {candidate}")
            # else:
            #     log_request(f"[REFLECTION] - {candidate[:50]}...(more)")
            # log_response(f"#######\n{candidate}########")

            try:
                # # 1. Assess candidate+fix in the environment. Find what went wrong
                # self.progress_bar.title(TITLES["evaluator"])
                # evaluator_agent = AgentWrapper(
                #     name="EVALUATOR",
                #     enable_mcp_toolsets=True,
                #     # simulated=True
                # )
                # assesment_new_messages, assessment = evaluator_agent.assess(system_prompt=candidate, prompt=objective)
                # self.token_count += count_tokens(assesment_new_messages)
                agent = AgentWrapper(name="AGENT", enable_mcp_toolsets=True)
                agent_new_messages, _ = agent.act(system_prompt=candidate, prompt=objective)
                _, agent_recap = agent.recap(agent_new_messages)

                # 2. Judge the quality of the simulated execution path against the original objective
                # This judgement supposes the problem is in the reflection and the evaluator is right
                self.progress_bar.title(TITLES["judge"])
                judge_agent = AgentWrapper(name="JUDGE", enable_mcp_toolsets=True, simulated=True, debug=self.debug)
                judge_new_messages, feedback = judge_agent.judge_reflection(
                    # instructions=candidate,
                    # outcome=objective
                    instructions=agent_recap,
                    outcome=raw_objective
                )
                # This judgement supposes the problem is in the evaluator's interpretation
                # judge_new_messages, feedback = judge_agent.judge_evaluation(
                #     premise=objective,
                #     proposal=agent_recap
                # )
                self.token_count += count_tokens(judge_new_messages)
                # 1.1. Stabilize JSON feedback from LLM
                # stable_feedback = stabilize_json(
                #     unstable_string = feedback,
                #     expected_keys = ["fix"]
                # )
                # score = float(stable_feedback["score"])
                # if self.debug:
                #     log_response(f"[JUDGE] - {stable_feedback["critique"]}")
                # else:
                #     log_response(f"[JUDGE] - {stable_feedback["critique"][:50]}...(more)")
                # log_response(f"[JUDGE] - {score}/100.0")

                # # # 2 (alt). Human is the judge in the loop (HITL) - nudges tool call chains during learning
                # with self.progress_bar.pause():
                #     hitl = input("(Press Enter to confirm judgment, or write something) > ")
                #     stable_feedback["critique"] = hitl
                #     if hitl != "\n":
                #         score = float(input("Override score (0-100) > "))

                # 3. Return score and ASI (feedback)
                score = float(feedback["score"])
                if score > self.best_score:
                    self.best_score = score
                    self.progress_bar(score/100.0)       # Update progress bar
                self.progress_bar.title(TITLES["reflection"])
                return score, {
                    "scores": {
                        "score": score
                    },
                    # "original_objective": objective,                            # remind reflection what the objective is
                    "objective": f"Refine current AI Agent instructions to solve the class of problems incorporating the following information: {objective}",
                    "fix_this_in_instructions": feedback["fix"],    # nudge reflection with judge critique
                    # "remark": "When fixing instructions, do not lose previous information. Merge instead of replace."
                }
            except Exception as e:
                # In case of exceptions (broken JSONs, unreachable APIs, ...) fallback to a low score
                log_error(f"CRITICAL ERROR - GEPA EVALUATE FAILED - {e}")
                return 0.0, {
                    "scores": {
                        "score": 0.0
                    },
                    "artifact": candidate,
                    "fix_this_in_instructions": f"Provided JSON is malformed. Hint to fix: {e}"
                }

        # 1. Optimize SEED to incorporate OBJECTIVE
        # if seed is None:
        # Optimization from scratch with GEPA
        gepa_results = self.opinionated_optimize_anything(
            reflection=propose_configuration,
            evaluator=evaluate_configuration,
            objective=f"Refine current AI Agent instructions to solve the class of problems incorporating the following information: {objective}",
            seed_candidate=seed,       # If seed is 100% good then the objective is known in memory and GEPA is mostly skipped
        )
        best_candidate = gepa_results.best_candidate
        best_score = gepa_results.val_aggregate_scores[gepa_results.best_idx]
        # else:
        #     # Optimization from existing fragment with MERGE
        #     best_candidate = seed
        #     best_score = 100

        # 2. Merge SEED and OPTIMIZED_SEED to mitigate information loss
        self.progress_bar.title(TITLES["merger"])
        merger_agent = AgentWrapper(name="MERGER", debug=self.debug)
        merger_new_messages, merger = merger_agent.merge(current=seed, inbound=best_candidate)
        self.token_count += count_tokens(merger_new_messages)
        return merger, best_score

    def opinionated_optimize_anything(
        self,
        reflection,
        evaluator,
        objective: str,
        seed_candidate: str = None,
        # criteria: list = None
    ):
        """
        Run a GEPA optimize_anything function with opinionated configuration (safe limits, stop on plateaus, stop on 100 score, ...).
        A generator of candidates and a judge collaborate to iterate artifact designs towards optimization goals.

        Who's who in GEPA:
            - REFLECTION (inside GEPA) -> objective + current candidate + feedback = new candidate
            - JUDGE (custom evaluator function) -> new candidate + criteria = feedback

        In the background, an evolutionary selection algorithm, run by GEPA internals, keeps a pool of best candidates.

        Args:
            evaluator: custom function to judge candidates
            objective: tells the REFLECTION what the optimization should achieve
            criteria: (optional) nudges consumed by EVALUATOR
            seed_candidate: (optional) starting artifact
        """

        class QuietLogger:
            """
            The minimum viable class to suppress verbose logs coming out of GEPA loops.
            """
            def log(self, message: str):
                # if "Proposed new text for current_candidate:" not in message:
                #     print(f"[GEPA] - {message}")
                pass  # swallow all messages

        # This is the core of the optimization capabilities of Optimus. A very curated GEPA configuration.
        return optimize_anything(
            evaluator=evaluator,
            # dataset=criteria if criteria is not None else [],
            # valset=criteria,                      # test against this set of criteria
            objective=objective,
            seed_candidate=seed_candidate,
            config=GEPAConfig(
                tracking=TrackingConfig(logger=QuietLogger()),
                engine=EngineConfig(
                    frontier_type="instance",
                    # candidate_selection_strategy="current_best", # Always select the candidate with the best score
                    max_metric_calls=500,                          # Safety limit to avoid infinite loops
                    parallel=False,
                    cache_evaluation=True,                         # Reuse redundant evaluations
                    raise_on_exception=True,                       # Continue on errors instead of stopping
                    # run_dir=self.run_dir,                        # Persistent GEPA state
                    display_progress_bar=False,
                ),
                reflection=ReflectionConfig(
                    reflection_lm=self.model_string,
                    reflection_minibatch_size=1,            # Reduce from default 3 to lower memory usage
                    perfect_score=90.0,                     # Empirically, when judge >90 there are often no fixes
                    skip_perfect_score=True,                # Skip unnecessary evaluations
                    custom_candidate_proposer=reflection    # Custom reflection function
                ),
                stop_callbacks=[
                    NoImprovementStopper(max_iterations_without_improvement=5),   # Stop when plateau
                    ScoreThresholdStopper(100)                                    # Stop when perfect
                ]
            )
        )

    # def human_gepa_prototyping():
    #     """
    #     Take the best from gepa_optimize_artifact and add human in the loop.
    #     Spec driven exploration and function optimization over billions of dimensions.
    #     Ideally, any kind of problem is reduced to a human_gepa_prototyping process, not only coding but also operations.
    #     Pair with LLM-simulated tool responses of Agent Compose for maximum effect without any kind of real system interaction. Fully virtual.

    #     gepa scopes on chat sessions
    #         <-> run scopes with simulated tools
    #         <-> gepa scopes with run scopes logs as criteria to spot errors and missing scopes
    #     """
    #     # Run 1: Quick exploration
    #     result1 = optimize_anything(
    #         seed_candidate=initial_design,
    #         evaluator=evaluator,
    #         config=GEPAConfig(engine=EngineConfig(max_metric_calls=50, run_dir="./run1")),
    #     )

    #     # Human adjusts based on result1.best_candidate
    #     adjusted_candidate = human_adjustment(result1.best_candidate)

    #     # Run 2: Deeper optimization with higher budget
    #     result2 = optimize_anything(
    #         seed_candidate=adjusted_candidate,
    #         evaluator=evaluator,
    #         config=GEPAConfig(engine=EngineConfig(max_metric_calls=200, run_dir="./run2")),
    #     )
