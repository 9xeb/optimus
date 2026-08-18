import json
import logging
import os

from gepa.optimize_anything import optimize_anything, GEPAConfig, EngineConfig, ReflectionConfig, TrackingConfig
from gepa.utils import NoImprovementStopper, ScoreThresholdStopper  
from pydantic_ai import Tool
from pydantic_ai.models.function import _estimate_usage
from alive_progress import alive_bar

from src.agent import AgentWrapper
from src.utils import initialize_openai_client, stabilize_json
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
    def __init__(self, debug: bool = False):
        self.concision_clause = "(Avoid repetitions and verbose instructions. Minimize additional assumptions. Do not include information that you would be able to infer naturally.)"
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
    def optimize(self, objective: str, seed: dict[str,str]):
        """
        Given an objective and a starting prompt draft, try to improve the draft.
        Improvement is guided by GEPA's optimize_anything evolutionary algorithm, specialized for AI agent instructions.

        Args:
            objective: a string to steer the evolution
            seed: a dictionary containing the title and content of the starting prompt draft
        
        Returns:
            A tuple containing the improved draft and its score.
        """
        seed_name = seed["name"]
        seed_text = seed["prompt"]
        # objective = objective + ".\n" + self.concision_clause
        titles = {
            "optimus": "[OPTIMUS] - Preparing experiments...",
            "evaluator": "[EVALUATOR] - Simulating execution path...",
            "reflection": "[REFLECTION] - Refining instructions...",
            "judge": "[JUDGE] - Judging execution path...",
            "merger": "[MERGER] - Merging instructions..."
        }
        self.best_score = 0
    
        # Wrap inside the progress bar for pretty printing
        with alive_bar(
            total=100,
            dual_line=True,
            manual=True,
            title_length=max([len(titles[title]) for title in titles])
        ) as progress_bar:
            progress_bar.title(titles["optimus"])
            # def ask_human_expert(message: str):
            #     """
            #     Ask a human expert that provided the PREMISE, for help in assessing parts of the PROPOSAL.

            #     Args:
            #         message: message for the human expert
            #     """
            #     return input(f"{message}\n")
            # def ask_user_question(question: str):
            #     """
            #     Ask a question to the user. Always use this instead of stopping to ask the question. 

            #     Args:
            #         question: message for the human expert
            #     """
            #     return input(f"#####\n{question}\n#####\n> ")

            def propose_configuration(candidate, reflective_dataset, components_to_update, *, metadata=None) -> dict[str, str]:
                """
                Custom reflection callback. The only way to exert control over what comes out in the candidates.
                """
                log_error("CUSTOM REFLECTION")
                new_candidate = {}
                for component in components_to_update:
                    simulations = ""
                    for record in reflective_dataset[component]:
                        # objective = record["objective"]
                        fix = record["fix_this_in_instructions"]

                        merger_agent = AgentWrapper(name="MERGER", enable_mcp_toolsets=True, simulated=True)
                        progress_bar.title(titles["merger"])
                        merged_candidate = merger_agent.merge(current=candidate, inbound=fix)      # merge old candidate with fix
                        progress_bar.title(titles["evaluator"])
                        evaluator_agent = AgentWrapper(name="EVALUATOR", enable_mcp_toolsets=True, simulated=True)
                        _, simulation = evaluator_agent.simulate(merged_candidate)
                        simulations += simulation
                    new_candidate[component] = simulations
                return new_candidate

            def evaluate_configuration(candidate: str):
                """
                This is the local evaluation function used by GEPA to assess and score candidates proposed by the Reflection.
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

                try:
                    # # 1. Evaluate the candidate via simulated execution paths
                    # # progress_bar.title(titles["evaluator"])     # this changes progress bar title
                    # evaluator_agent = AgentWrapper(
                    #     name="EVALUATOR",
                    #     enable_mcp_toolsets=True,
                    #     simulated=True,
                    #     debug=self.debug
                    # )  # with access MCP toolsets for performing operations
                    # evaluation_new_messages, evaluation_result = evaluator_agent.simulate(prompt=candidate)
                    # evaluation_token_usage = _estimate_usage(evaluation_new_messages).input_tokens + _estimate_usage(evaluation_new_messages).output_tokens
                    # if evaluation_token_usage > self.max_context_segment:
                    #     self.max_context_segment = evaluation_token_usage
                    # self.token_count += evaluation_token_usage
                    # # if self.debug:
                    # #     log_response(f"[EVALUATOR] - {evaluation_result}")
                    # # else:
                    # #     log_response(f"[EVALUATOR] - {evaluation_result[:50]}...(more)")
                    # #     # log_response(f"[EVALUATOR] - {evaluation_result}")

                    # 2. Judge the quality of the simulated execution path against the original objective
                    # This judgement supposes the problem is in the reflection and the evaluator is right
                    progress_bar.title(titles["judge"])
                    judge_agent = AgentWrapper(name="JUDGE", enable_mcp_toolsets=True, simulated=True, debug=self.debug)
                    judge_new_messages, feedback = judge_agent.judge_reflection(
                        # instructions=objective,
                        # instructions=candidate,
                        # outcome=evaluation_result
                        instructions=candidate,
                        outcome=objective
                    )
                    # This judgement supposes the problem is in the evaluator's interpretation
                    # judge_new_messages, feedback = judge_agent.judge_evaluation(
                    #     premise=objective,
                    #     proposal=candidate
                    #     # proposal=evaluation_result
                    # )
                    judge_token_usage = _estimate_usage(judge_new_messages)
                    self.token_count += judge_token_usage.input_tokens + judge_token_usage.output_tokens
                    # 2.1. Stabilize JSON feedback from LLM
                    stable_feedback = stabilize_json(
                        unstable_string = feedback,
                        expected_keys = ["critique"]
                    )
                    score = float(stable_feedback["score"])
                    # if self.debug:
                    #     log_response(f"[JUDGE] - {stable_feedback["critique"]}")
                    # else:
                    #     log_response(f"[JUDGE] - {stable_feedback["critique"][:50]}...(more)")
                    # log_response(f"[JUDGE] - {score}/100.0")

                    # # 2 (alt). Human is the judge in the loop (HITL) - nudges tool call chains during learning
                    # with progress_bar.pause():
                    #     stable_feedback = {}
                    #     stable_feedback["critique"] = input("NUDGE > ")
                    #     score = float(input("SCORE (0-100) > "))
                    
                    # 3. Return score and ASI (feedback)
                    if score > self.best_score:
                        self.best_score = score
                        progress_bar(score/100.0)       # Update progress bar
                    progress_bar.title(titles["reflection"])
                    return score, {
                        "scores": {
                            "score": score
                        },
                        # "original_objective": objective,                            # remind reflection what the objective is
                        "objective": f"Refine current AI Agent instructions to solve the class of problems incorporating the following information: {objective}",
                        "fix_this_in_instructions": stable_feedback["critique"],    # nudge reflection with judge critique
                        # "remark": "When fixing instructions, do not lose previous information. Merge instead of replace."
                    }
                except Exception as e:
                    # In case of exceptions (broken JSONs, unreachable APIs, ...) fallback to a low score
                    log_error(f"CRITICAL ERROR - GEPA STEP FAILED - {e}")
                    return 0.0, {
                        "scores": {
                            "score": 0.0
                        },
                        "artifact": candidate,
                        "feedback": f"Provided JSON is malformed. Hint to fix: {e}"
                    }

            # This is the actual optimization call, that uses the above local evaluation function
            # progress_bar.title(titles["merger"])
            merger_agent = AgentWrapper(name="MERGER", debug=self.debug)
            # _, merged_seed = merger_agent.merge(current=seed_text, inbound=objective)
            # if not seed_text:   # optimize with gepa only if seed is new
            gepa_results = self.opinionated_optimize_anything(
                reflection=propose_configuration,
                evaluator=evaluate_configuration,
                objective=f"Refine current AI Agent instructions to solve the class of problems incorporating the following information: {objective}",
                seed_candidate=seed_text,
                # seed_candidate=merged_seed,   # pre-merge user prompt into seed
            )
            best_candidate = gepa_results.best_candidate
            best_score = gepa_results.val_aggregate_scores[gepa_results.best_idx]
            # else:
            #     # Quick shortcut to naive merge the objective into the existing seed
            #     best_candidate = objective
            #     best_score = 100

            # The following step is crucial to mitigate information loss in repeated optimizations
            if seed_text:
                # If previous seed existed, merge seed and GEPA optimization results
                progress_bar.title(titles["merger"])
                # agent = AgentWrapper(debug=self.debug)
                _, merger = merger_agent.merge(current=seed_text, inbound=best_candidate)
                return merger, best_score
            else:
                # We get here if there was no original seed, just return the optimized prompt as is.
                return best_candidate, best_score

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
                    max_metric_calls=500,                        # Safety limit to avoid infinite loops
                    parallel=False,
                    cache_evaluation=True,                       # Reuse redundant evaluations
                    raise_on_exception=True,                     # Continue on errors instead of stopping
                    # run_dir=self.run_dir,                        # Persistent GEPA state
                    display_progress_bar=False,
                ),
                reflection=ReflectionConfig(
                    reflection_lm=self.model_string,
                    reflection_minibatch_size=1,      # Reduce from default 3 to lower memory usage
                    perfect_score=100.0,
                    skip_perfect_score=True,          # Skip unnecessary evaluations
                    custom_candidate_proposer=reflection
                ),
                stop_callbacks=[
                    NoImprovementStopper(max_iterations_without_improvement=2),   # Stop when plateau
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
