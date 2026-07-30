import json
import logging
import os

from gepa.optimize_anything import optimize_anything, GEPAConfig, EngineConfig, ReflectionConfig
from gepa.utils import NoImprovementStopper, ScoreThresholdStopper  
from pydantic_ai import Tool
from pydantic_ai.models.function import _estimate_usage

from src.agent import AgentWrapper
from src.utils import initialize_openai_client, stabilize_json
from src.log import log_error, log_internal_event, log_request, log_response

# Suppress verbose LiteLLM logging
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)

class GepaWrapper:
    def __init__(self, objective: str, seed: str = None, debug: bool = False):
        self.concision_clause = """
        WARNING: generate just the amount of words necessary. Avoid repetitions and verbose instructions. Do not include information that you would otherwise be able to infer.
        """
        # Model string in format 'openai/unsloth/gemma-4-E4B-it-GGUF:Q4_K_M'
        # self.model_string = model_string
        self.model_string = os.environ["OPENAI_API_MODEL"]
        self.objective = objective + "\n" + self.concision_clause
        self.seed_candidate = seed
        self.agent_cache = {}
        self.debug = debug
        self.client = initialize_openai_client(self.model_string)
        # self.run_dir = './.optimus'         # Where GEPA state will be stored

        self.token_count = 0

    # def steer(self, criteria: list[str]):
    def optimize(self):
        """
        Apply criteria to artifact via evolutionary algorithms.
        Evolution is powered by GEPA's optimize_anything function.
        """

        def ask_question(question: str):
            """
            Always ask questions when the proposed solution made assumptions that are not clear in the objective.
            The response to the question is authoritative and can steer the feedback.
            """
            return input(f"{question}\n")

        def evaluate_configuration(candidate: str):
            log_request(f"[REFLECTION] - {candidate}")
            try:
                agent = AgentWrapper(lm=self.model_string, debug=self.debug)

                # 1. Deploy candidate in the environment and store results
                log_internal_event("[EVALUATOR] - Running draft...")
                evaluation_new_messages, evaluation_result, self.agent_cache = agent.step(
                    task="",
                    response_format="",
                    user_prompt=candidate,
                    # tools=[],
                    # debug=self.debug,
                    cache=self.agent_cache
                )
                evaluation_token_usage = _estimate_usage(evaluation_new_messages)
                self.token_count += evaluation_token_usage.input_tokens + evaluation_token_usage.output_tokens
                log_response(f"[EVALUATOR] - RESULTS - {evaluation_result[:50]}...(more)")
                evaluation_tool_calls = agent.list_tool_calls(evaluation_new_messages)
                log_response(f"[EVALUATOR] - CALLS - {json.dumps(evaluation_tool_calls)}]")

                # 2. Judge the result of a candidate
                log_internal_event("[JUDGE] - Judging draft evaluation...")
                judge_new_messages, feedback, self.agent_cache = agent.step(
                    task="Criticize the proposed solution attempt. Focus on missing or wrong points that violate the goal.",
                    # task="Grill the proposed solution attempt.",
                    response_format="""
                    Output format in JSON:
                    {{"critique": "...", "score": "between 0 and 100"}}
                    Return ONLY valid JSON.
                    Escape all quotes inside string values.
                    Escape all backslashes.
                    Do not include markdown fences.
                    """,
                    user_prompt=f"""
                    # GOAL
                    {self.objective}

                    # SOLUTION ATTEMPT
                    ## TOOLS CALLED
                    {json.dumps(evaluation_tool_calls)}
                    ## RESPONSE
                    {evaluation_result}
                    """,
                    tools=[
                        Tool(ask_question, takes_ctx=False, metadata={'read_only': True})
                    ],
                    # declarative_tools={},
                    # toolsets={},
                    # history=past_history,
                    # debug=self.debug,
                    cache=self.agent_cache,
                    # Use needs and tools to hit in cache same condition checking with same context, efficiently
                    # metadata={"needs": procedure.needs, "tools": procedure.tools}
                )
                judge_token_usage = _estimate_usage(judge_new_messages)
                self.token_count += judge_token_usage.input_tokens + judge_token_usage.output_tokens
                log_response(f"[JUDGE] - {feedback}")

                # 3. Stabilize JSON feedback from LLM
                stable_feedback = stabilize_json(
                    unstable_string = feedback,
                    # expected_keys = ["score_explanation"]
                    # expected_keys = ["pros","cons"]
                    expected_keys = ["critique"]
                )
                score = float(stable_feedback["score"])
                log_internal_event(f"CONFIRMED CANDIDATE SCORE - {score}")
                # 4. Return score and ASI
                return score, {
                    "scores": {
                        "score": score
                    },
                    "artifact": candidate,
                    "feedback": stable_feedback
                }
            except Exception as e:
                log_error(f"CRITICAL ERROR - GEPA STEP FAILED - {e}")
                # raise e
                return 0.0, {
                    "scores": {
                        "score": 0.0
                    },
                    "artifact": candidate,
                    "feedback": f"Provided JSON is malformed. Hint to fix: {e}"
                }

        # criteria = [
        #     # TODO: expand criteria to run logs and human inputs, to achieve a continuous learning agent
        #     {"id": "New Information", "criteria": f"Rate the presence of the following information: {nudge}"},
        #     {"id": "Objective", "criteria": f"Rate the level of fulfillment of the following objective: {self.objective}"},
        # ]
        # if self.artifact != "...":
        #     # Add old info criteria for repated nudges
        #     criteria+=[{"id": "Old Information", "criteria": f"Rate the presense of the following information: {self.artifact}"}]

        # Optimize scopes configuration
        gepa_results = self.opinionated_optimize_anything(
            evaluator=evaluate_configuration,
            objective=f"Generate an LLM prompt to fulfill the following task: {self.objective}",
            seed_candidate=self.seed_candidate,
            # criteria=criteria,
            # seed=self.artifact
        )
        # self.artifact = gepa_results.best_candidate         # Update artifact
        return gepa_results.best_candidate

    def opinionated_optimize_anything(
        self,
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
            seed: (optional) starting artifact
        """
        return optimize_anything(
            # seed_candidate=seed,
            evaluator=evaluator,
            # dataset=criteria if criteria is not None else [],
            # valset=criteria,                      # test against this set of criteria
            objective=objective,
            seed_candidate=seed_candidate,
            config=GEPAConfig(
                engine=EngineConfig(
                    frontier_type="instance",
                    # candidate_selection_strategy="current_best", # Always select the candidate with the best score
                    max_metric_calls=500,                        # Safety limit to avoid infinite loops
                    parallel=False,
                    cache_evaluation=True,                       # Reuse redundant evaluations
                    raise_on_exception=True,                     # Continue on errors instead of stopping
                    # run_dir=self.run_dir,                        # Persistent GEPA state
                    # display_progress_bar=True,
                ),
                reflection=ReflectionConfig(
                    reflection_lm=self.model_string,
                    reflection_minibatch_size=1,      # Reduce from default 3 to lower memory usage
                    perfect_score=100.0,
                    skip_perfect_score=True           # Skip unnecessary evaluations
                ),
                stop_callbacks=[
                    NoImprovementStopper(max_iterations_without_improvement=10),  # Stop when plateau
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
