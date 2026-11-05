import asyncio
import datetime
from enum import Enum
import json
from pathlib import Path

import numpy
import dspy

from src.log import log_internal_event, log_error
from src.utils import stream_dspy_program, State, Task

class RCS(dspy.Module):
    """
    Context Segmentation in DSPy. Based on the paper submitted at RAISE 2026, with a few twists:
        1. give_up tool is replaced with self.user;
        2. run_strategy tool is replaced with self.agents, enabling more sophisticated architectures to spawn;
        3. async design, to integrate RCS into real world systems.

    As RCS works the problem, it updates a shared internal state and evolves agent architecture.
    """

    class ObserveTools(dict):
        """
        Class to mutate how dspy.ReAct handles wrong tool names.
        By default, dspy.ReAct crashes the agent.
        This lets weaker LLMs observe their mistakes.
        """
        def __missing__(self, name):
            log_internal_event(f"[RCS] LLM called wrong tool: {name}.")
            raise KeyError(
                f"Unknown direct tool '{name}'. Use `run_tools` "
                f"to invoke tools."
            )

    def __init__(self, state: State, tools: list[dspy.Tool] = None, info: list = None):
        super().__init__()

        self.info = info if info else {}    # additional info to always pass to agents
        self.external_tools = tools if tools else []
        self.external_tools_dict = {
            tool.name: tool
            for tool in tools
        }
        self.state = state
        self.events = {}            # dictionary of events
        # self.transferable_lessons = []

        log_internal_event(f"[RCS] Building standard signature")
        self.standard_signature = "info: list, task: dict -> final_result"
        log_internal_event(f"[RCS] Building explorer module")
        self.explore = dspy.ReAct(
            signature=self.standard_signature,
            tools=[dspy.Tool(self.agents), dspy.Tool(self.user)]
        )

        # TODO: decision model loop
        # self.agents = DeAct(
        #     # Signature is implicitly the task
        #     tools=self.external_tools
        # )

        self.state.parent = None     # Task focus, used to connect dependencies
        self.awakened_list = []

    # def forward(self, task, **kwargs):
    #     """
    #     Expand on explorer-worker segmentation with knowledge_base
    #     """
    #     exploration = self.agents(
    #         tasks=[Task(title=task, task=task, kind="root", dependencies=[], ttl=60)]
    #     )
    #     log_internal_event(f"Sync NOT Stored answer to: {json.dumps([task], indent=4, default=str)} in dataset.")
    #     return dspy.Prediction(report=exploration[0]["report"])

    async def aforward(self, task, **kwargs):
        """
        Run RCS program asynchronously.

        1. Re-Route to existing task if present.
        2. Grow new root task otherwise.
        3. When the root task is done, waits for user review in self.state
        4. If review is not empty, grow a new root task dependent on the previous root task        
        """
        try:
            self.awakened_list = []     # reset working memry
            previous_task = None
            while task:
                log_internal_event(f"[RCS] Calling agent for [{task}]")
                log_error(f"[RCS] Available tasks: {list(self.state.tasks.keys())}")

                # First, decide to either reuse a task or grow a new one
                if task in list(self.state.tasks.keys()):
                    # Case 1: reuse known task
                    log_internal_event(f"[RCS] Picking known task: [{task}].")

                    # Reuse task if present (can be non-root)
                    task_payload = Task(
                        title=task,
                        task=self.state.tasks[task]["task"],
                        kind=self.state.tasks[task]["kind"],
                        # only reuse dependencies that are not expired, unless they are a root task
                        dependencies=[
                            dependency
                            for dependency in self.state.tasks[task]["dependencies"]
                            if not self.is_task_expired(dependency) and self.state.tasks[dependency]["kind"] != "root"
                        ],
                        ttl=self.state.tasks[task]["ttl"]
                    )
                else:
                    # Case 2: create a new root task for unseen problem
                    log_error(f"[RCS] Growing new root task: [{task}]")
                    task_payload = Task(
                        title=task,
                        task=task,
                        kind="root",
                        # attach to previous root task to allow pivoting
                        dependencies=[] if previous_task is None else [previous_task],
                        ttl=60
                    )
                    self.state.parent = task_payload

                # Second, actually run the task via the recursive agents interface
                exploration = await self.agents(
                    tasks=[task_payload]
                )
                await self.state.tokens.put(None) # signal end of CtxSeg loop
                self.state.parent = None   # reset focus

                # Third, trigger a situation and wait for user feedback
                self.state.situation = {
                    "type": "review",
                    "task": task
                }
                log_internal_event(f"[SITUATION] {self.state.situation}")
                await self.state.event.wait()
                self.state.situation = {}      # clear situation so backend knows
                previous_task = task
                task = self.state.resolution
                self.state.event.clear()

            # Finally, save memories to file
            if self.state.filepath:
                with open(self.state.filepath, 'w', encoding='utf-8') as f:
                    f.write(json.dumps({"tasks": self.state.tasks, "qna": self.state.qna}, indent=4))
            log_internal_event(f"[RCS] Saved tasks to file {self.state.filepath}")

            # And return last report to caller
            return dspy.Prediction(report=exploration[0]["report"])
        except Exception as e:
            log_error(f"[RCS] Critical aforward error: {e}")
            return dspy.Prediction(report="Agents crashed")
    async def agents(self, tasks: list[Task]):
        """
        Run the most conservative minimal immediate best next sub-tasks architecture, given what is known so far.
        If presented with partial information, run an equally partial architecture, since you can always run the next when more info arrives.

        Example call:
            next_tool_args = {"tasks": [{"title": "...", "task": "...", "kind": "...", "dependencies": []}]}
        """

        # How this works:
        # 1. Sanity checks (ghost dependencies, broken names, ...)
        # 2. Populate 'info' for RAG
        # 3. Run task with appropriate dspy program (complex or simple)

        try:
            available_tasks = [
                key for key, value in self.state.tasks.items()
            ] + [task.title for task in tasks]
            ghost_dependencies = []
            # 1. Sanity checks (LLM can observe these)
            for task in tasks:
                # Redundancy check, a.k.a. prevent re-calling the same task
                log_internal_event(f"[RCS] Redundancy check (does not apply to root tasks)")
                if task.kind != "root" and self.state.parent is not None:
                    if task.title.lower() == self.state.parent.title.lower() and (not self.state.tasks.get(task.title) or not self.state.tasks[task.title].get("report")):
                        task_payload = {"title": task.title, "task": task.task}
                        return [{
                            "report": f"Redundancy Error: breaking down your task into the exact same task is not allowed: {task_payload}."
                        }]

                # Accumulate list of ghost dependencies
                ghost_dependencies += list(set(task.dependencies).difference(available_tasks))

            # Ghost dependences check
            log_internal_event(f"[RCS] Ghost dependencies check")
            if len(ghost_dependencies) > 0:
                log_error(f"Dependencies Error: the following requested task titles do not exist: {ghost_dependencies}.")
                return [{
                    "report": f"Dependencies Error: the following requested task titles do not exist: {ghost_dependencies}."
                }]

            # After sanity checks, register current tasks as dependencies of parent task
            log_internal_event(f"[RCS] Connecting dependencies for [{self.state.parent.title}]")
            for task in tasks:
                if task.kind != "root" and self.state.parent is not None:
                    # NOTE: the order of execution in dependencies list matters. That's how they came out of the LLM.
                    if task.title not in self.state.parent.dependencies:
                        # This is what makes future calls to this task cheaper: instead of going through reasoning again, reports are requested by network design.
                        log_internal_event(f"[{task.title}] becomes dependency of [{self.state.parent.title}]")
                        self.state.parent.dependencies += [task.title]


            # This rebuilds context and runs each task
            for task in tasks:
                log_internal_event(f"[AGENTS] Triggered task type {task.kind}")

                # 0. Attempt a cache hit on previously reported tasks
                if not self.is_task_expired(task.title):
                    log_internal_event(f"Cache hit for task [{task.title}]")
                    continue    # skip this task and trust the cache in self.state.tasks

                # 1. Recursively refresh shared state for dependencies
                await self.agents(
                    tasks=[
                        Task(
                            title=dependency,
                            task=self.state.tasks[dependency]["task"],
                            kind=self.state.tasks[dependency]["kind"],
                            ttl=self.state.tasks[dependency]["ttl"],
                            dependencies=self.state.tasks[dependency]["dependencies"]
                        )
                        for dependency in task.dependencies
                    ]
                )

                # 2. Route to module according to task's kind
                if task.kind.lower() in ["simple"]:
                    module = dspy.ReAct(
                        signature=self.standard_signature,
                        tools=self.external_tools + [dspy.Tool(self.user)]
                    )
                    # Allow the LM to emit any tool name and observe wrong tool names
                    module.react.signature.output_fields["next_tool_name"].annotation = str
                    # Preserve the compiled dict of tools, but let LLM see when tool name is wrong, instead of crashing
                    module.tools = self.ObserveTools(module.tools)
                else:
                    # agent capable of spawning other agents
                    module = self.explore
                log_internal_event(f"Awakened list: {self.awakened_list}")

                # 3. Build up info for RAG (includes working memory and examples)
                info = self.build_info(task.kind)

                # 4. Run task
                log_internal_event(f"[RCS] Running task")
                try:
                    # Ensure sane defaults for new task
                    if not self.state.tasks.get(task.title):
                        self.state.tasks[task.title] = {
                        }

                    # Wake up task
                    self.state.tasks[task.title] = self.state.tasks[task.title] | {
                        "kind": task.kind,
                        "task": task.task,
                        "dependencies": task.dependencies,
                        "ttl": task.ttl,
                        "state": "awake",
                    }

                    if task.kind in ["complex", "root"]:
                        # If task is not simple, sub-tasks can spawn, so we switch parent
                        previous = self.state.parent
                        log_internal_event(f"[RCS] Switching complex focus from [{previous}] to [{task}]")
                        self.state.parent = task       # switch parent

                    # Start the actual ReAct loop
                    execution = await self.recurse(
                        module=module,
                        task=task,
                        info=info,
                    )
                    if task.kind in ["complex", "root"]:
                        # Restore focus to parent complex task
                        log_internal_event(f"[RCS] Restoring complex focus from [{task}] to [{previous}]")
                        self.state.parent = previous   # restore previous focus

                    # 5. Update shared state and put task to sleep
                    log_internal_event(f"Putting [{task.title}] back to sleep and clearing discovered tools.")
                    self.state.discovered_tools = []    # clear list of discovered tools
                    self.state.tasks[task.title] = self.state.tasks[task.title] | {
                        "state": "sleeping",
                        "dependencies": task.dependencies,  # dependencies may update after execution
                        "report": execution.final_result,
                        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
                    }

                    # Update working memory index
                    self.awakened_list += [task.title]                        

                except (dspy.LMError, dspy.AdapterParseError) as e:
                    log_error(f"[AGENTS] [{task}] ERROR: {e}")
                    self.state.tasks[task.title] = self.state.tasks[task.title] | {"state": "failing"}
                    continue    # skip task update
                
        except Exception as e:
            log_error(f"[AGENTS] Critical error: {e}")
            raise e

        # Return reports in LLM-friendly format
        log_internal_event(f"Returning tasks: {tasks}")
        return [
            {"title": task.title} | self.state.tasks[task.title]
            for task in tasks
        ]

    async def recurse(self, module: dspy.Module, task: Task, info: list, retries: int = 3):
        """
        Run a dspy.module against a task with RAG info

        Return:
            a report on the task
        """
        try:
            log_payload = {"info": info, "task": task}
            log_internal_event(f"[RECURSE] {json.dumps(log_payload, indent=4, default=str)}")

            self.state.current = task
            # Stream tokens to specified asyncio queue
            return await stream_dspy_program(
                queues=[self.state.tokens],
                module=module,
                # task=task,
                task={"title": task.title, "task": task.task, "dependencies": task.dependencies},
                info=info
            )

            # TODO: implement self-review here, with self.user tool access to curate examples
        except (dspy.LMError, dspy.AdapterParseError) as e:
            log_error(f"DSPy LM Error: {e}")
            # Retry a bunch of times (LLM backend might be flaky)
            if retries > 0:
                await asyncio.sleep(3)
                # self.state.qna["How to call a tool from the list?"] = {
                #     "kind": self.state.current.kind,
                #     "content": "Tools must be wrapped around the 'run_tools' interface."
                # }
                # log_error(f"Updated qna: {json.dumps(self.state.qna, indent=4)}")

                # Retry failing agents
                return await self.recurse(
                    module=module,
                    task=task,
                    info=self.build_info(task.kind), # rebuild info after qna update
                    retries=retries-1
                )
            else:
                raise e         # raise it to self.agents


    async def user(self, questions: list[str]):
        """
        Escalate to the user for guidance when unable to fully complete the task.
        """
        try:
            answers = []
            for question in questions:
                # Trigger situation, await for user response
                self.state.situation = {"type": "question", "parent": self.state.current.title, "question": question}
                log_internal_event(f"[SITUATION] {self.state.situation}")
                await self.state.event.wait()
                self.state.situation = {}      # clear situation so backend knows
                answer = self.state.resolution
                self.state.event.clear()

                # Update examples dataset
                self.state.qna[question] = {"kind": self.state.current.kind, "content": "User: "+answer}
                log_internal_event(f"[RCS] {self.state.qna}")
                answers += [{"question": question, "answer": answer}]
        except Exception as e:
            log_error(f"User error: {e}")
            raise e
        return answers

    def build_info(self, kind: str):
        """Rebuild info for context"""
        log_internal_event(f"[RCS] Rebuilding info for RAG of task kind [{kind}]")
        info = [{"current_time": datetime.datetime.now(datetime.UTC).isoformat()}]
        info += [
            {
                "examples": [
                    {"question": question, "answer": answer["content"]}
                    for question,answer in self.state.qna.items()
                    if answer["kind"] == kind
                ]
            },
            {
                "working_memory": [
                    {
                        "title": awakened,
                        "final_results": self.state.tasks[awakened].get("report"),
                    }
                    for awakened in self.awakened_list
                ],
            } | self.info
        ]
        return info

    def is_task_expired(self, title: str):
        """Tell if a task is expired"""
        if self.state.tasks.get(title) and self.state.tasks[title].get("report"):
            # Determine expiration time of cached report
            expiration_time = (
                datetime.datetime.fromisoformat(self.state.tasks[title]["timestamp"]) +
                datetime.timedelta(minutes=self.state.tasks[title]["ttl"])
            )
            # If current date is not yet past expiration time, cache is valid and report generation is skipped for this turn
            if datetime.datetime.now(datetime.UTC) <= expiration_time:
                log_internal_event(f"[AGENTS] Cached [{title}] within TTL")
                return False
            else:
                log_internal_event(f"[AGENTS] Cached [{title}] TTL expired")
                return True
        else:
            return True

    # def chat(self):
    #     """
    #     Standard chat with DSPy
    #     """
    #     class QA(dspy.Signature):
    #         question: str = dspy.InputField()
    #         history: dspy.History = dspy.InputField()
    #         answer: str = dspy.OutputField()

    #     predict = dspy.Predict(QA)
    #     history = dspy.History(messages=[])
    #     while True:
    #         question = input("Type your question, end conversation by typing 'finish': ")
    #         if question == "finish":
    #             break
    #         outputs = predict(question=question, history=history)
    #         print(f"\n{outputs.answer}\n")
    #         history.messages.append({"question": question, **outputs})
