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
            log_internal_event(f"[RCS] LLM did not wrap tool: {name}. Letting it observe the mistake.")
            raise KeyError(
                f"You must wrap `{name}` with `run_tools`."
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
        self.standard_signature = "info: list, task: dict -> answer"
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
            # # Reset ephemeral dependencies
            for existing_task in list(self.state.tasks.keys()):
                if self.state.tasks[existing_task]["kind"] == "root":
                    log_internal_event(f"[RCS] Remove ephemeral dependencies of [{existing_task}]")
                    self.state.tasks[existing_task]["dependencies"] = [
                        dependency
                        for dependency in self.state.tasks[existing_task]["dependencies"]
                        if self.state.tasks.get(dependency) and self.state.tasks[dependency]["kind"] == "root"
                    ]
            log_internal_event("[RCS] Remove all ephemeral tasks")
            self.state.tasks = {
                key: value
                for key,value in self.state.tasks.items()
                if self.state.tasks[key]["kind"] == "root"
            }

            # # # # previous_root_tasks = []
            while task:
                self.awakened_list = []     # reset working memory
                # self.state.trajectory = []  # reset trajectory

                log_internal_event(f"[RCS] Calling agent for [{task}]")
                # First, decide to either reuse a task or grow a new one
                root_tasks = [
                    title
                    for title,task in self.state.tasks.items()
                    if task["kind"] == "root"
                ]
                if task in root_tasks:
                    # Case 1: reuse known task
                    log_internal_event(f"[RCS] Picking known root task: [{task}]")

                    # Rebuild task's payload
                    task_payload = Task(
                        title=task,
                        task=self.state.tasks[task]["task"],
                        kind=self.state.tasks[task]["kind"],
                        # dependencies=self.state.tasks[task]["dependencies"],
                        # only reuse dependencies that are not expired, unless they are a root task
                        # # # # dependencies=[
                        # # # #     dependency
                        # # # #     for dependency in self.state.tasks[task]["dependencies"]
                        # # # #     if self.state.tasks[dependency]["kind"] == "root"
                        # # # #     # if not self.is_task_expired(dependency) or self.state.tasks[dependency]["kind"] == "root"
                        # # # # ],
                        ttl=self.state.tasks[task]["ttl"]
                    )

                    # Refine the task's prompt if tier is zero (examples changed)
                    # Basically a hybrid optimization with human in the loop
                    # if self.state.tasks[task]["tier"] == 0:
                    if len(
                        self.state.tasks[task]["examples"] +
                        self.state.tasks[task]["tips"] +
                        self.state.tasks[task]["qna"]
                    ) > 0:
                        log_internal_event("[RCS] Triggered task instructions refinement.")
                        merge = dspy.ChainOfThought("old_task, updates -> updated_task")
                        merging = merge(old_task=task_payload.task, updates=self.build_examples(task_payload))
                        log_internal_event(f"[RCS] Merged [{task}] with user tips and QnA: [{merging.updated_task}]")
                        task_payload.task = merging.updated_task      # override task's prompt
                        log_internal_event("[RCS] Consuming user tips and agent questions")
                        self.state.tasks[task]["examples"] = [] # consume examples
                        self.state.tasks[task]["tips"] = []     # consume user tips
                        self.state.tasks[task]["qna"] = []      # consume agent questions
                else:
                    # Case 2: create a new root task for unseen problem
                    # log_error(f"[RCS] Growing new root task: [{task}]")
                    task_payload = Task(
                        title=task,
                        task=task,
                        kind="root",
                        # dependencies=[],
                        # attach to previous root task to allow pivoting
                        # # # # dependencies=[] if previous_root_task is None else [previous_root_task],
                        ttl=60
                    )

                # root tasks have self as parent (to build info coherently)
                self.state.parent = task_payload
                self.state.current = task_payload
                self.state.tasks[self.state.parent.title]["trajectory"] = []  # reset trajectory
                if self.state.tasks[self.state.parent.title].get("report"):
                    del self.state.tasks[self.state.parent.title]["report"]       # reset report
                # Second, actually run the task via the recursive agents interface
                exploration = await self.agents(
                    subtasks=[task_payload]
                )
                self.state.tasks[self.state.parent.title]["tier"] += 1

                # # # # if len(previous_root_tasks) > 0:
                # # # #     # Attach task to pivot for display purposes
                # # # #     FIX self.state.tasks[task]["dependencies"] = previous_root_tasks
                # # # #     INFO FLOW, TAP AND DRAG NODES, OR CLEAN SLATE AND NEW NODE

                # # # # # self.state.tasks[task]["trajectory"] = self.state.trajectory
                # # # # # Third, trigger a situation and wait for user feedback
                # # # # self.state.situation = {
                # # # #     "type": "review",
                # # # #     "trajectory": self.state.trajectory,
                # # # #     "task": task
                # # # # }
                # # # # log_internal_event(f"[SITUATION] {self.state.situation}")
                # # # # await self.state.event.wait()
                # # # # self.state.situation = {}      # clear situation so backend knows
                # # # # # # # # previous_root_tasks += [task]
                # # # # # Merge task, trajectory and review
                # # # # # # # # task = self.state.resolution
                # # # # if self.state.resolution is not None:
                # # # #     self.state.tasks[task]["tips"] += [self.state.resolution]
                # # # #     # Signal that task needs refinement
                # # # #     self.state.tasks[self.state.parent.title]["tier"] = 0
                # # # #     # task will retry with user's tips
                # # # # else:
                # # # #     # Increase task tier (number of satisfied users)
                # # # #     self.state.tasks[self.state.parent.title]["tier"] += 1
                # # # #     task = None             # signal to stop retrying the task
                # # # # self.state.event.clear()
                await self.state.tokens.put(None)  # signal end of CtxSeg loop
                await self.state.actions.put(None) # signal end of CtxSeg loop
                self.state.parent = None   # reset focus
                task = None

            # Finally, save memories to file
            if self.state.filepath:
                with open(self.state.filepath, 'w', encoding='utf-8') as f:
                    # f.write(json.dumps({"tasks": self.state.tasks, "qna": self.state.qna}, indent=4))
                    f.write(json.dumps({"tasks": self.state.tasks}, indent=4))
            log_internal_event(f"[RCS] Saved tasks to file {self.state.filepath}")

            # And return last report to caller
            return dspy.Prediction(report=exploration[0]["report"])
        except Exception as e:
            log_error(f"[RCS] Critical aforward error: {e}")
            return dspy.Prediction(report="Agents crashed")

    async def agents(self, subtasks: list[Task]):
        """
        Break down into subtasks, given the available information.
        When previous subtasks yield new information, capitalize on it and proceed with new subtasks.
        If some subtasks fail, think outside the box and break down new subtasks.

        Example call (no 'items' key under 'subtasks'):
            next_tool_args = {"tasks": [{"title": "...", "task": "...", "kind": "..."}]}
        """
        # Let LLM observe tips before proceeding
        if len(self.state.tasks[self.state.current.title]["tips"]) > 0:
            # Promote tips to examples, and consume tips
            self.state.tasks[self.state.current.title]["examples"] += self.state.tasks[self.state.current.title]["tips"]
            output = f"Before you proceed, the user provided these additional tips: {self.state.tasks[self.state.current.title["tips"]]}"
            self.state.tasks[self.state.current.title]["tips"] = []
            log_internal_event(f"[RCS] Some tips were found. {output}")
            return output

        log_internal_event(f"[AGENTS] Spawned with tasks: {subtasks}")
        try:
            # # # # available_tasks = [
            # # # #     key for key, value in self.state.tasks.items()
            # # # # ] + [task.title for task in tasks]
            # # # # ghost_dependencies = []
            # # # # # 1. Sanity checks (LLM can observe these)
            # # # # for task in tasks:
            # # # #     # Redundancy check, a.k.a. prevent re-calling the same task
            # # # #     log_internal_event(f"[RCS] Redundancy check (does not apply to root tasks)")
            # # # #     if task.kind != "root" and self.state.parent is not None:
            # # # #         if task.title.lower() == self.state.parent.title.lower() and (not self.state.tasks.get(task.title) or not self.state.tasks[task.title].get("report")):
            # # # #             # task.title = "Simple task - " + task.title
            # # # #             task_payload = {"title": task.title, "task": task.task}
            # # # #             return [{
            # # # #                 # "report": f"Redundancy Error: breaking down your task into the exact same task is not allowed: {task_payload}."
            # # # #                 "report": f"Redundancy Error: rephrase or break down the task: [{task.title}]"
            # # # #             }]

            # # # #     # Accumulate list of ghost dependencies
            # # # #     ghost_dependencies += list(set(task.dependencies).difference(available_tasks))

            # # # # # Ghost dependences check
            # # # # log_internal_event(f"[RCS] Ghost dependencies check")
            # # # # if len(ghost_dependencies) > 0:
            # # # #     log_error(f"Dependencies Error: the following requested task titles do not exist: {ghost_dependencies}.")
            # # # #     return [{
            # # # #         "report": f"Dependencies Error: the following requested task titles do not exist: {ghost_dependencies}."
            # # # #     }]


            # After sanity checks, register current tasks as dependencies of parent task
            for task in subtasks:
                if task.kind != "root" and self.state.parent is not None:
                    log_internal_event(f"[RCS] Connecting dependencies for [{self.state.parent.title}]")
                    # NOTE: the order of execution in dependencies list matters. That's how they came out of the LLM.
                    # if task.title not in self.state.parent.dependencies:
                    if task.title not in self.state.tasks[self.state.parent.title]["dependencies"]:
                        # This is what makes future calls to this task cheaper: instead of going through reasoning again, reports are requested by network design.
                        log_internal_event(f"[{task.title}] becomes dependency of [{self.state.parent.title}]")
                        # self.state.parent.dependencies += [task.title]
                        self.state.tasks[self.state.parent.title]["dependencies"] += [task.title]


            # This rebuilds context and runs each task
            for task in subtasks:
                # log_internal_event(f"[AGENTS] Triggered task type {task.kind}")

                # Ensure sane defaults for new task
                if not self.state.tasks.get(task.title):
                    self.state.tasks[task.title] = {
                        "dependencies": [],
                        "tips": []
                    }

                # 0. Attempt a cache hit on previously reported root task dependencies
                if task.kind == "root" and task.title != self.state.parent.title and not self.is_task_expired(task.title):
                    log_internal_event(f"Cache hit for [{task.title}]")
                    # Update working memory index
                    self.awakened_list += [task.title]
                    continue    # skip this task and trust the cache in self.state.tasks                


                # log_internal_event(f"[AGENTS] Dependencies: {task.dependencies}")
                # 1. Recursively refresh shared state for dependencies
                if task.kind != "simple":
                    # Only non-simple tasks have meaningful dependencies
                    log_internal_event(f"[AGENTS] Dependencies: {self.state.tasks[task.title]["dependencies"]}")
                    if len(self.state.tasks[task.title]["dependencies"]) > 0:
                        # Fill working memory with dependencies
                        await self.agents(
                            subtasks=[
                                Task(
                                    title=dependency,
                                    task=self.state.tasks[dependency]["task"],
                                    kind=self.state.tasks[dependency]["kind"],
                                    ttl=self.state.tasks[dependency]["ttl"],
                                    # # # # dependencies=self.state.tasks[dependency]["dependencies"]
                                )
                                for dependency in self.state.tasks[task.title]["dependencies"]
                                if self.state.tasks[task.title]["kind"] == "root"
                            ]
                        )

                # 2. Route to module according to task's kind
                if task.kind.lower() in ["simple"]:
                    module = dspy.ReAct(
                        signature=self.standard_signature,
                        # tools=self.external_tools + [dspy.Tool(self.user)]
                        tools=self.external_tools
                    )
                    # Allow the LM to emit any tool name and observe wrong tool names
                    module.react.signature.output_fields["next_tool_name"].annotation = str
                    # Preserve the compiled dict of tools, but let LLM see when tool name is wrong, instead of crashing
                    module.tools = self.ObserveTools(module.tools)
                else:
                    # agent capable of spawning other agents
                    module = self.explore
                    await self.state.actions.put({"task": task.title, "data": "Breaking down the task..."})
                log_internal_event(f"Awakened list: {self.awakened_list}")

                # 3. Build up info for RAG (includes working memory)
                # info = self.build_tips(task) + self.build_working_memory(task)
                info = self.build_working_memory(task)

                # 4. Run task
                log_internal_event(f"[RCS] Running task")
                try:
                    # Wake up task
                    self.state.tasks[task.title] = self.state.tasks[task.title] | {
                        "kind": task.kind,
                        "task": task.task,
                        # # # # "dependencies": task.dependencies,
                        "ttl": task.ttl,
                        "state": "awake",
                        "trajectory": []    # reset trajectory
                    }

                    if task.kind in ["complex", "root"]:
                        # If task is not simple, sub-tasks can spawn, so we switch parent
                        previous = self.state.parent
                        # log_internal_event(f"[RCS] Switching complex focus from [{previous}] to [{task}]")
                        self.state.parent = task       # switch parent

                    await self.state.actions.put({"task": task.title, "data": "Getting some context..."})
                    # Start the actual ReAct loop
                    execution = await self.recurse(
                        module=module,
                        task=task,
                        info=info
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
                        # # # # "dependencies": task.dependencies,  # dependencies may update after execution
                        "report": execution.answer,
                    }
                    if self.is_task_expired(task.title):
                        # Update timestamp if cache expired
                        self.state.tasks[task.title]["timestamp"] = datetime.datetime.now(datetime.UTC).isoformat()

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
        log_internal_event(f"Returning tasks: {subtasks}")
        # await self.state.actions.put({"task": self.state.parent.title, "data": "Collecting reports..."})
        return [
            {"title": task.title} | self.state.tasks[task.title]
            for task in subtasks
        ]
        #  + [
        #     # Add tips coming from user after task started
        #     {"tips_from_user": self.state.tasks[self.state.parent.title]["tips"]}
        # ]

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
                # task={"title": task.title, "task": task.task, "dependencies": task.dependencies},
                task={"title": task.title, "task": task.task},
                info=info,
            )

            # TODO: implement self-review here, with self.user tool access to curate examples
        except (dspy.LMError, dspy.AdapterParseError) as e:
            log_error(f"DSPy LM Error: {e}")
            # Retry a bunch of times (LLM backend might be flaky)
            if retries > 0:
                await asyncio.sleep(3)

                # Retry failing agents
                return await self.recurse(
                    module=module,
                    task=task,
                    # info=self.build_info(task.kind), # rebuild info after qna update
                    # info=self.build_tips(task.kind) + self.build_working_memory(task.kind), # rebuild info after qna update
                    info=self.build_working_memory(task), # rebuild working memory
                    retries=retries-1,
                )
            else:
                raise e         # raise it to self.agents


    async def user(self, questions: list[str]):
        """
        Escalate to the user, when out of strategies.
        """
        # Escalate to the user for guidance when unable to fully complete the task.
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
                # self.state.qna[question] = {"kind": self.state.current.kind, "content": "User: "+answer}
                # self.state.qna[question] = {"title": self.state.current.title, "content": "User: "+answer}
                self.state.tasks[self.state.parent.title]["qna"] += [{
                    "question": question,
                    "answer": "User: "+answer
                }]
                # Signal that task needs refinement
                self.state.tasks[self.state.parent.title]["tier"] = 0
                # log_internal_event(f"[RCS] {self.state.qna}")
                log_internal_event(f"[RCS] Updated QnA for [{self.state.current.title}]: {self.state.tasks[self.state.parent.title]["qna"]}")
                answers += [{"question": question, "answer": answer}]
        except Exception as e:
            log_error(f"User error: {e}")
            raise e
        return answers

    def build_working_memory(self, task: Task):
        """Rebuild working memory for context"""
        log_internal_event(f"[RCS] Rebuilding working memory for RAG of task [{task.title}]")
        info = [{"current_time": datetime.datetime.now(datetime.UTC).isoformat()}]
        info += [
            {
                # "trajectory": self.state.tasks[self.state.parent.title]["trajectory"],
                "working_memory": [
                    {
                        "title": awakened,
                        "answer": self.state.tasks[awakened].get("report"),
                    }
                    for awakened in self.awakened_list
                ]
            }
            # | self.info
        ]
        return info

    def build_examples(self, task: Task):
        """Rebuild info for refining task's prompt"""
        log_internal_event(f"[RCS] Rebuilding training examples for [{task}]")
        info = [
            {
                # "tips": [
                #     {"question": question, "answer": answer["content"]}
                #     for question,answer in self.state.qna.items()
                #     # if answer["kind"] == kind
                #     if answer["title"] == task.kind
                "examples": (
                    self.state.tasks[task.title]["qna"] +
                    self.state.tasks[task.title]["examples"] +
                    self.state.tasks[task.title]["tips"]
                )
            }
        ]
        # log_internal_event(f"[RCS] {self.state.tasks[task.title]}")
        # log_internal_event(f"[RCS] Training examples for [{task}]: {info}")
        return info

    def is_task_expired(self, title: str):
        """Tell if a task is expired"""
        log_internal_event(f"[RCS] Expiration check for {title}")
        if self.state.tasks.get(title) and self.state.tasks[title].get("report") and self.state.tasks[title].get("timestamp"):
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
