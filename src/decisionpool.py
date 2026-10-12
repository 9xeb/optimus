import json
import numpy
import dspy
from anytree import Node

from src.log import log_internal_event, log_error
from src.utils import stream_dspy_program, render_optimus_tree
from src.deact import DeAct

# TODO: break away from rigid tree structure via cheap decisions. Branch becomes Pool.
# The underlying structure is a loose pool of tasks.
# self.explore adds to possible tasks **that depend on each other** to pool.
# intermediate decision level chooses which task to do next and **which other task** output it should see
# self.exploit DeActs on a task when

class DecisionPool(dspy.Module):
    """
    Context Segmentation in DSPy. Based on the paper submitted at RAISE 2026, with a few twists:
        1. give_up tool is replaced with escalate (up). run_strategy tool is replaced with branch (down).
        2. Explorer-Worker becomes recursive Explorer.
        3. Traces are collected for future optimizations on what went wrong.
    """
    def __init__(self, tools: list[dspy.Tool] = None):
        super().__init__()

        self.external_tools = tools if tools else []

        self.qna = []       # merges memory system AND examples for search and compute

        # These traces will be used to distil optimized dspy programs with Flex (fewer tokens)
        # self.unclassified_traces = []            # yet to be classified traces (they get here first)
        # self.good_traces = []       # good traces upon human review
        # self.bad_traces = []        # bad traces upon human review

        self.explore = dspy.ReActV2(
            signature="task -> report",
            # tools=[self.tap, self.branch, self.escalate]
            tools=[self.branch, self.escalate]
        )


        # TODO: replace this with decision model loop
        # self.exploit = DeAct(
        #     # Signature is implicitly the task
        #     tools=self.external_tools
        # )
        self.exploit = dspy.ReActV2(
            signature="task -> report",
            # tools=[self.tap, self.branch, self.escalate] + self.external_tools
            tools=[self.branch, self.escalate] + self.external_tools
        )

        # Setup execution tree and navigation
        self.nodes = []
        self.task_count = 0
        self.cursor = 0

        # TODO: MCP tool to python function wrapper with tool approval
        # TODO: tmux and replace self.exploit with an external call to opencode/codex

    def forward(self, task, **kwargs):
        """
        Expand on explorer-worker segmentation with knowledge_base
        """
        log_internal_event(f"Exploring: ({task})")
        if len(self.nodes) == 0:
            self.nodes = [Node(task, prediction=None, id=self.cursor)]

        exploration = self.explore(
            task=task,
            # dataset=self.qna
        )

        # Store answer to human task in QnA dataset.
        # if self.cursor == "root":
        self.qna += [dspy.Example(question=task, answer=exploration.report)]
        log_internal_event(f"Stored answer to: {json.dumps([task], indent=4)} in dataset.")

        return exploration

    async def aforward(self, task, **kwargs):
        """
        Expand on explorer-worker segmentation with knowledge_base
        """
        log_internal_event(f"Exploring: ({task})")
        if len(self.nodes) == 0:
            self.nodes = [Node(task, prediction=None, id=self.cursor)]

        exploration = await self.explore.acall(
            task=task,
            # dataset=self.qna
        )

        # Store answer to human task in QnA dataset.
        # if self.cursor == "root":
        self.qna += [dspy.Example(question=task, answer=exploration.report)]
        log_internal_event(f"Stored answer to: {json.dumps([task], indent=4)} in dataset.")

        return exploration

    async def branch(self, tasks: list[str]):
        """
        Main interface to get things done via tool-powered agents.
        Make steps to close in on the solution.
        Either:
            - Split a non-trivial situation or a list of situations into parallel branches, independent of each other.
            - Delegate one simple task on the environment on your behalf.
        """
        try:
            if len(tasks) == 1:
                log_internal_event("Branching tool-powered context for single task")
                # Tap
                return await self.tap(task=tasks[0])
            else:
                log_internal_event("Branching recursive context multiple tasks")
                # Branch
                return await self.split(tasks=tasks)
        except Exception as _:
            return "Syntax Error. The correct structure is {'tasks': [...]}"

    async def tap(self, task: str):
        """
        Prompt a tool-powered agent to perform one simple task in the environment on your behalf.

        Return:
            a report on the task
        """
        log_internal_event(f"Advancing: ({task})")

        try:
            # 1. Create a new node and point the cursor at it
            self.task_count += 1
            self.nodes += [Node(task, parent=self.nodes[self.cursor], prediction=None, id=self.task_count)]

            # 2. Perform a ReAct loop with MCP tools
            render_optimus_tree(self.nodes[0])
            # exploitation = self.exploit(task=json.dumps(task))      # streaming decisions is not yet supported
            exploitation = await stream_dspy_program(
                self.exploit,
                task=task,
            )

            # 3. Fill new node with prediction data
            self.nodes[self.cursor].prediction = exploitation
            # self.unclassified_traces += [exploitation.trajectory]

            # 4. Store response in dataset
            self.qna += [dspy.Example(question=task, answer=exploitation.report)]

            return exploitation.report  # this return is just for the agent branch that called it
        except Exception as e:
            log_error(f"{e}")
            input("Debug > ")
    async def split(self, tasks: list[str]):
        """
        Split non-trivial tasks or a list of tasks into parallel branches, independent of each other.

        Return:
            a report on each task
        """
        # Prompt a team of AI agents to perform one or more parallel actions for you.
        # Split a larger or more complex situation into multiple prompts to be delegated.
        log_internal_event(f"Branching: ({json.dumps(tasks, indent=4)})")
        try:
            # if len(tasks) == 1:
            #     return await self.advance(task=tasks[0])
            old_cursor = self.cursor
            for task in tasks:
                # 1. Create new node and point the cursor at its id
                self.task_count += 1
                self.nodes += [Node(task, parent=self.nodes[self.cursor], prediction=None, id=self.task_count)]
                self.cursor = self.task_count

                # 2. Recurse down the tree
                render_optimus_tree(self.nodes[0])
                print(self.qna)
                program = await stream_dspy_program(
                    self.explore,
                    task=task,
                    # dataset=self.qna
                )

                # 3. Fill new node with prediction data
                self.nodes[self.cursor].prediction = program

            self.cursor = old_cursor
            return [
                {"task": task, "report": self.nodes[task].prediction.report}
                for task in tasks
            ]
        except Exception as e:
            log_error(f"{e}")
            input("Debug > ")


    # async def branch_v2(self, tasks: list):
    #     """
    #     Make careful steps to get closer to the solution.
    #     Split non-trivial tasks or a list of tasks into parallel branches, independent of each other.

    #     Each task must be flagged as low-level or high-level.
    #     Low-level tasks 
    #     """
    #     # Each task in the list is tagged either high-level or low-level.
    #     # Low-level tasks are routed to exploit, high-level to explore
    #     # Alternatively, a decision model assigns MCP tools and management tools to branches
    #     pass

    async def research(self, questions: list[str]):
        """
        Asking the right questions is the first step towards learning when you are stuck.
        Dispel any doubt regarding:
            - questions that can be answered with public information.
        """
        # return await self.branch(tasks=questions)
        return [
            {"question": question, "answer": await self.tap(task=question)}
            for question in questions
        ]

    async def escalate(self, questions: list[str]):
        """
        Asking the right questions is the first step towards learning when you are stuck.
        Dispel any doubt regarding any of:
            - private or proprietary information.
            - confirmation before making any change to the environment.
            - problems with tool calls.
        """
        # - environment-specific knowledge that the user might have, that would make things much simpler.

        # This is called when an Explorer feels stuck.
        # TODO: Branch a chat with the user to get unstuck.
        # TODO: do not ask human if question is already known
        # TODO: implement recursive EPHEMERAL copy of parent node to answer the question (root node is human)
        
        # TODO: L1 with RLM when self.qna is not empty. So self.qna can grow forever and RLM can search it.

        # Decide worthy memories
        # triage = dspy.Predict(TriageMemory)
        answers = []
        for question in questions:
            # # 1. Use decision model to find if answer is in dataset
            # log_internal_event(f"Triaging dataset for question {question}")
            # answer = []
            # for data in self.qna:
            #     if await stream_dspy_program(
            #         triage,
            #         question=question,
            #         information=data
            #     ).relevant:
            #         answer += [data]

            # # 2. If answer is not in dataset, escalate to human
            # if len(answer) == 0:
            log_internal_event("Escalating to human.")
            print("#################### HUMAN REQUIRED #####################")
            print(question)
            answer = input("Answer > ")
            log_internal_event("Updating dataset.")
            self.qna += [dspy.Example(question=question, answer=answer)]
            log_internal_event(f"{json.dumps([data.question for data in self.qna], indent=4)}")
            print("#########################################")
            answers += [{"question": question, "answer": answer}]
        log_internal_event(f"Answers: {answers}")
        return answers

    def recall(self, task_id: int):
        """Retrieve a report by ID from a completed task in the task tree."""
        return {
            "task_id": task_id,
            "task": self.nodes[task_id].name,
            "report": self.nodes[task_id].prediction.report
        } if self.nodes[task_id].prediction is not None else "Report not yet available."

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
