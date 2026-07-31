## Optimus - The science of prompt engineering
Optimus accelerates continuous, evidence-driven design of LLM instructions to achieve specified goals. For agentic goals, Optimus can be configured to experiment on your test environment via MCP servers. Humans retain ownership of the goals.

Give Optimus a goal and it will explore different prompts that could achieve the goal, then choose the prompt that performed best.

Start from scratch:
![Start from scratch](docs/scratch.gif)
Or refine/pivot existing instructions:
![Refine existing prompts](docs/scratch.gif)


Powered by GEPA[[1]][GEPA] and LLMs. Optimize those prompts with science and automation.

### Quickstart
Install
```bash
# Requires Python > 3.12
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```
Point to your LLM (preferably local) and environment:
```bash
export OPENAI_API_BASE=http://127.0.0.1:8080/v1
export OPENAI_API_KEY=1234
export OPENAI_API_MODEL=openai/my/local/model
```
Pure text optimization (no MCP):
```bash
echo "Formal email for job application as DevOps engineer." | optimus -f EMAIL.txt
# ... (you decide to steer the goal)
echo "Replace Kubernetes with Nomad." | optimus -f EMAIL.txt
# ... (you change your mind)
echo "Insert AWS, GCP and Azure." | optimus -f EMAIL.txt
```
Agentic:
```bash
export MCP_CONFIG=path/to/mcp_config.json   # Optional
echo "Find pod problems in the kubernetes cluster" | optimus -f TROUBLESHOOTING.txt
```

### How it works
Suppose your AI agent achieved the intended goal with no mistakes and no hiccups, then look just one step back. All your AGENT.md, MEMORY.md, skills, tool signatures, RAG knowledge were merged into a single, final prompt. And that prompt got the job done. 

What if you could instead start from scratch and rely on an evidence-driven prompt search algorithm, that learned directly from you and the environment on which it is supposed to be deployed, with no additional structural overhead? Enter Optimus.

At the heart of Optimus lies a multi-agent, evolutionary search loop.
It starts with nothing but a given objective and an optional prompt draft to pick up from. It **runs** the prompt on your environment, **criticizes** the outcomes against the objective, and **reflects** on a new prompt to run. In between, GEPA **keeps** an evolutionary pool of least criticized prompts. Optimus stops when it either reaches a **plateau** in the optimization or it finds a prompt that yielded **perfect** results on your repeatable, test environment. Every time Optimus stops, you can steer the objective and watch it explore again.

### Why it works
Instead of manually chasing outcomes, Optimus focuses on applying a scientific method to explore the prompts that might generate those outcomes. Anything that the LLM can infer at runtime is left to be inferred at runtime.

### From Conversations to Prototyping
We are used to chatting with AI. On one side, conversations and prompt engineering with LLMs can become exhausting for humans when faced with complex tasks. The gap between what you want to say and how you should say it can get frustrating. On the other side, computers require O(n) RAM as the conversations grow. Optimus differs from a typical conversational interface for a few reasons:
 - effort is shifted away from the user and towards the LLM.
 - conversational context size does not build up.
 - prompts are tested and scored in your (repeatable, test) environment before you receive an actual response.

Focus is the objective. Persistency is in the prompt. Soundness is provided by your environment's feedback.

<!-- ### Examples (code, email, memory, ...)
Try out these goals
```
Formal email for job application as devops engineer
AI Agent prompt for ...
Python script to iteratively optimize cpu requests and mem requests of a kubernetes deployment
Add AWS, GCP
Remove Kubernetes and add Nomad
``` -->
 
### Credits
[gepa]: https://gepa-ai.github.io/gepa/blog/2026/02/18/introducing-optimize-anything/
[[1]][GEPA] - GEPA - optimize_anything

## TODO
Add tk/s count and total token count
Conversations become prototypes. You can scroll them, steer them. Everywhere is an optimus loop.
Add more scores (turns taken, total tokens, other metrics) to optimize
