# TODO
Add tk/s count and total token count
Conversations become prototypes. You can scroll them, steer them. Everywhere is an optimus loop.

# gepa-cli a.k.a. optimus
Opinionated CLI interface for GEPA's optimize_anything.
Universal text-based artifacts optimizer, with optional human in the loop (HITL).
Optimizes specs instead of results.

```bash
# Requires Python > 3.12
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Try out these goals
```
Formal email for job application as devops engineer
AI Agent prompt for ...
Python script to iteratively optimize cpu requests and mem requests of a kubernetes deployment
```

- Import GEPA parts from agent-compose repo
- Refine CLI interface
- Write pytests (with SLM in github action)

# Opinionated
- only one LLM for judge, reflection
- evaluation is done by LLM (pydantic-ai)
- asks questions mid-optimization for human feedback

# How GEPA works
JUDGE (inside your evaluator function) -> candidate + criteria = feedback
REFLECTION (inside GEPA) -> objective + feedback = candidate

Judge is ...
Reflection is ...
GEPA works like this ...