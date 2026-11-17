import asyncio
import argparse
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
# from collections.abc import AsyncIterable
import uvicorn

from src.optimus import Optimus
from src.log import log_error, log_internal_event

parser = argparse.ArgumentParser(
    prog="optimus",
    description="Automatic optimization of text artifacts according to predefined criteria"
)
# run_mode = parser.add_mutually_exclusive_group(required=True)
# run_mode.add_argument('-f', '--file', help="source file")
# parser.add_argument('-d', '--dir', help='Where to save the prompt files', required=False)
# parser.add_argument('-c', '--criteria', help='criteria', action='append')
# parser.add_argument('-l', '--learn', action='store_true', help='enable learn mode')
parser.add_argument('-f', '--file', help='Tasks file to load', required=False)
parser.add_argument('-a', '--approval', action='store_true', help='Enable tool approval')
args = parser.parse_args()

app = FastAPI()
app.frontend("/", directory="src/dist")

optimus = Optimus(filepath=args.file, approval=args.approval)

@app.get("/health")
async def health():
    """State the health of the backend"""
    return {"status": "ok"}

@dataclass
class TaskPayload():
    """Valid payload for POST requests at /tasks"""
    task: str               # title of the task, used as key in the tasks database
    dependencies: list[str]
    tip: str = None         # optional additional user nudge

@app.post("/tasks")
async def create_task(task: TaskPayload):
    """Create a new task"""
    result = optimus.plant(task=task.task, deps=task.dependencies)
    if result is not None:
        raise HTTPException(status_code=409, detail=result)

@app.post("/tasks/{name}/run")
async def run_task(name: str):
    """Update a task and run it"""
    if optimus.state.job is not None and not optimus.state.job.done():
        # For now, only one concurrent root task is allowed
        raise HTTPException(status_code=409, detail=f"Busy with task: [{optimus.state.parent.title}]")
    elif any([task["state"] == "consolidating" for title,task in optimus.state.tasks.items()]):
    # optimus.state.tasks.get(name) and optimus.state.tasks[name]["state"] == "sleeping":
        raise HTTPException(status_code=409, detail="Busy consolidating tasks.")
    else:
        # log_internal_event(f"Triggering Optimus")
        optimus.state.job = asyncio.create_task(optimus.reap(name))

@app.put("/tasks")
async def update_task(task: TaskPayload):
    """Override dependencies and tip for a task"""
    result = optimus.graft(task=task.task, deps=task.dependencies)
    if task.tip:
        optimus.grow(task=task.task, new_examples=[task.tip])
    if result is not None:
        raise HTTPException(status_code=404, detail=result)

@app.delete("/tasks/{name}")
async def delete_task(name: str):
    """Delete a task"""
    result = optimus.prune(name)
    if result is not None:
        raise HTTPException(status_code=404, detail=result)

@app.get("/tasks/stream/actions", response_class=EventSourceResponse)
async def stream_actions():
    """Resume streaming of actions in current job"""
    if optimus.state.job is None or optimus.state.job.done():
        raise HTTPException(status_code=404, detail="Optimus is idle.")
    else:
        # Pick up queue and start streaming all there is
        while True:
            # chunk = await optimus.state.tokens.get()
            chunk = await optimus.state.actions.get()
            if chunk:
                # yield ServerSentEvent(data=chunk["data"].chunk, event=chunk["task"])
                yield ServerSentEvent(data=chunk["data"], event=chunk["task"])
            else:
                break

@app.get("/tasks/stream/tokens", response_class=EventSourceResponse)
async def stream_tokens():
    """Resume streaming of tokens in current job"""
    if optimus.state.job is None or optimus.state.job.done():
        raise HTTPException(status_code=404, detail="Optimus is idle.")
    else:
        # Pick up queue and start streaming all there is
        while True:
            chunk = await optimus.state.tokens.get()
            if chunk:
                # yield ServerSentEvent(data=chunk["data"].chunk, event=chunk["task"])
                yield ServerSentEvent(data=chunk["data"], event=chunk["task"])
            else:
                break

@app.post("/tasks/cancel")
async def cancel_tasks():
    """Stop the currently running job"""
    if optimus.state.job is None or optimus.state.job.done():
        raise HTTPException(status_code=404, detail="Optimus is idle.")
    else:
        await optimus.stop()
        # optimus.state.job.cancel()
        # # Let it observe CancelledError and run its own cleanup
        # await asyncio.gather(optimus.state.job, return_exceptions=True)
        # optimus.state.job = None
        # # Clean up after cancellation
        # log_internal_event("[BACKEND] Cleaning up after client cancellation.")
        # optimus.state.situation = {}
        # optimus.state.tokens = asyncio.Queue()
        # for title,task in optimus.state.tasks.items():
        #     if task.get("state") == "awake":
        #         task["state"] = "sleeping"

@app.get("/tasks")
async def tasks_page():
    """Get tasks network"""
    return {
        "tasks": optimus.state.tasks,
        "situation": optimus.state.situation,
        "error": optimus.state.error,
        "metrics": optimus.state.metrics
    }

@app.post("/review")
async def review_situation(review: dict):
    """Review situation and provide an optional feedback"""
    if (
        optimus.state.situation.get("type") in
        ["task-proposal", "tool-approval", "question", "review"]
    ):
        optimus.state.resolution = review.get("review")
        optimus.state.event.set()
    else:
        # If there are no pending situation, review trigger an instruction update and a task restart
        # This is called steering
        log_internal_event(f"[BACKEND] No pending situations, steering agents with: [{review.get('review')}].")
        root_task = optimus.state.parent.title
        # 1. Stop current job
        await cancel_tasks()
        # await optimus.stop()
        # 2. Refine root task
        optimus.grow(task=root_task, new_examples=[review.get("review")])
        # 3. Re-run task
        return await run_task(name=root_task)
        # optimus.state.tasks[optimus.state.parent.title]["tips"] += [
        #     review.get("review")
        # ]

# @app.get("/", response_class=HTMLResponse)
# async def home_page():
#     """Frontend Home Page"""
#     with open("index.html", 'r', encoding="utf-8") as homepage:
#         return homepage.read()

# For reference:
# @app.post("/tasks/{task_id}")
# async def create_task(
#     task_id: int,                       # path param
#     task: TaskRequest,                  # body (Pydantic model or dataclass)
#     stream_verbose: bool = False,       # query param
# ):

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=False)
