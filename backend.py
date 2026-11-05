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
optimus = Optimus(filepath=args.file, approval=args.approval)

@app.get("/health")
async def health():
    """State the health of the backend"""
    return {"status": "ok"}

@dataclass
class TaskPayload():
    """Valid payload for POST requests at /tasks"""
    task: str
@app.post("/tasks")
async def create_task(task: TaskPayload):
    """Submit task and store job"""
    if optimus.state.job is not None and not optimus.state.job.done():
        # For now, only one concurrent root task is allowed
        raise HTTPException(status_code=409, detail="Optimus is busy with another root task.")
    else:
        log_internal_event(f"Triggering Optimus")
        optimus.state.job = asyncio.create_task(optimus.grow(task.task))
        return {"status": "started"}

@app.get("/tasks/stream", response_class=EventSourceResponse)
async def stream_tasks():
    """Resume streaming of tasks in current job"""
    if optimus.state.job is None or optimus.state.job.done():
        raise HTTPException(status_code=404, detail="Optimus is idle.")
    else:
        # Pick up queue and start streaming all there is
        while True:
            chunk = await optimus.state.tokens.get()
            if chunk:
                yield ServerSentEvent(data=chunk["data"].chunk, event=chunk["task"])
            else:
                break

@app.post("/tasks/cancel")
async def cancel_tasks():
    """Stop the currently running job"""
    if optimus.state.job is None or optimus.state.job.done():
        raise HTTPException(status_code=404, detail="Optimus is idle.")
    else:
        optimus.state.job.cancel()
        # Let it observe CancelledError and run its own cleanup
        await asyncio.gather(optimus.state.job, return_exceptions=True)
        optimus.state.job = None
        # Clean up after cancellation
        log_internal_event("[BACKEND] Cleaning up after client cancellation.")
        optimus.state.situation = {}
        optimus.state.tokens = asyncio.Queue()
        for title,task in optimus.state.tasks.items():
            if task.get("state") == "awake":
                task["state"] = "sleeping"

@app.get("/tasks")
async def tasks_page():
    """Get tasks network"""
    return {
        "tasks": optimus.state.tasks,
        "situation": optimus.state.situation
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
        log_internal_event("[BACKEND] There are no pending situations")

@app.get("/", response_class=HTMLResponse)
async def home_page():
    """Frontend Home Page"""
    with open("index.html", 'r', encoding="utf-8") as homepage:
        return homepage.read()

# For reference:
# @app.post("/tasks/{task_id}")
# async def create_task(
#     task_id: int,                       # path param
#     task: TaskRequest,                  # body (Pydantic model or dataclass)
#     stream_verbose: bool = False,       # query param
# ):

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
