# TODO:
# - improve user experience in UI,
# - review and simplify and secure frontend,
# - desktop/mobile app
# - minimalistic input interface (less clicking, more typing). Single input box, greyed out when no situations or any task is awake (meaning the engine is busy doing something). Emergency button to stop all tasks and go full sleep.
# - try different inspiring visual flavors: network of neurons, galaxy of stars, ...

# - User manual: rule of thumb: approve anything that makes sense and seems related to the task and is not destructive. No need to deep dive into what the agent is doing, that's the point of having an agent
# - implement tool output scrolling, sort of mini-RLM

# 3. implement cron: repeat non-root tasks at scheduled intervals;
# 5. full TUI version of this, callable in a terminal (by providing the backend URL as argument);
# 6. allow reviewing QnA if anything changes
# 7. Vibe code a white collar MCP sandbox with file management, notes, email, calendar, logs in to google/microsoft suites, handles secrets and credentials via agent (it's local anyway)
# 8. backend multiplexing when two UIs are open and consume the same queue of tokens;
# 9. implement fzf-like search
# 10. make a truly air-gapped frontend (no CDN for js)

import anyio
import asyncio
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from pyvis.network import Network
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
# from collections.abc import AsyncIterable
import uvicorn

from src.optimus import Optimus
# from src.utils import stream_dspy_program
from src.utils import prettify_chunk
from src.log import log_error, log_internal_event
from src.ctxseg import Task

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

# With FastAPI
# POST /tasks to send a task and get a streaming response
# GET /tasks to get a pyvista network.html to render and navigate
app = FastAPI()
task_lock = anyio.Lock()
question_lock = anyio.Lock()
optimus = Optimus(filepath=args.file, approval=args.approval)

# Load tasks from file
# if args.file and Path(args.file).exists():
#     with open(args.file, 'r', encoding='utf-8') as f:
#         tasks = json.loads(f.read())
#         log_internal_event(f"Loaded tasks from file: {json.dumps(tasks, indent=4)}")
#         optimus.state.tasks = tasks
# Only dataclass and pydantic model end in POST body
@dataclass
class TaskPayload():
    """Valid payload for POST requests at /tasks"""
    task: str

@app.get("/health")  
async def health():  
    return {"status": "ok"}

@app.post("/tasks")
async def create_task(task: TaskPayload):
    """Submit task and store job"""
    if optimus.state.job is not None and not optimus.state.job.done():
        # For now, only one concurrent root task is allowed
        raise HTTPException(status_code=409, detail="Optimus is busy with another root task.")
    else:
        log_internal_event(f"Triggering Optimus")
        # Detach long-running optimus work
        # optimus.state.job = await optimus.grow(task.task)
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
                # pretty_chunk = prettify_chunk(chunk["data"])
                # print(pretty_chunk, end="", flush=True)
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

# @app.post("/tasks", response_class=EventSourceResponse)
# async def create_task(task: TaskPayload):
#     """Create a task"""
#     if task_lock.locked():
#         # For now, only one concurrent root task is allowed
#         raise HTTPException(status_code=409, detail="Optimus is busy.")
#     async with task_lock:
#         stream = optimus.grow(task.task)
#         try:
#             async for chunk in stream:
#             # async for chunk in optimus.solve(task.task):
#                 pretty_chunk = prettify_chunk(chunk["data"])
#                 print(pretty_chunk, end="", flush=True)
#                 yield ServerSentEvent(data=chunk["data"].chunk, event=chunk["task"])
#                 # await anyio.sleep(0)
#         except Exception as e:
#             log_error(f"Backend caught exception: {e}")
#             return
#         finally:
#             await stream.aclose()
#             log_internal_event(f"Task {task} closed by client.")
#             if args.file:
#                 with open(args.file, 'w', encoding='utf-8') as f:
#                     f.write(json.dumps(optimus.state.tasks, indent=4))
#             log_internal_event(f"Saved tasks to file {args.file}")

@app.get("/tasks")
async def tasks_page():
    """Get tasks network"""
    # tasks = optimus.tasks
    # return tasks
    return {
        "tasks": optimus.state.tasks,
        # "candidate": {"parent": optimus.state.focus, "tasks": optimus.state.candidate},
        "situation": optimus.state.situation
    }
    # return HTMLResponse(net.generate_html())

@app.post("/review")
async def review_situation(review: dict):
    """Review situation and provide an optional feedback"""
    # if optimus.state.situation.get("tasks"):
    if optimus.state.situation.get("type") in ["task-proposal", "tool-approval", "question", "review"]:
        optimus.state.resolution = review.get("review")
        optimus.state.event.set()
    else:
        log_internal_event("[BACKEND] There are no pending situations")

# @app.post("/upload")
# async def upload_file():
#     """Upload a file to working memory of root task"""
#     pass

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