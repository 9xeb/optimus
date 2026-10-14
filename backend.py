import anyio
# import asyncio
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
# from src.ctxseg import Task

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
optimus = Optimus(approval=args.approval)
# Load tasks from file
if args.file and Path(args.file).exists():
    with open(args.file, 'r', encoding='utf-8') as f:
        tasks = json.loads(f.read())
        log_internal_event(f"Loaded tasks from file: {json.dumps(tasks, indent=4)}")
        optimus.tasks = tasks

# Only dataclass and pydantic model end in POST body
@dataclass
class TaskPayload():
    """Valid payload for POST requests at /tasks"""
    task: str

@app.post("/tasks", response_class=EventSourceResponse)
async def create_task(task: TaskPayload):
    """Create a task"""
    if task_lock.locked():
        # For now, only one concurrent root task is allowed
        raise HTTPException(status_code=409, detail="Optimus is busy with another root task.")
    async with task_lock:
        stream = optimus.grow(task.task)
        try:
            async for chunk in stream:
            # async for chunk in optimus.solve(task.task):
                pretty_chunk = prettify_chunk(chunk["data"])
                print(pretty_chunk, end="", flush=True)
                yield ServerSentEvent(data=chunk["data"].chunk, event=chunk["task"])
                # await anyio.sleep(0)
        finally:
            await stream.aclose()
            log_internal_event(f"Task {task} closed by client.")
            with open(args.file, 'w', encoding='utf-8') as f:
                f.write(json.dumps(optimus.tasks, indent=4))
            log_internal_event(f"Saved tasks to file {args.file}")

@app.get("/reports", response_class=EventSourceResponse)
async def stream_report(task_title: str):
    """Stream a task report by title"""
    # We separate task creation from report streaming, in case frontend breaks or is reloaded.
    pass

@app.get("/tasks")
async def tasks_page():
    """Get tasks network"""
    tasks = optimus.tasks
    return tasks
    # return HTMLResponse(net.generate_html())

# @app.get("/tasks", response_class=HTMLResponse)
# async def tasks_page():
#     """Get tasks graph"""
#     net = Network(directed=True, height="750px", width="100%",
#                   bgcolor="#222222", font_color="white", select_menu=True, cdn_resources="in_line")
#     net.barnes_hut()
#     for title, task in optimus.tasks.items():
#         # Connect each task in network
#         net.add_node(title, label=title, title=title, metadata=task)
#         for dependency in task["dependencies"]:
#             net.add_edge(dependency, title)
#     # net.add_edges([(3, 1), (3, 2, 5)])  # batch; 3rd tuple element = width
#     # net.show("digraph.html")
#     # net.show_buttons(filter_=['physics'])
#     return HTMLResponse(net.generate_html())

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