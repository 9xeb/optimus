import logging
import json
import sys
from pydantic_ai import ModelMessage

DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
CRITICAL = logging.CRITICAL

WHITE = "\x1b[37m"
GREY = "\x1b[38;20m"
MAGENTA = "\x1b[35;95m"
BLUE = "\x1b[34;94m"
CYAN = "\x1b[96m"
GREEN = "\x1b[32;92m"
YELLOW = "\x1b[33;20m"
RED = "\x1b[31;20m"
BOLD_RED = "\x1b[31;1m"

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
    # format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def log(message, level=logging.DEBUG, color=GREY):
    logging.log(level, color + message + "\x1b[0m")

def log_tool_request(command):
    log(
        message="%s - CALLS TOOL - " % (command),
        level=INFO,
        color=YELLOW
    )

def log_tool_response(command):
    log(
        message="TOOL RESPONSE - %s" % (command),
        level=INFO,
        color=YELLOW
    )

def log_answer(history):
    log(
        message="ANSWER - %s" % history[-1]["content"],
        level=INFO,
        color=GREEN
    )


def log_request(request):
    log(
        message="REQUEST - %s" % request,
        level=INFO,
        color=MAGENTA
    )

def log_thought(thought):
    log(
        message="THOUGHT - %s" % thought,
        level=INFO,
        color=CYAN
    )

def log_response(answer):
    log(
        message="%s - " % answer,
        level=INFO,
        color=GREEN
    )

def log_internal_event(event):
    log(
        # message="INTERNAL EVENT - %s" % event,
        message="%s" % event,
        level=INFO,
        color=CYAN,
        # color=WHITE
    )

def log_error(event):
    log(
        message=" ERROR - %s" % event,
        level=INFO,
        color=RED
    )

def log_part(prefix, part):
    if part.part_kind == "user-prompt":
        log(
            message=f"{prefix} - {part.content}",
            level=INFO,
            color=WHITE
        )
    elif part.part_kind == "thinking":
        log_thought(f"{prefix} - {part.content}")
    elif part.part_kind == "tool-call":
        log_tool_request(f"{prefix} - {json.dumps({"tool_name": part.tool_name, "args": part.args})}")
    elif part.part_kind == "tool-return":
        log_tool_response(f"{prefix} - {json.dumps({"tool_name": part.tool_name, "result": part.content})}")
    elif part.part_kind == "text":
        log_response(f"{prefix} - {part.content.encode("unicode_escape").decode("utf-8")}")