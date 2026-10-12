from dataclasses import dataclass
import dspy
import os
import requests

from src.log import log_error

# Base DSPy program for decision models (current original implementation does not work for me)
# TODO: add support for Score and Noul too

@dataclass
class Choice:
    """Choice dataclass"""
    name: str
    instructions: str
    criteria: dict[str, str]

class Choose(dspy.Module):
    """Simple program that support TypeSafe's choices."""
    def __init__(self):
        self.typesafe_url = os.environ["TYPESAFE_API_BASE"]

    def forward(self, state, choices: list[Choice], **kwargs):
        """Make decisions on the state"""        
        response = requests.post(self.typesafe_url, json={
            "state": state,
            "questions": {
                choice.name: {
                    "type": "choice",
                    "instructions": choice.instructions,
                    "criteria": choice.criteria
                }
                for choice in choices
            }
        }, timeout=10)

        try:
            return dspy.Prediction(choices={
                key: value.choice
                for key, value in response.json()["answers"].items()
            })
        except Exception as e:
            print(response.json()["answers"])
            log_error(f"{e}")
            input("Debug Decide > ")

