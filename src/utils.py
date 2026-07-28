import os
import json
import json5
from json_repair import repair_json

from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings

from src.log import log_internal_event

def initialize_openai_client(model_string: str, temperature: float = 0.0):
    """
    Translate model strings into the appropriate Pydantic AI Model, while preserving the original model string for GEPA
    """
    # Extract provider and model name
    model_provider = model_string.split('/')[0]
    model_name = model_string.split('/', 1)[1]
    log_internal_event(f"Initializing model: {model_provider}/{model_name}")
    if model_provider == "openai":
        # Override openai provider to support custom endpoints
        provider = OpenAIProvider(
            base_url=os.environ["OPENAI_API_BASE"],
            api_key=os.environ["OPENAI_API_KEY"]
        )
        # Construct client with exceptions to manage reasoning effort for models that support it
        try:
            client = OpenAIChatModel(
                model_name=model_name,
                provider=provider,
                # control model parameters
                settings=OpenAIChatModelSettings(
                    openai_reasoning_effort="low",
                    temperature=temperature,
                    frequency_penalty=1.5,       # discourage repetitive tokens
                    # max_tokens=self.context_size
                ),
            )
        except Exception:
            log_internal_event("OpenAI Chat Model - Falling back to safe non-thinking settings")
            client = OpenAIChatModel(
                model_name=model_name,
                provider=provider,
                # control model parameters
                settings=OpenAIChatModelSettings(
                    # openai_reasoning_effort="low",
                    temperature=temperature,
                    frequency_penalty=1.5,       # discourage repetitive tokens
                    # max_tokens=self.context_size
                ),
            )
    else:
        raise ValueError(f"Unsupported model provider: {model_provider}")
    return client

def extract_first_json(text):
    """
    Extract the first available JSON in a free form string

    Args:
        text: free form string

    Returns
        Extracted JSON
    """
    # Extract between the first '{' and the last '}'
    first_bracket = "{"+"{".join(text.split('{')[1:])
    last_bracket = "}".join(first_bracket.split('}')[:-1])+"}"
    return last_bracket

def stabilize_json(unstable_string: str, expected_keys: list = None) -> dict:
    """
    Stabilize a given input JSON, usually from an LLM output.
    Several methods coming from json and json5 modules are tried until one works.
    If all fails, one exception of either AssertionError, IndexError or TypeError is raised
    Args:
        unstable_string: input string containing a JSON string to stablize
        expected_keys: list of expected keys the final dictionary must have
    Raises
        AssertionError if one of the expected keys is not present in the final dictionary
        IndexError if string splitting fails to find a JSON inside curly brackets
        TypeError if both json5 and json fail to parse
    Returns:
        The final stable dictionary
    """
    # Grab the first JSON that is found in the string
    stable_json = extract_first_json(unstable_string)
    repaired_json = repair_json(stable_json)
    try:
        stable_dict = json5.loads(stable_json)
    except ValueError:
        stable_dict = json.loads(stable_json)
    # Ensure all expected_keys are present
    for key in expected_keys:
        assert key in stable_dict.keys()
    return stable_dict
