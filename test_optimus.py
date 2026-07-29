from src.agent import AgentWrapper
from src.gepa import GepaWrapper

class TestAgent:
    def test_init(self):
        """
        Test AgentWrapper constructor
        """
        assert AgentWrapper(lm="openai/test")

class TestGepa:
    def test_init(self):
        """
        Test GepaWrapper constructor
        """
        assert GepaWrapper(
            model_string="openai/test",
            objective="Write a python test function",
            debug=True
        )
