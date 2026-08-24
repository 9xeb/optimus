from alive_progress import alive_bar

from src.agent import AgentWrapper
from src.gepa import GepaWrapper

class TestAgent:
    def test_init(self):
        """
        Test AgentWrapper constructor
        """
        assert AgentWrapper(name="TEST")

class TestGepa:
    def test_init(self):
        """
        Test GepaWrapper constructor
        """
        progress_bar = alive_bar(
            total=100,
            dual_line=True,
            manual=True,
            # title_length=max([len(TITLES[title]) for title in TITLES])
        )
        assert GepaWrapper(
            progress_bar=progress_bar,
            # model_string="openai/test",
            # objective="Write a python test function",
            debug=True
        )
