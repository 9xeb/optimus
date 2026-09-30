from src.ctxseg import CtxSeg
from src.optimus import Optimus

class TestCtxSeg():
    def test_init(self):
        """Test CtxSeg constructor"""
        assert CtxSeg(tools=[])

class TestOptimus():
    def test_init(self):
        """Test Optimus constructor"""
        assert Optimus()

