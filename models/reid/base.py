
class BaseReID:
    def embed(self, frame, xyxy): raise NotImplementedError
class DummyReID(BaseReID):
    def embed(self, frame, xyxy): return None
