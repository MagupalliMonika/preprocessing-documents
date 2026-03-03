import os

class BaseTechnique:

    def __init__(self, name, output_dir="output"):
        self.name = name
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def detect(self, image, metadata=None):
        raise NotImplementedError

    def apply(self, image, metadata=None):
        raise NotImplementedError