import random


class Processor():
    """Handle processes and diplay summaries."""
    def __init__(self, process):
        self.process=process
        self.id=random.randrange(10**2,10**4)

    def run(self, verbose=True):
        try:
            # Iterate through pipeline
            for _ in self.process:
                pass
        except StopIteration:
            return
        except KeyboardInterrupt:
            return
        