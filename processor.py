

class Processor():
    """Handle processes and diplay summaries."""
    def __init__(self, process):
        self.process=process

    def run(self, verbose=True):
        try:
            # Iterate through pipeline
            for _ in self.process:
                pass
        except StopIteration:
            return
        except KeyboardInterrupt:
            return
        