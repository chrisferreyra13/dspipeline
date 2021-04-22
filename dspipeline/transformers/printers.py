from .pipeline import Pipeline


class Printer(Pipeline):
    def map(self, value):
        print (value)
        return value


class PrintToFile(Pipeline):
    def __init__(self,output):
        self.output = output
        super(PrintToFile, self).__init__()

    def map(self, value):
        with open(self.output,'w') as f:
            f.write(value)
        return value