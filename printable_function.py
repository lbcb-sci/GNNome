import inspect

class PrintableFunction:
    def __init__(self, function):
        self.function = function

    def __str__(self):
        result = str(inspect.getsourcelines(self.function)[0])
        result = result.strip('["\\n"],')
        index = result.find('lambda')
        return result[index:]
