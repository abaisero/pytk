from .factory import Factory


# TODO change to FactoryValues
class FactoryValues(Factory):
    def __init__(self, values):
        super().__init__()
        self.values = values
        self.nitems = len(values)

    def i(self, value):
        try:
            return self.values.index(value)
        except ValueError as e:
            re = ValueError(f'Value {value} not valid.')
            raise re.with_traceback(e.__traceback__)

    def value(self, i):
        return self.values[i]


class FactoryBool(FactoryValues):
    def __init__(self):
        super().__init__((False, True))


class FactoryN(FactoryValues):
    def __init__(self, n):
        super().__init__(list(range(n)))
