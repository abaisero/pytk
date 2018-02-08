from .factory import Factory


# TODO change to FactoryValues
class FactoryValues(Factory):
    def __init__(self, values):
        super(FactoryValues, self).__init__()
        self.values = values
        self.nitems = len(values)

    def i(self, value):
        try:
            return self.values.index(value)
        except ValueError as e:
            re = ValueError('Value {} not valid.'.format(value))
            raise re.with_traceback(e.__traceback__)

    def value(self, i):
        return self.values[i]


class FactoryBool(FactoryValues):
    def __init__(self):
        super(FactoryBool, self).__init__((False, True))


class FactoryN(FactoryValues):
    def __init__(self, n):
        super(FactoryN, self).__init__(list(range(n)))
