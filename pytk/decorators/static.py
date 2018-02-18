def static(**kwargs):
    """ static function variables """

    def decorate(f):
        for k in kwargs:
            setattr(f, k, kwargs[k])
        return f
    return decorate
