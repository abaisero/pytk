class sentinel(object):
    """ Re-compute property only conditionally to another property """
    class _sentinel_property(property):
        def __init__(self, fget, watch):
            super(sentinel._sentinel_property, self).__init__(fget=fget)
            self.watch = watch
            self.watch_cache = '__sentinel_watch_{}'.format(fget.__name__)
            self.watchman_cache = '__sentinel_watchman_{}'.format(fget.__name__)

        def __get__(self, obj, objtype=None):
            if obj is None:
                return self
            if self.fget is None:
                raise AttributeError('unreadable attribute')
            watch = getattr(obj, self.watch)
            if not hasattr(obj, self.watchman_cache) or getattr(obj, self.watch_cache) is not watch:
                setattr(obj, self.watch_cache, watch)
                setattr(obj, self.watchman_cache, self.fget(obj))
            return getattr(obj, self.watchman_cache)

        def __delete__(self, obj):
            delattr(obj, self.watchman_cache)

    def __init__(self, attr_name):
        self.attr_name = attr_name

    def __call__(self, f):
        return sentinel._sentinel_property(f, self.attr_name)
