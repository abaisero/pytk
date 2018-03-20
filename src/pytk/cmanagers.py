import tempfile, shutil
from contextlib import contextmanager


@contextmanager
def tmpdir():
    dname = tempfile.mkdtemp()
    try:
        yield dname
    finally:
        shutil.rmtree(dname)


class tmpdir(object):
    def __enter__(self):
        self.dname = tempfile.mkdtemp()
        return self.dname

    def __exit__(self):
        shutil.rmtree(self.dname)
