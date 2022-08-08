import os

if "LEGATE_MAX_DIM" in os.environ and "LEGATE_MAX_FIELDS" in os.environ:
    from legate.timing import time
else:
    import time as t

    def time():
        return t.time_ns() / 1e3
