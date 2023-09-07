import os

if "LEGATE_MAX_DIM" in os.environ and "LEGATE_MAX_FIELDS" in os.environ:
    from legate.timing import time as _time
    from legate.core.runtime import runtime

    def time():
        # Ensure that all pending operations are flushed
        # before we start timing.
        runtime.flush_scheduling_window()
        return _time()
else:
    import time as t

    def time():
        return t.time_ns() / 1e3
