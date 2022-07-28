from typing import Callable
from .timing import time
import logging

logger = logging.getLogger()

def record_timestamps(func: Callable) -> Callable:
    """Decorator to print timestamps at the start and end of a function invocation."""
    def decorated_function(*args, **kwargs):
        start = time()
        logger.info("%s started at %s ns", func.__name__, str(start))
        out = func(*args, **kwargs)
        finish = time() 
        logger.info("%s ended at %s ns", func.__name__, str(finish))

        duration = (finish - start)/1e3      # ms
        logger.info("%s took %s ms\n", func.__name__, str(duration))

        return out

    return decorated_function

