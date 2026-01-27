# coding=utf-8
"""
Parallel execution utilities for jedireporter.
"""

import itertools
import time
from collections import deque
from concurrent.futures import Executor
from typing import Any, Callable, Iterable, TypeVar

U = TypeVar('U')


def parallelMap(pool: Executor, fn: Callable[..., U], *iterables: Any,
                threadCount: int = 1, timeout: float = None) -> Iterable[U]:
    """
    Lazy map given funcion on given data using a thread/process pool.
    @param pool: thread or process pool with submit() function
    @param fn: function to map
    @param iterables: iterables of individual arguments
    @param threadCount: number of worker threads available
    @param timeout: The maximum number of seconds to wait. If None, then there
            is no limit on the wait time.

    NOTE: We override Executor.map because the original code was not memory efficient since
    it stored all Future objects in a list. This implementation is using a queue.
    """
    if timeout is not None:
        end_time = timeout + time.time()

    argStream = zip(*iterables)

    # Create a queue of size 2 * max_workers
    buffer = deque([pool.submit(fn, *args) for args in list(itertools.islice(argStream, 2 * threadCount))])

    # Yield must be hidden in closure so that the futures are submitted
    # before the first iterator value is required.
    def result_iterator() -> Iterable[U]:
        try:
            # In a loop, pop a result from the queue and submit new data to be processed
            while buffer:
                future = buffer.popleft()
                if timeout is None:
                    yield future.result()
                else:
                    yield future.result(end_time - time.time())
                try:
                    args = next(argStream)
                    buffer.append(pool.submit(fn, *args))
                except StopIteration:
                    pass
        finally:
            for future in buffer:
                future.cancel()

    return result_iterator()
