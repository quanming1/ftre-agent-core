"""
线程池管理
"""
from concurrent.futures import ThreadPoolExecutor


class _ThreadPool:
    def __init__(self):
        self.io = ThreadPoolExecutor(max_workers=16, thread_name_prefix="io")


thread_pool = _ThreadPool()
