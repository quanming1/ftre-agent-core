"""
全局线程池注册表。

将不同类型的阻塞任务隔离到独立线程池中，避免互相饥饿：
- chat:       chat workflow 执行（LLM 调用 + agent 循环）
- io:         diff/snapshot/轻量文件 I/O
- background: memory/compaction 等后台 agent
- tool:       工具执行（全局共享，替代每个 agent 各建一个 8 线程池）

使用方式：
    from ftre_agent_core.threading import thread_pool

    # 在 async handler 里
    await loop.run_in_executor(thread_pool.io, blocking_fn, arg1, arg2)

    # 在 sync 代码里提交任务
    future = thread_pool.tool.submit(fn, arg1)

生命周期：
    app.py lifespan 启动时无需操作（池按需创建）
    app 关闭时调用 thread_pool.shutdown()
"""

import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class ThreadPoolRegistry:
    """命名线程池注册表。每个池独立隔离、独立命名、统一关闭。"""

    def __init__(self) -> None:
        self._pools: dict[str, ThreadPoolExecutor] = {}

        # ── 预定义池 ──────────────────────────────────────────────
        self.chat = self._create("chat", max_workers=16)
        self.io = self._create("io", max_workers=16)
        self.background = self._create("background", max_workers=8)
        self.tool = self._create("tool", max_workers=24)

    def _create(self, name: str, max_workers: int) -> ThreadPoolExecutor:
        pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=f"ftre-{name}",
        )
        self._pools[name] = pool
        logger.info(f"[ThreadPool] 创建 '{name}' 池 (max_workers={max_workers})")
        return pool

    def shutdown(self, wait: bool = True) -> None:
        """关闭所有线程池。在 app lifespan shutdown 时调用。"""
        for name, pool in self._pools.items():
            logger.info(f"[ThreadPool] 关闭 '{name}' 池 (wait={wait})")
            pool.shutdown(wait=wait)
        self._pools.clear()


# ── 全局单例 ──────────────────────────────────────────────────────
thread_pool = ThreadPoolRegistry()
