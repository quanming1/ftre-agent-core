"""协议所有权回归：Core 只能使用 ftre-llm 的 StreamChunk 类型。"""

from ftre_llm.events import FinishChunk, TextDeltaChunk

from ftre_agent_core.llm import FinishChunk as CoreFinishChunk
from ftre_agent_core.llm import TextDeltaChunk as CoreTextDeltaChunk


def test_core_event_exports_are_ftre_llm_types() -> None:
    assert CoreTextDeltaChunk is TextDeltaChunk
    assert CoreFinishChunk is FinishChunk
