#!/usr/bin/env python3
"""Verify ftdre docs against source code."""
import re
from pathlib import Path

DOCS_DIR = Path("E:/ftre-docs/src/content")
FTRE = Path("E:/ftre/src/ftre")
CORE = Path("E:/ftre-agent-core/src/ftre_agent_core")
DESKTOP = Path("E:/binn/ftre-desktop")

issues = []

def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")

# Check context-management.md claims
ctx_doc = read(DOCS_DIR / "context-management.md")

# Verify compact_handler.py constants and logic
ch = read(FTRE / "agent/compact_handler.py")
if "DEFAULT_PRECOMPACT_THRESHOLD = 0.5" not in ch:
    issues.append("compact_handler.py: DEFAULT_PRECOMPACT_THRESHOLD missing/wrong")
if "DEFAULT_COMPACT_THRESHOLD = 0.6" not in ch:
    issues.append("compact_handler.py: DEFAULT_COMPACT_THRESHOLD missing/wrong")
if 'if not summary or len(summary) < 200 or "## " not in summary:' not in ch:
    issues.append("compact_handler.py: summary validation mismatch")
if 'payload["tokens_after"] = estimate_events_tokens([synthetic])' not in ch:
    issues.append("compact_handler.py: tokens_after computation mismatch")
if 'getattr(config.context, "compact_threshold", DEFAULT_COMPACT_THRESHOLD)' not in ch:
    issues.append("compact_handler.py: enable_ratio metadata source mismatch")

# Verify loop.py compact logic
loop = read(FTRE / "agent/loop.py")
for needle in [
    'data["need_compact"] = True',
    'threshold=getattr(config.context, "precompact_threshold", 0.5)',
    'silent = getattr(config.context, "silent", True)',
]:
    if needle not in loop:
        issues.append(f"loop.py missing: {needle}")

if "COMPACT_UNRETRYABLE_LLM_CODES = {\"auth_error\", \"bad_request\", \"content_filter\"}" not in loop:
    issues.append("loop.py: COMPACT_UNRETRYABLE_LLM_CODES mismatch")

# Verify config.py context fields
config = read(FTRE / "config.py")
for field in ["precompact_threshold", "compact_threshold", "consolidation_ratio", "safety_buffer", "idle_compaction", "silent"]:
    if field not in config:
        issues.append(f"config.py: missing context field {field}")

# Verify session/manager.py to_openai_messages context_compact handling
mgr = read(FTRE / "session/manager.py")
if 'elif _t == "context_compact":' not in mgr:
    issues.append("manager.py: context_compact branch missing")
if '[历史上下文摘要]' not in mgr:
    issues.append("manager.py: summary prefix mismatch")

# Verify ws_channel.py attachments
ws = read(FTRE / "channel/ws_channel.py")
allowed_mimes = ['"image/png"', '"image/jpeg"', '"image/webp"', '"image/gif"']
for m in allowed_mimes:
    if m not in ws:
        issues.append(f"ws_channel.py missing mime {m}")
if "3 * 1024 * 1024" not in ws:
    issues.append("ws_channel.py: 3MB size limit missing")

# Verify plugin files
builtin_dir = FTRE / "plugin/builtin"
for name in ["title_gen.py", "context_govern.py", "skill_plugin.py", "mcp_plugin.py"]:
    if not (builtin_dir / name).exists():
        issues.append(f"builtin plugin missing: {name}")

# Report
if issues:
    print("DISCREPANCIES FOUND:")
    for i in issues:
        print(" -", i)
else:
    print("No major discrepancies found by script.")
