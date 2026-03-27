"""
Context lifecycle management — compaction and reset.

Implements the strategies from the Anthropic article:
  1. Compaction: summarize old messages, keep recent ones (same session).
     Preserves continuity but does NOT give a clean slate.
  2. Reset: write a structured checkpoint to file, start a brand-new message list.
     Solves "context anxiety" — the model gets a fresh window and stops
     trying to wrap up prematurely.

The article notes that compaction alone is insufficient for models that exhibit
context anxiety. Reset is the stronger intervention.
"""
from __future__ import annotations

import re
import subprocess
import logging

import tiktoken

import config

log = logging.getLogger("harness")

# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        try:
            _encoder = tiktoken.encoding_for_model(config.MODEL)
        except Exception:
            _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def count_tokens(messages: list[dict]) -> int:
    """Rough token count for a message list."""
    enc = _get_encoder()
    total = 0
    for msg in messages:
        content = msg.get("content") or ""
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "") for block in content if isinstance(block, dict)
            )
        total += len(enc.encode(str(content))) + 4
        for tc in msg.get("tool_calls", []):
            total += len(enc.encode(str(tc.get("function", {}).get("arguments", ""))))
    return total


# ---------------------------------------------------------------------------
# Context anxiety detection
# ---------------------------------------------------------------------------

# Patterns that indicate the model is trying to wrap up prematurely
_ANXIETY_PATTERNS = [
    r"(?i)let me wrap up",
    r"(?i)i('ll| will) finalize",
    r"(?i)that should be (enough|sufficient)",
    r"(?i)i('ll| will) stop here",
    r"(?i)due to (context |token )?limit",
    r"(?i)running (low on|out of) (context|space|tokens)",
    r"(?i)to (save|conserve) (context|space|tokens)",
    r"(?i)i('ve| have) covered the (main|key|essential)",
    r"(?i)in the interest of (time|space|brevity)",
]


def detect_anxiety(messages: list[dict]) -> bool:
    """
    Check recent assistant messages for signs of context anxiety —
    the model trying to wrap up work prematurely because it thinks
    it's running out of context space.
    """
    # Only check the last few assistant messages
    recent_texts = []
    for msg in reversed(messages[-10:]):
        if msg.get("role") == "assistant" and msg.get("content"):
            recent_texts.append(msg["content"])
        if len(recent_texts) >= 3:
            break

    combined = " ".join(recent_texts)
    matches = sum(1 for p in _ANXIETY_PATTERNS if re.search(p, combined))
    if matches >= 2:
        log.warning(f"Context anxiety detected ({matches} signals found)")
        return True
    return False


# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------

def compact_messages(messages: list[dict], llm_call, role: str = "default") -> list[dict]:
    """
    Summarize the older portion of messages, keep the system prompt
    and recent messages intact.

    Different roles get different compaction strategies:
    - "evaluator": keeps more history (50%) for cross-round comparison
    - "builder": aggressive compaction, keeps only 20% + current contract/feedback
    - "default": balanced at 30%

    llm_call: a callable(messages) -> str that calls the LLM for summarization.
    """
    if not messages:
        return messages

    # Role-specific retention ratios
    retention = {"evaluator": 0.50, "builder": 0.20}.get(role, 0.30)

    system = [messages[0]] if messages[0].get("role") == "system" else []
    non_system = messages[len(system):]

    keep_count = max(4, int(len(non_system) * retention))
    old = non_system[:-keep_count]
    recent = non_system[-keep_count:]

    if not old:
        return messages

    old_text = _messages_to_text(old)

    # Role-specific summarization instructions
    if role == "evaluator":
        summarize_instruction = (
            "Summarize the following QA work log. Preserve: all scores given, "
            "bugs found, quality assessments, and cross-round comparisons. "
            "The evaluator needs this history to track improvement trends."
        )
    elif role == "builder":
        summarize_instruction = (
            "Summarize the following build log. Preserve: files created/modified, "
            "current architecture decisions, and the latest error states. "
            "Discard intermediate debugging steps and superseded code."
        )
    else:
        summarize_instruction = (
            "Summarize the following agent work log. Preserve: key decisions, "
            "files created/modified, current progress, and errors encountered."
        )

    summary = llm_call([
        {"role": "system", "content": f"You are a concise summarizer. {summarize_instruction}"},
        {"role": "user", "content": old_text},
    ])

    summary_msg = {
        "role": "user",
        "content": f"[COMPACTED CONTEXT — summary of earlier work]\n{summary}",
    }

    return system + [summary_msg] + recent


# ---------------------------------------------------------------------------
# Reset (checkpoint + fresh start)
# ---------------------------------------------------------------------------

def create_checkpoint(messages: list[dict], llm_call) -> str:
    """
    Serialize current state into a structured handoff document.
    Persists to progress.md so it survives across sessions.
    Returns the checkpoint text.
    """
    from pathlib import Path

    text = _messages_to_text(messages)
    checkpoint = llm_call([
        {"role": "system", "content": (
            "You are creating a handoff document for the next agent session. "
            "The next session starts with a COMPLETELY EMPTY context window — "
            "it has zero memory of anything that happened here.\n\n"
            "Structure the handoff as:\n"
            "## Completed Work\n(what was built, with file paths)\n"
            "## Current State\n(what works, what's broken right now)\n"
            "## Next Steps\n(exactly what to do next, in order)\n"
            "## Key Decisions & Rationale\n(why things were done this way)\n"
            "## Known Issues\n(bugs, incomplete features, technical debt)\n\n"
            "Be thorough and specific — file paths, function names, error messages. "
            "The next session's success depends entirely on this document."
        )},
        {"role": "user", "content": text},
    ])

    # Persist to file
    progress_path = Path(config.WORKSPACE) / config.PROGRESS_FILE
    progress_path.write_text(checkpoint, encoding="utf-8")
    log.info(f"Checkpoint written to {config.PROGRESS_FILE}")

    return checkpoint


def restore_from_checkpoint(checkpoint: str, system_prompt: str) -> list[dict]:
    """
    Build a fresh message list from a checkpoint.
    Includes recent git diff for additional grounding.
    """
    # Get recent code changes for extra context
    git_context = ""
    try:
        result = subprocess.run(
            "git diff --stat HEAD~5 2>/dev/null || git log --oneline -5 2>/dev/null",
            shell=True,
            cwd=config.WORKSPACE,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.stdout.strip():
            git_context = f"\n\nRecent code changes:\n```\n{result.stdout.strip()[:2000]}\n```"
    except Exception:
        pass

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": (
            "You are resuming an in-progress project. Your previous session's "
            "context was reset to give you a clean slate.\n\n"
            "Here is the handoff document from the previous session:\n\n"
            + checkpoint
            + git_context
            + "\n\nContinue from where the previous session left off. "
            "Do NOT redo work that's already completed."
        )},
    ]


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _messages_to_text(messages: list[dict]) -> str:
    """Flatten messages into readable text for summarization."""
    parts = []
    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content") or ""
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "") for block in content if isinstance(block, dict)
            )
        if content:
            parts.append(f"[{role}] {content[:3000]}")
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            parts.append(f"[tool_call] {fn.get('name', '?')}({fn.get('arguments', '')[:500]})")
    return "\n".join(parts)
