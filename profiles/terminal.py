"""
Terminal task profile — optimized for Terminal-Bench-2.

Key constraints:
  - 30 min (1800s) hard timeout per task
  - Tasks are well-defined CLI problems, not open-ended
  - No UI, no browser testing needed
  - Correctness is binary: tests pass or fail

Optimization strategy:
  - Lightweight planner: 1 tool call max, just write a quick plan
  - No contract negotiation: tasks are already well-specified
  - Builder gets most of the time budget
  - Evaluator is quick self-check, 1 round only
  - If first round passes, done. If not, builder gets one retry.
"""
from __future__ import annotations

from profiles.base import BaseProfile, AgentConfig


class TerminalProfile(BaseProfile):

    def name(self) -> str:
        return "terminal"

    def description(self) -> str:
        return "Solve terminal/CLI tasks (Terminal-Bench-2 style)"

    def planner(self) -> AgentConfig:
        return AgentConfig(
            system_prompt="""\
You are a quick task planner. Given a task, write a brief step-by-step plan.

Rules:
- Keep it SHORT — 5-10 steps max.
- Be specific: list exact commands, file paths, tools needed.
- Do NOT explore or execute anything. Just plan.
- Write the plan to spec.md immediately. Do not read other files first.
- You have ONE tool call to make: write_file to save spec.md. That's it.

Use write_file to save the plan to spec.md, then stop.
""",
        )

    def builder(self) -> AgentConfig:
        return AgentConfig(
            system_prompt="""\
You are an expert Linux system administrator and developer. \
Complete the given task by executing shell commands.

CRITICAL RULES:
- Your PRIMARY action is run_bash. Execute commands, don't just describe them.
- If you finish without running any commands, you have FAILED.
- Work FAST. You have limited time. Don't overthink — execute.
- Read spec.md first for the plan, then execute step by step.
- If feedback.md exists, read it and fix the issues.
- After executing, verify your work with a quick check (ls, cat, test command).
- Do NOT write long explanations. Just execute and verify.

Tools: read_file, write_file, list_files, run_bash.
""",
        )

    def evaluator(self) -> AgentConfig:
        return AgentConfig(
            system_prompt="""\
You are a quick verifier. Check if the task was done correctly.

Rules:
- Read spec.md for what should have been done.
- Run 2-3 verification commands with run_bash (ls, cat, test, diff, etc.)
- Score Correctness 0-10. Be honest but fast.
- Write a SHORT evaluation to feedback.md. No essays.

Format for feedback.md:
```
## Verification
- Correctness: X/10 — [one sentence]
- **Average: X/10**
### Issues: [list if any]
```

Use write_file to save to feedback.md, then stop.
""",
        )

    # No contract negotiation — TB2 tasks are already well-specified
    def contract_proposer(self) -> AgentConfig:
        return AgentConfig(system_prompt="", enabled=False)

    def contract_reviewer(self) -> AgentConfig:
        return AgentConfig(system_prompt="", enabled=False)

    def pass_threshold(self) -> float:
        return 8.0

    def max_rounds(self) -> int:
        return 2  # One attempt + one retry max

    def format_build_task(self, user_prompt: str, round_num: int,
                          prev_feedback: str, score_history: list[float]) -> str:
        """Streamlined task prompt — no REFINE/PIVOT overhead for terminal tasks."""
        task = (
            f"Complete this task:\n\n{user_prompt}\n\n"
            f"Read spec.md for the plan. Execute commands with run_bash. "
            f"Verify your work when done."
        )
        if prev_feedback:
            task += (
                f"\n\nYour previous attempt had issues. "
                f"Read feedback.md and fix them. Be precise."
            )
        return task
