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
  - Builder gets most of the time budget with env bootstrapping
  - Builder does self-verification before finishing (enforced)
  - Evaluator is quick self-check, 1 round only
  - If first round passes, done. If not, builder gets one retry.
"""
from __future__ import annotations

from profiles.base import BaseProfile, AgentConfig

# Commands to bootstrap environment awareness at the start of each build.
# Output is injected as context so the model doesn't waste time exploring.
ENV_BOOTSTRAP_COMMANDS = [
    "uname -a",
    "pwd",
    "ls -la /app/ 2>/dev/null || echo '/app not found'",
    "ls -la . 2>/dev/null",
    "python3 --version 2>/dev/null; python --version 2>/dev/null",
    "which gcc g++ make cmake 2>/dev/null || true",
    "pip3 list 2>/dev/null | head -30 || true",
    "cat /etc/os-release 2>/dev/null | head -5 || true",
    "df -h / 2>/dev/null | tail -1 || true",
    "free -h 2>/dev/null | head -2 || true",
    "env | grep -iE '^(PATH|HOME|USER|LANG|LC_)' 2>/dev/null || true",
]


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
- Do NOT write long explanations. Just execute and verify.

MANDATORY SELF-VERIFICATION (you MUST do this before stopping):
After completing the task, switch to reviewer mode and verify your work:
1. Re-read the original task requirements from spec.md.
2. For each requirement, run a concrete check command (ls, cat, test, diff, grep, etc.)
3. Ask yourself: "If I were a test script, would this pass?"
4. If ANY check fails, fix it immediately before stopping.
Do NOT skip verification. Do NOT just say "it looks good". Run actual commands.

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
        """Streamlined task prompt with environment bootstrapping."""
        # Collect environment info on first round
        env_section = ""
        if round_num == 1:
            import subprocess, config as _cfg
            env_lines = []
            for cmd in ENV_BOOTSTRAP_COMMANDS:
                try:
                    r = subprocess.run(
                        cmd, shell=True, cwd=_cfg.WORKSPACE,
                        capture_output=True, text=True, timeout=10,
                    )
                    out = (r.stdout + r.stderr).strip()
                    if out:
                        env_lines.append(f"$ {cmd}\n{out}")
                except Exception:
                    pass
            if env_lines:
                env_section = (
                    "\n\n--- ENVIRONMENT INFO (pre-collected, do NOT re-run these) ---\n"
                    + "\n\n".join(env_lines)
                    + "\n--- END ENVIRONMENT INFO ---\n"
                )

        task = (
            f"Complete this task:\n\n{user_prompt}\n\n"
            f"Read spec.md for the plan. Execute commands with run_bash. "
            f"Verify your work when done."
            f"{env_section}"
        )
        if prev_feedback:
            task += (
                f"\n\nYour previous attempt had issues. "
                f"Read feedback.md and fix them. Be precise."
            )
        return task
