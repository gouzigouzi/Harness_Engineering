"""
Terminal task profile — optimized for Terminal-Bench-2.

Key constraints:
  - 30 min (1800s) hard timeout per task
  - Tasks are well-defined CLI problems, not open-ended
  - No UI, no browser testing needed
  - Correctness is binary: tests pass or fail

All tunable parameters are read via self.cfg.resolve(), so you can override
them without touching this file:

  # Via environment variables:
  PROFILE_TERMINAL_TASK_BUDGET=1800
  PROFILE_TERMINAL_PLANNER_BUDGET=120
  PROFILE_TERMINAL_PASS_THRESHOLD=8.0
  PROFILE_TERMINAL_LOOP_FILE_EDIT_THRESHOLD=4
  PROFILE_TERMINAL_TIME_WARN_THRESHOLD=0.45

  # Or via ProfileConfig in code:
  from profiles.base import ProfileConfig
  cfg = ProfileConfig(task_budget=1200, pass_threshold=9.0)
  profile = TerminalProfile(cfg=cfg)
"""
from __future__ import annotations

from profiles.base import BaseProfile, AgentConfig, ProfileConfig
from middlewares import (
    LoopDetectionMiddleware,
    PreExitVerificationMiddleware,
    TimeBudgetMiddleware,
    TaskTrackingMiddleware,
    ErrorGuidanceMiddleware,
    SkeletonDetectionMiddleware,
)

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
    # Git context — many tasks involve git repos
    "git -C /app log --oneline -5 2>/dev/null || true",
    "git -C /app status --short 2>/dev/null || true",
    "git -C /app branch -a 2>/dev/null | head -10 || true",
    # Service detection — tasks may need running services
    "which qemu-system-x86_64 qemu-system-i386 docker postfix 2>/dev/null || true",
    "ss -tlnp 2>/dev/null | head -10 || netstat -tlnp 2>/dev/null | head -10 || true",
]


class TerminalProfile(BaseProfile):

    # --- Default values (overridable via ProfileConfig or env vars) ---
    _DEFAULTS = {
        "task_budget": 1800,
        "planner_budget": 120,
        "evaluator_budget": 180,
        "pass_threshold": 8.0,
        "max_rounds": 2,
        "loop_file_edit_threshold": 3,
        "loop_command_repeat_threshold": 3,
        "task_tracking_nudge_after": 5,
        "time_warn_threshold": 0.40,
        "time_critical_threshold": 0.70,
    }

    def _get(self, key: str):
        """Resolve a config value: env var > ProfileConfig > default."""
        return self.cfg.resolve(key, self.name(), self._DEFAULTS[key])

    @property
    def _builder_budget(self) -> float:
        return self._get("task_budget") - self._get("planner_budget") - self._get("evaluator_budget")

    def name(self) -> str:
        return "terminal"

    def description(self) -> str:
        return "Solve terminal/CLI tasks (Terminal-Bench-2 style)"

    # --- TB2 task metadata for dynamic timeout ---
    _tb2_tasks: dict | None = None

    @classmethod
    def _load_tb2_tasks(cls) -> dict:
        """Load TB2 task metadata from bundled JSON."""
        if cls._tb2_tasks is None:
            import json
            from pathlib import Path
            tb2_path = Path(__file__).parent.parent / "benchmarks" / "tb2_tasks.json"
            if tb2_path.exists():
                cls._tb2_tasks = json.loads(tb2_path.read_text(encoding="utf-8"))
            else:
                cls._tb2_tasks = {}
        return cls._tb2_tasks

    def resolve_task_timeout(self, user_prompt: str) -> float | None:
        """Look up TB2 task timeout by matching task name in prompt or workspace path."""
        meta = self._lookup_task_meta(user_prompt)
        return meta.get("agent_timeout_sec") if meta else None

    def _lookup_task_meta(self, user_prompt: str) -> dict | None:
        """Look up full TB2 task metadata (timeout, difficulty, category)."""
        import config as _cfg
        tasks = self._load_tb2_tasks()
        if not tasks:
            return None

        # Check workspace path first (most reliable)
        ws_lower = _cfg.WORKSPACE.lower()
        for task_name, meta in tasks.items():
            if task_name in ws_lower:
                return meta

        # Check user prompt
        prompt_lower = user_prompt.lower()
        for task_name, meta in tasks.items():
            if len(task_name) > 6 and (
                task_name in prompt_lower or
                task_name.replace("-", " ") in prompt_lower or
                task_name.replace("-", "_") in prompt_lower
            ):
                return meta

        return None

    def resolve_time_allocation(self, user_prompt: str) -> dict:
        """Dynamic time allocation based on TB2 task timeout and difficulty.

        Key insight from TB2 leaderboard analysis: top agents (ForgeCode, Letta,
        Claude Code) are all single-agent architectures. Every second spent on
        planner/evaluator LLM calls is a second the builder can't use for actual
        work. For TB2's binary pass/fail verification, the builder's own
        PreExitVerificationMiddleware + running tests is more effective than
        a separate evaluator agent.

        Strategy:
        - <= 900s: Skip both planner and evaluator. Builder gets 100% of time.
        - <= 1800s: Skip planner. Evaluator only on round 2 if needed.
        - > 1800s: Keep planner (complex tasks benefit from decomposition).
                    Evaluator enabled for multi-round correction.
        """
        meta = self._lookup_task_meta(user_prompt)
        timeout = meta.get("agent_timeout_sec") if meta else self._get("task_budget")
        difficulty = meta.get("difficulty", "medium") if meta else "medium"

        if timeout <= 900:
            # Short tasks: every second counts. Builder only.
            # PreExitVerificationMiddleware handles verification internally.
            return {
                "planner": 0.0,
                "builder": 1.0,
                "evaluator": 0.0,
                "planner_enabled": False,
                "evaluator_enabled": False,
            }
        elif timeout <= 1800:
            # Medium tasks: skip planner, builder gets all time.
            # PreExitVerificationMiddleware is more effective than a separate
            # evaluator for TB2's binary pass/fail verification.
            return {
                "planner": 0.0,
                "builder": 1.0,
                "evaluator": 0.0,
                "planner_enabled": False,
                "evaluator_enabled": False,
            }
        else:
            # Long tasks (>30 min): skip planner still — direct execution is
            # faster. But keep evaluator for a potential round 2 fix pass.
            return {
                "planner": 0.0,
                "builder": 0.90,
                "evaluator": 0.10,
                "planner_enabled": False,
                "evaluator_enabled": True,
            }

    def planner(self) -> AgentConfig:
        return AgentConfig(
            system_prompt="""\
You are a quick task planner for a terminal/CLI task. \
You are running autonomously — NEVER ask questions, just plan and execute.

Workflow:
1. DISCOVER: Use list_files and run_bash to understand the environment:
   - What files exist in the workspace?
   - Are there existing tests, scripts, or Makefiles?
   - What does the task actually require?
2. PLAN: Based on what you found, write a brief step-by-step plan.
3. DECOMPOSE: If the task has multiple independent parts, mark which steps \
can be delegated to sub-agents via delegate_task. Use this format:

```
## Plan

### Step 1: [description]
- Command: ...
- Verify: ...

### Step 2: [description] [DELEGATE]
- Delegate to sub-agent with role: "module_writer"
- Task: "Write the X module that does Y, save to Z"
- Verify: ...

### Step 3: [description] [DELEGATE]
- Delegate to sub-agent with role: "parser_writer"
- Task: "Write a parser for X format, save to Z"
- Verify: ...

### Step 4: [description]
- Command: integrate outputs from steps 2-3
- Verify: ...
```

Plan rules:
- Keep it SHORT — 5-10 steps max.
- Be specific: list exact commands, file paths, tools needed.
- Mark steps as [DELEGATE] only if they are truly independent \
(no dependency on other delegate steps).
- Note how to VERIFY each step.

Use write_file to save the plan to spec.md, then stop.
""",
            time_budget=self._get("planner_budget"),
        )

    def builder(self) -> AgentConfig:
        builder_budget = self._builder_budget
        return AgentConfig(
            system_prompt="""\
You are an expert Linux system administrator and developer. \
Complete the given task by executing shell commands.

## EXECUTION MODE
You are running AUTONOMOUSLY with NO human. NEVER ask questions. NEVER say \
"I need more information". Just DO IT. If ambiguous, pick the most reasonable \
interpretation and execute.

## MANDATORY WORKFLOW (follow this EXACT order)

### Step 1: DISCOVER (spend ≤10% of your time here)
- Run `ls -la` to see what files exist in the workspace.
- Read any existing code files, READMEs, Makefiles, or skeleton files.
- Identify: What EXACTLY must be produced? What files, what format, what behavior?
- If skeleton/template files exist with TODO markers, you MUST fill them in.

### Step 2: IDENTIFY OUTPUT CONTRACT
Before writing ANY code, state to yourself:
- What files must exist when I'm done? (exact paths)
- What must each file contain or do?
- How will automated tests verify my work?
This is the MOST IMPORTANT step. Getting the output contract wrong = guaranteed failure.

### Step 3: BUILD
- Your PRIMARY tool is run_bash. Execute commands, don't describe them.
- If you finish without running any commands, you have FAILED.
- For code tasks: write code with write_file, then BUILD and TEST it with run_bash.
- For skeleton/template tasks: READ the existing files FIRST, then MODIFY them \
to fill in the TODO sections. Do NOT create new files that duplicate existing ones.
- NEVER leave TODO, FIXME, NotImplementedError, or placeholder code in output files.

### Step 4: VERIFY (mandatory before stopping)
- Run the SAME commands the test script would run.
- Check that ALL required output files exist: `ls -la <expected_file>`
- Check file contents match spec: `cat <file>` or `head -20 <file>`
- If the task has a test/benchmark script, RUN IT: `python3 benchmark.py` etc.

## CRITICAL RULES
- Follow task specifications LITERALLY — exact file names, exact output \
formats, exact paths. Do not improvise or rename things.
- If the task says "write output to result.txt", it means EXACTLY result.txt.
- If the task specifies a format, match it character-for-character.
- If skeleton files exist with TODO markers, FILL THEM IN. Do not create \
separate files that ignore the skeleton structure.
- Think: "If a test script checks for this, would it pass?"

## WHEN THINGS GO WRONG
- Command not found → install it: `apt-get update && apt-get install -y <pkg>` \
or `pip install <pkg>`.
- Command times out → retry with larger timeout parameter.
- Approach failing after 2 attempts → STOP. Re-read the error. Try a \
COMPLETELY DIFFERENT strategy. Do NOT keep tweaking the same broken approach.
- Read error messages carefully — they tell you exactly what's wrong.
- For unfamiliar domains → use web_search BEFORE coding. 5 min of research \
saves 10 min of dead-end coding.

## BACKGROUND PROCESSES & SERVICES
- Start in background: `nohup <cmd> &` or `<cmd> &`
- Wait for readiness: poll with `sleep 2 && curl ...` or `ss -tlnp | grep <port>`
- QEMU VMs need 15-30 seconds to boot before interacting.
- If a task needs a service running during verification, keep it running.

## AVAILABLE TOOLS
- run_bash: Execute shell commands (your PRIMARY tool).
- write_file / edit_file / read_file / list_files: File operations.
  - edit_file: PREFERRED for modifying existing files (old_string → new_string replacement).
  - write_file: For creating NEW files or complete rewrites.
  - IMPORTANT: For skeleton/template files with TODO markers, use edit_file to replace \
the TODO block with your implementation. Do NOT rewrite the entire file with write_file.
- delegate_task: Spawn isolated sub-agent for independent subtasks.
- web_search / web_fetch: Search web for docs, algorithms, examples.
- read_skill_file: Load a skill guide if relevant (see catalog below).
""",
            middlewares=[
                SkeletonDetectionMiddleware(),
                LoopDetectionMiddleware(
                    file_edit_threshold=self._get("loop_file_edit_threshold"),
                    command_repeat_threshold=self._get("loop_command_repeat_threshold"),
                ),
                ErrorGuidanceMiddleware(),
                TaskTrackingMiddleware(
                    nudge_after_n_tools=self._get("task_tracking_nudge_after"),
                ),
                PreExitVerificationMiddleware(
                    verification_prompt=(
                        "STOP. Switch to REVIEWER mode. Forget what you think you did.\n\n"
                        "Run these checks IN ORDER:\n"
                        "1. `ls -la` — verify ALL required output files exist.\n"
                        "2. For each required file, `cat <file>` or `head -30 <file>` — "
                        "verify it has real content (not empty, not placeholder).\n"
                        "3. `grep -r 'TODO\\|NotImplementedError\\|FIXME' *.py *.c 2>/dev/null` — "
                        "if ANY match, you MUST fix them.\n"
                        "4. If the task has a test/benchmark script, RUN IT NOW.\n"
                        "5. Compare ACTUAL output against what the task asked for.\n"
                        "6. If ANY check fails, fix it BEFORE stopping."
                    ),
                    include_task_requirements=True,
                ),
                TimeBudgetMiddleware(
                    budget_seconds=self._get("task_budget"),
                    warn_threshold=self._get("time_warn_threshold"),
                    critical_threshold=self._get("time_critical_threshold"),
                ),
            ],
            time_budget=builder_budget,
        )

    def evaluator(self) -> AgentConfig:
        return AgentConfig(
            system_prompt="""\
You are a quick verifier. Check if the task was done correctly.

Rules:
- Read spec.md for what should have been done.
- Run 2-3 verification commands with run_bash (ls, cat, test, diff, etc.)
- Check EXACT file paths, output formats, and behavior against the task spec.
- Score Correctness 0-10. Be honest but fast.
- Write a SHORT evaluation to feedback.md. No essays.

Format for feedback.md:
```
## Verification
- Correctness: X/10 — [one sentence]
- **Average: X/10**
### Issues: [list if any, with exact details of what's wrong]
```

Use write_file to save to feedback.md, then stop.
""",
            time_budget=self._get("evaluator_budget"),
        )

    # No contract negotiation — TB2 tasks are already well-specified
    def contract_proposer(self) -> AgentConfig:
        return AgentConfig(system_prompt="", enabled=False)

    def contract_reviewer(self) -> AgentConfig:
        return AgentConfig(system_prompt="", enabled=False)

    def pass_threshold(self) -> float:
        return self._get("pass_threshold")

    def max_rounds(self) -> int:
        return self._get("max_rounds")

    def format_build_task(self, user_prompt: str, round_num: int,
                          prev_feedback: str, score_history: list[float]) -> str:
        """Streamlined task prompt with environment bootstrapping and difficulty-aware hints."""
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

        # Build difficulty-aware strategy hint
        strategy_hint = ""
        meta = self._lookup_task_meta(user_prompt)
        if meta:
            difficulty = meta.get("difficulty", "unknown")
            timeout = meta.get("agent_timeout_sec", 900)
            alloc = self.resolve_time_allocation(user_prompt)
            builder_time = int(timeout * alloc.get("builder", 0.85) / 60)
            total_mins = int(timeout / 60)

            if difficulty == "hard" or timeout >= 1800:
                strategy_hint = (
                    f"\n\n--- TIME BUDGET: ~{builder_time} min (difficulty: {difficulty}) ---\n"
                    "Complex task. Break into independent subtasks if possible. "
                    "Use delegate_task for isolated pieces. "
                    "Get a WORKING solution first, then optimize. "
                    "If stuck after 3 tries, pivot to a different approach.\n"
                    "--- END ---\n"
                )
            elif difficulty == "easy":
                strategy_hint = (
                    f"\n\n--- TIME BUDGET: ~{builder_time} min (difficulty: {difficulty}) ---\n"
                    "Straightforward task. Execute directly — don't overthink.\n"
                    "--- END ---\n"
                )
            else:
                strategy_hint = (
                    f"\n\n--- TIME BUDGET: ~{builder_time} min (difficulty: {difficulty}) ---\n"
                    "Work methodically: implement, test, verify.\n"
                    "--- END ---\n"
                )

        # Direct task injection — no need to read spec.md for TB2 tasks
        task = (
            f"## YOUR TASK\n\n{user_prompt}\n\n"
            f"## INSTRUCTIONS\n"
            f"1. First run `ls -la` to see what files already exist in the workspace.\n"
            f"2. Read any existing code files — look for TODO markers or skeleton code.\n"
            f"3. If skeleton files exist, FILL THEM IN. Do not create separate files.\n"
            f"4. Implement the solution using run_bash and write_file.\n"
            f"5. Test your work before stopping."
            f"{env_section}"
            f"{strategy_hint}"
        )

        # Auto-inject matching skill content (if any)
        skill_section = self._match_and_load_skill(user_prompt)
        if skill_section:
            task += skill_section

        if prev_feedback:
            task += (
                f"\n\n## PREVIOUS ATTEMPT FAILED\n"
                f"Read feedback.md and fix the issues. Here's a summary:\n"
                f"{prev_feedback[:2000]}"
            )
        return task

    def _match_and_load_skill(self, user_prompt: str) -> str:
        """Auto-match a skill to the current task and return its content for injection.

        Matching strategy (first match wins):
        1. Exact task name match: skill directory name appears in workspace path
        2. Keyword overlap: skill description words overlap with task prompt

        Returns the skill content wrapped in a section header, or empty string.
        Only injects ONE skill to avoid context bloat.
        """
        import config as _cfg
        from pathlib import Path

        skills_dir = Path(__file__).parent.parent / "skills"
        if not skills_dir.is_dir():
            return ""

        ws_lower = _cfg.WORKSPACE.lower()
        prompt_lower = user_prompt.lower()

        # Sort skills by name length DESCENDING so longer (more specific) names
        # match first. This prevents "path-tracing" from matching before
        # "path-tracing-reverse" when the task is path-tracing-reverse.
        skill_dirs = sorted(
            [d for d in skills_dir.iterdir() if d.is_dir()],
            key=lambda d: len(d.name),
            reverse=True,
        )

        # Strategy 1: Name match against workspace path (longest match first)
        for skill_dir in skill_dirs:
            skill_name = skill_dir.name
            if len(skill_name) > 5 and skill_name in ws_lower:
                return self._load_skill_content(skill_dir / "SKILL.md", skill_name)

        # Strategy 2: Name match against prompt text (longest match first)
        for skill_dir in skill_dirs:
            skill_name = skill_dir.name
            if len(skill_name) > 5 and (
                skill_name in prompt_lower
                or skill_name.replace("-", " ") in prompt_lower
                or skill_name.replace("-", "_") in prompt_lower
            ):
                return self._load_skill_content(skill_dir / "SKILL.md", skill_name)

        return ""

    @staticmethod
    def _load_skill_content(skill_path, skill_name: str) -> str:
        """Load a SKILL.md file and wrap it for injection into the task prompt."""
        from pathlib import Path
        p = Path(skill_path)
        if not p.exists():
            return ""
        content = p.read_text(encoding="utf-8", errors="replace")
        # Strip YAML frontmatter
        import re
        content = re.sub(r"^---\s*\n.*?\n---\s*\n", "", content, flags=re.DOTALL)
        # Cap at 12000 chars to avoid context bloat
        if len(content) > 12000:
            content = content[:12000] + "\n... (skill guide truncated)"
        return (
            f"\n\n--- SKILL GUIDE: {skill_name} (auto-loaded, follow this guidance) ---\n"
            f"{content.strip()}\n"
            f"--- END SKILL GUIDE ---\n"
        )
