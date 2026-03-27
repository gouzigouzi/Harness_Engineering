#!/usr/bin/env python3
"""
Harness — three-agent architecture for long-running application development.

Reproduces the design from Anthropic's "Harness design for long-running
application development" using a pure Python + OpenAI-compatible API approach.

Architecture:
  Layer 1 (Harness.run)   — outer loop: plan → build → evaluate → repeat
  Layer 2 (Agent.run)     — inner loop: llm.call → tool use → context management
  Layer 3 (context.py)    — compaction / reset lifecycle

Usage:
  export OPENAI_API_KEY="sk-..."
  export OPENAI_BASE_URL="https://api.openai.com/v1"   # or any compatible endpoint
  export HARNESS_MODEL="gpt-4o"                         # or any model
  python harness.py "Build a fully featured DAW in the browser using the Web Audio API"
"""
from __future__ import annotations

import logging
import os
import re
import sys
import time
from pathlib import Path

import config
import prompts
import tools
from agents import Agent
from skills import SkillRegistry

log = logging.getLogger("harness")


class Harness:
    """
    Orchestrates three agents:
      1. Planner  — expands a short prompt into a full product spec
      2. Builder  — implements the spec, addressing QA feedback each round
      3. Evaluator — tests the build and scores it on 4 criteria

    The build→evaluate loop repeats until the score passes or we hit max rounds.
    Communication between agents is via files in the workspace (spec.md, feedback.md).
    """

    def __init__(self):
        self.skill_registry = SkillRegistry()
        skill_catalog = self.skill_registry.build_catalog_prompt()

        # Inject skill catalog (Level 1: metadata only) into agents that need it.
        # The agents themselves decide when to read_skill_file() for full content.
        self.planner = Agent(
            "planner", prompts.PLANNER_SYSTEM + skill_catalog, use_tools=True,
        )
        self.builder = Agent(
            "builder", prompts.BUILDER_SYSTEM + skill_catalog, use_tools=True,
        )
        self.evaluator = Agent(
            "evaluator", prompts.EVALUATOR_SYSTEM, use_tools=True,
            extra_tool_schemas=tools.BROWSER_TOOL_SCHEMAS,
        )
        # Lightweight agents for contract negotiation (no bash needed)
        self.contract_proposer = Agent(
            "contract_proposer", prompts.CONTRACT_BUILDER_SYSTEM, use_tools=True,
        )
        self.contract_reviewer = Agent(
            "contract_reviewer", prompts.CONTRACT_REVIEWER_SYSTEM, use_tools=True,
        )

    def run(self, user_prompt: str) -> None:
        # Create a unique project subdirectory under workspace
        from datetime import datetime
        slug = re.sub(r'[^a-z0-9]+', '-', user_prompt.lower().strip())[:40].strip('-')
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        project_name = f"{timestamp}_{slug}"
        project_dir = os.path.join(config.WORKSPACE, project_name)

        # Override workspace for this run so all tools operate in the project dir
        config.WORKSPACE = os.path.abspath(project_dir)
        Path(config.WORKSPACE).mkdir(parents=True, exist_ok=True)

        log.info(f"Project directory: {config.WORKSPACE}")

        # Initialize git in project dir
        git_dir = Path(config.WORKSPACE) / ".git"
        if not git_dir.exists():
            os.system(f"cd {config.WORKSPACE} && git init && git add -A 2>/dev/null; git commit -m 'init' --allow-empty 2>/dev/null")

        total_start = time.time()

        # ---- Phase 1: Planning ----
        log.info("=" * 60)
        log.info("PHASE 1: PLANNING")
        log.info("=" * 60)
        phase_start = time.time()

        self.planner.run(
            f"Create a detailed product specification for the following idea:\n\n"
            f"{user_prompt}\n\n"
            f"Save the spec to spec.md."
        )

        plan_duration = time.time() - phase_start
        log.info(f"Planning completed in {plan_duration:.0f}s")

        # ---- Phase 2: Build → Evaluate loop ----
        score_history: list[float] = []

        for round_num in range(1, config.MAX_HARNESS_ROUNDS + 1):

            # ---- Phase 2a: Contract negotiation ----
            log.info("=" * 60)
            log.info(f"ROUND {round_num}/{config.MAX_HARNESS_ROUNDS}: CONTRACT NEGOTIATION")
            log.info("=" * 60)
            contract_start = time.time()

            self._negotiate_contract(round_num)

            contract_duration = time.time() - contract_start
            log.info(f"Contract negotiation completed in {contract_duration:.0f}s")

            # ---- Phase 2b: Build ----
            log.info("=" * 60)
            log.info(f"ROUND {round_num}/{config.MAX_HARNESS_ROUNDS}: BUILD")
            log.info("=" * 60)
            build_start = time.time()

            feedback_path = Path(config.WORKSPACE) / config.FEEDBACK_FILE
            prev_feedback = ""
            if feedback_path.exists():
                prev_feedback = feedback_path.read_text(encoding="utf-8")

            build_task = (
                "Read spec.md and contract.md. Build exactly what the contract specifies.\n"
            )
            if prev_feedback:
                # Build score trend context
                trend_info = ""
                if len(score_history) >= 2:
                    delta = score_history[-1] - score_history[-2]
                    if delta > 0:
                        trend_info = f"Score trend: IMPROVING (+{delta:.1f}). Previous scores: {score_history}"
                    elif delta == 0:
                        trend_info = f"Score trend: STAGNANT. Previous scores: {score_history}"
                    else:
                        trend_info = f"Score trend: DECLINING ({delta:.1f}). Previous scores: {score_history}"
                elif len(score_history) == 1:
                    trend_info = f"Last score: {score_history[0]:.1f}/10"

                build_task += (
                    "\nThe QA evaluator found issues in the previous round. "
                    "Read feedback.md and address every issue.\n"
                    f"\n{trend_info}\n"
                    "\nMAKE A STRATEGIC DECISION before writing any code:\n"
                    "- If scores are trending UP → REFINE: keep the current approach, fix bugs, polish details.\n"
                    "- If scores are STAGNANT or DECLINING → PIVOT: scrap the current aesthetic/architecture "
                    "and try a fundamentally different approach. A pivot means new design language, "
                    "new layout structure, new color palette — not just tweaking the same thing.\n"
                    "\nState your decision (REFINE or PIVOT) and your reasoning BEFORE starting work.\n"
                )
            else:
                build_task += (
                    "\nThis is the first build round. Start from scratch.\n"
                )
            build_task += (
                "\nAfter building, make sure the app compiles/runs without errors. "
                "Commit your work with git.\n"
                "\nREMINDER: You MUST use write_file to create actual source code files. "
                "Do not just read files and respond — write the code."
            )

            self.builder.run(build_task)
            build_duration = time.time() - build_start
            log.info(f"Build round {round_num} completed in {build_duration:.0f}s")

            # ---- Phase 2c: Evaluate ----
            log.info("=" * 60)
            log.info(f"ROUND {round_num}/{config.MAX_HARNESS_ROUNDS}: EVALUATE")
            log.info("=" * 60)
            eval_start = time.time()

            self.evaluator.run(
                f"This is QA round {round_num}.\n"
                f"Read spec.md to understand what was promised.\n"
                f"Read contract.md to see the acceptance criteria for this round.\n"
                f"Examine the codebase (list_files, read_file).\n"
                f"Use browser_test to launch the app and interact with it in a real browser. "
                f"Test each acceptance criterion from the contract.\n"
                f"Score each criterion honestly. Write your evaluation to feedback.md.\n"
                f"Call stop_dev_server when done testing."
            )

            eval_duration = time.time() - eval_start
            log.info(f"Evaluation round {round_num} completed in {eval_duration:.0f}s")

            # Ensure dev server is stopped between rounds
            tools.stop_dev_server()

            # ---- Check score ----
            score = self._extract_score()
            score_history.append(score)
            log.info(f"Round {round_num} average score: {score:.1f} / 10  (threshold: {config.PASS_THRESHOLD})")
            log.info(f"Score history: {score_history}")

            if score >= config.PASS_THRESHOLD:
                log.info(f"PASSED QA at round {round_num}.")
                break
        else:
            log.warning(f"Did not pass QA after {config.MAX_HARNESS_ROUNDS} rounds.")

        total_duration = time.time() - total_start
        log.info("=" * 60)
        log.info(f"HARNESS COMPLETE — total time: {total_duration / 60:.1f} minutes")
        log.info(f"Output in: {config.WORKSPACE}")
        log.info("=" * 60)

    def _negotiate_contract(self, round_num: int, max_iterations: int = 3) -> None:
        """
        Builder proposes a sprint contract, Evaluator reviews it.
        They iterate until the reviewer approves or we hit max_iterations.
        Result is saved to contract.md in the workspace.
        """
        # Step 1: Builder proposes
        self.contract_proposer.run(
            f"This is round {round_num}.\n"
            f"Read spec.md. If feedback.md exists, read it too.\n"
            f"Propose a sprint contract for this round. Write it to contract.md."
        )

        # Step 2: Reviewer iterates
        for i in range(max_iterations):
            log.info(f"[contract] Review iteration {i + 1}/{max_iterations}")

            result = self.contract_reviewer.run(
                f"Review the sprint contract in contract.md for round {round_num}.\n"
                f"Read spec.md for context. Read feedback.md if it exists.\n"
                f"If acceptable, write APPROVED at the top and save to contract.md.\n"
                f"If changes needed, write revision requests and save updated contract to contract.md."
            )

            # Check if approved
            contract_path = Path(config.WORKSPACE) / "contract.md"
            if contract_path.exists():
                contract_text = contract_path.read_text(encoding="utf-8")
                if "APPROVED" in contract_text.upper()[:200]:
                    log.info("[contract] Contract approved.")
                    return

            # If not approved, builder revises
            if i < max_iterations - 1:
                log.info("[contract] Contract needs revision, builder revising...")
                self.contract_proposer.run(
                    f"The reviewer requested changes to the contract.\n"
                    f"Read contract.md to see the revision requests.\n"
                    f"Update the contract and save to contract.md."
                )

        log.warning("[contract] Max iterations reached, proceeding with current contract.")

    def _extract_score(self) -> float:
        """Parse the average score from feedback.md."""
        feedback_path = Path(config.WORKSPACE) / config.FEEDBACK_FILE
        if not feedback_path.exists():
            return 0.0
        text = feedback_path.read_text(encoding="utf-8")
        # Look for "Average: X/10" or "Average: X.X/10"
        match = re.search(r"[Aa]verage[:\s]*(\d+\.?\d*)\s*/\s*10", text)
        if match:
            return float(match.group(1))
        # Fallback: average all X/10 scores found
        scores = re.findall(r"(\d+\.?\d*)\s*/\s*10", text)
        if scores:
            vals = [float(s) for s in scores]
            return sum(vals) / len(vals)
        return 0.0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    from logger import setup_logging
    setup_logging(verbose="--verbose" in sys.argv or "-v" in sys.argv)

    # Remove flags from argv before parsing prompt
    args = [a for a in sys.argv[1:] if a not in ("--verbose", "-v")]

    if not config.API_KEY:
        print("Error: Set OPENAI_API_KEY in .env or environment.")
        sys.exit(1)

    if len(args) < 1:
        print("Usage: python harness.py \"<your product idea>\" [--verbose]")
        print()
        print("Examples:")
        print('  python harness.py "Build a fully featured DAW in the browser using the Web Audio API"')
        print('  python harness.py "Create a 2D retro game maker with level editor, sprite editor, and playable test mode"')
        sys.exit(1)

    user_prompt = " ".join(args)
    log.info(f"Prompt: {user_prompt}")
    log.info(f"Model: {config.MODEL}")
    log.info(f"Base URL: {config.BASE_URL}")
    log.info(f"Workspace: {config.WORKSPACE}")

    # --- Preflight: verify API connectivity ---
    log.info("Verifying API connection...")
    try:
        from agents import get_client
        resp = get_client().chat.completions.create(
            model=config.MODEL,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5,
        )
        log.info(f"API OK — model responded: {resp.choices[0].message.content}")
    except Exception as e:
        log.error(f"API preflight failed: {e}")
        print(f"\nCannot connect to API. Check your .env:\n"
              f"  OPENAI_API_KEY  — is it valid?\n"
              f"  OPENAI_BASE_URL — is {config.BASE_URL} correct?\n"
              f"  HARNESS_MODEL   — does {config.MODEL} exist on this provider?")
        sys.exit(1)

    harness = Harness()
    harness.run(user_prompt)


if __name__ == "__main__":
    main()
