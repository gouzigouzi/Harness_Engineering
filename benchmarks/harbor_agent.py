"""
Harbor adapter — runs our harness agent on Terminal-Bench 2.0 via Harbor framework.

Harbor has two agent types:
  - External (BaseAgent): agent runs outside container, sends commands via environment.exec()
  - Installed (BaseInstalledAgent): agent is installed inside the container

We use Installed agent — our harness.py runs natively inside the container,
so run_bash just works as subprocess without any bridging.

Usage:
  # Install harbor
  pip install harbor

  # Test on hello-world task
  harbor run -d "terminal-bench@2.0" \
    --agent-import-path benchmarks.harbor_agent:HarnessAgent \
    --task-names hello-world

  # Full benchmark
  harbor run -d "terminal-bench@2.0" \
    --agent-import-path benchmarks.harbor_agent:HarnessAgent

  # With Daytona (no Docker needed locally)
  harbor run -d "terminal-bench@2.0" \
    --agent-import-path benchmarks.harbor_agent:HarnessAgent \
    --env daytona
"""
from __future__ import annotations

import os
import shlex
from pathlib import Path

from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


class HarnessAgent(BaseInstalledAgent):
    """
    Installs our harness inside the Harbor container and runs it
    with --profile terminal for each task.
    """

    @staticmethod
    def name() -> str:
        return "harness-agent"

    def __init__(self, model_name: str | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model_name = model_name

    async def install(self, environment: BaseEnvironment) -> None:
        """Install Python, dependencies, and clone our repo into the container.

        Most TB2 prebuilt images already have python3/pip/git.
        We check first and only install what's missing to save time.
        """
        # Only install missing system deps — most TB2 images already have python3
        await self.exec_as_root(
            environment,
            command=(
                "( command -v python3 && command -v pip3 && command -v git ) >/dev/null 2>&1 || "
                "( apt-get update -qq && apt-get install -y -qq python3 python3-pip git >/dev/null 2>&1 )"
            ),
        )

        # Clone repo + install Python deps in one shot
        await self.exec_as_agent(
            environment,
            command=(
                "git clone --depth 1 "
                "https://github.com/lazyFrogLOL/Harness_Engineering.git "
                "/home/user/harness-agent && "
                "pip3 install --break-system-packages --ignore-installed -q openai tiktoken"
            ),
        )

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Run our harness with --profile terminal on the given task."""
        escaped = shlex.quote(instruction)

        # Build env vars string for the command
        env_vars = []
        for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "HARNESS_MODEL"):
            val = os.environ.get(key)
            if val:
                env_vars.append(f"{key}={shlex.quote(val)}")

        # Force workspace to /app so agent writes outputs where tests expect them
        env_vars.append("HARNESS_WORKSPACE=/app")
        # Skip subdirectory creation — outputs must land directly in /app
        env_vars.append("HARNESS_FLAT_WORKSPACE=1")

        env_prefix = " ".join(env_vars)

        await self.exec_as_agent(
            environment,
            command=(
                f"cd /home/user/harness-agent && "
                f"{env_prefix} "
                f"python3 harness.py --profile terminal {escaped}"
            ),
        )

    def populate_context_post_run(self, context: AgentContext) -> None:
        """Called after run() completes. Could parse logs if needed."""
        pass
