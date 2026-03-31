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

    # micromamba paths — self-contained Python env, no apt-get needed
    MAMBA_ROOT = "/opt/mamba"
    MAMBA_BIN = "/opt/mamba/bin/micromamba"
    MAMBA_ENV = "/opt/mamba/envs/agent"
    MAMBA_PYTHON = "/opt/mamba/envs/agent/bin/python3"

    @staticmethod
    def name() -> str:
        return "harness-agent"

    def __init__(self, model_name: str | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model_name = model_name

    async def install(self, environment: BaseEnvironment) -> None:
        """Install dependencies and clone our repo into the container.

        Strategy (ordered by speed):
        1. If python3 + pip exist -> use them directly (fastest, ~10s)
        2. If python3 exists but no pip -> curl wheel install (~15s)
        3. If no python3 at all -> micromamba install (~30-60s, no apt-get)

        micromamba is a single static binary that installs a complete
        Python environment from conda-forge. No apt-get, no dpkg locks.
        """
        # Step 1: Ensure git is available
        await self.exec_as_root(
            environment,
            command=(
                "command -v git >/dev/null 2>&1 || "
                "( for i in $(seq 1 30); do "
                "    fuser /var/lib/dpkg/lock >/dev/null 2>&1 || break; sleep 2; "
                "  done && "
                "  apt-get update -qq 2>/dev/null && "
                "  apt-get install -y -qq git 2>/dev/null ) || true"
            ),
        )

        # Step 2: Clone repo
        await self.exec_as_agent(
            environment,
            command=(
                "git clone --depth 1 "
                "https://github.com/lazyFrogLOL/Harness_Engineering.git "
                "/home/user/harness-agent"
            ),
        )

        # Step 3: Ensure python3 + openai are available
        await self.exec_as_root(
            environment,
            command=(
                # Fast path: already works
                "python3 -c 'import openai' 2>/dev/null || "
                # Medium path: python3 exists, try pip
                "( command -v python3 >/dev/null 2>&1 && "
                "  ( pip3 install --break-system-packages -q openai 2>/dev/null || "
                "    pip install --break-system-packages -q openai 2>/dev/null || "
                "    python3 -m pip install --break-system-packages -q openai 2>/dev/null ) && "
                "  python3 -c 'import openai' 2>/dev/null ) || "
                # Medium path: python3 exists but no pip, use curl+unzip
                "( command -v python3 >/dev/null 2>&1 && "
                "  SITE=$(python3 -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null) && "
                "  cd /tmp && "
                "  curl -sL -o openai.whl "
                "'https://files.pythonhosted.org/packages/2a/9e/5bfa2270f902d5b92ab7d41ce0475b8630572e71e349b2a4996d14bdda93/openai-2.30.0-py3-none-any.whl' && "
                "  python3 -m zipfile -e openai.whl \"$SITE\" && "
                "  python3 -c 'import openai' 2>/dev/null ) || "
                # Slow path: no python3 at all, use micromamba
                "( echo 'No python3 found, installing via micromamba...' && "
                f"  mkdir -p {self.MAMBA_ROOT} && "
                "  curl -sL https://micro.mamba.pm/api/micromamba/linux-64/latest "
                f"    | tar -xj -C {self.MAMBA_ROOT} --strip-components=1 bin/micromamba && "
                f"  {self.MAMBA_BIN} create -y -q -p {self.MAMBA_ENV} "
                "    -c conda-forge python=3.12 pip && "
                f"  {self.MAMBA_ENV}/bin/pip install -q openai && "
                f"  {self.MAMBA_PYTHON} -c 'import openai; print(\"openai installed via micromamba\")' ) || "
                # Nuclear: standalone python from GitHub + openai wheel
                "( echo 'micromamba failed, trying standalone python...' && "
                "  cd /tmp && "
                "  curl -sL -o python.tar.zst "
                "    'https://github.com/astral-sh/python-build-standalone/releases/download/20250604/cpython-3.12.11+20250604-x86_64-unknown-linux-gnu-install_only.tar.gz' && "
                f"  mkdir -p {self.MAMBA_ROOT} && "
                f"  tar -xf python.tar.zst -C {self.MAMBA_ROOT} --strip-components=1 && "
                f"  {self.MAMBA_ROOT}/bin/python3 -m pip install -q openai && "
                f"  {self.MAMBA_ROOT}/bin/python3 -c 'import openai; print(\"openai installed via standalone python\")' ) || "
                "true"
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

        env_vars.append("HARNESS_WORKSPACE=/app")
        env_vars.append("HARNESS_FLAT_WORKSPACE=1")
        env_prefix = " ".join(env_vars)

        # Auto-detect which python has openai: system > mamba env > standalone
        await self.exec_as_agent(
            environment,
            command=(
                f"cd /home/user/harness-agent && "
                f"if python3 -c 'import openai' 2>/dev/null; then PYTHON=python3; "
                f"elif {self.MAMBA_PYTHON} -c 'import openai' 2>/dev/null; then PYTHON={self.MAMBA_PYTHON}; "
                f"elif {self.MAMBA_ROOT}/bin/python3 -c 'import openai' 2>/dev/null; then PYTHON={self.MAMBA_ROOT}/bin/python3; "
                f"else echo 'FATAL: no working python with openai found' && exit 1; fi && "
                f"{env_prefix} "
                f"$PYTHON harness.py --profile terminal {escaped}"
            ),
        )

    def populate_context_post_run(self, context: AgentContext) -> None:
        """Called after run() completes. Could parse logs if needed."""
        pass
