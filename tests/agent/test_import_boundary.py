"""The boundary is physical, not a convention.

`src/agent/` may import `src/core/ports.py` and the standard library, nothing
else. If the policy could reach `src/world/`, it could see the future and the
comparison against the baselines would be worthless.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_DIR = PROJECT_ROOT / "src" / "agent"

# The only project module outside the package the agent is allowed to see.
ALLOWED_PROJECT_MODULE = "src.core.ports"
AGENT_PACKAGE = "src.agent"

# Standard library only. No numpy, no pandas, no h3: those belong to the world.
ALLOWED_TOP_LEVEL = {"__future__", "math", "dataclasses", "typing", "enum", "collections"}

FORBIDDEN_FRAGMENTS = ("world", "engine", "platform", "enrichment", "osmnx", "h3", "pandas", "numpy")


def agent_modules() -> list[Path]:
    files = sorted(AGENT_DIR.rglob("*.py"))
    assert files, "expected python modules under src/agent/"
    return files


def imported_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative imports hide the boundary: %s" % path
            if node.module:
                names.add(node.module)
    return names


def test_agent_imports_only_ports_and_the_standard_library() -> None:
    for path in agent_modules():
        for name in imported_names(path):
            top = name.split(".")[0]
            if top == "src":
                allowed = name == ALLOWED_PROJECT_MODULE or name.startswith(AGENT_PACKAGE + ".")
                assert allowed, (
                    "%s imports %s; only %s and %s.* are allowed"
                    % (path.name, name, ALLOWED_PROJECT_MODULE, AGENT_PACKAGE)
                )
            else:
                assert top in ALLOWED_TOP_LEVEL, "%s imports %s" % (path.name, name)


def test_no_agent_source_line_mentions_a_world_module() -> None:
    for path in agent_modules():
        for name in imported_names(path):
            for fragment in FORBIDDEN_FRAGMENTS:
                assert fragment not in name.lower(), "%s imports %s" % (path.name, name)


def test_importing_the_agent_does_not_pull_the_world_into_memory() -> None:
    """The automated version of the check: import the package in a clean
    interpreter and look at what actually landed in `sys.modules`."""
    code = (
        "import sys;"
        "import src.agent, src.agent.baseline, src.agent.smart;"
        "leaked = sorted(m for m in sys.modules"
        " if m.startswith(('src.world', 'src.engine', 'src.platform', 'src.enrichment')));"
        "print('LEAKED=' + ','.join(leaked));"
        "print('THIRDPARTY=' + ','.join(sorted("
        "m for m in ('numpy', 'pandas', 'h3', 'osmnx', 'shapely', 'networkx') if m in sys.modules)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "LEAKED=\n" in result.stdout or "LEAKED=" in result.stdout
    leaked = [line for line in result.stdout.splitlines() if line.startswith("LEAKED=")][0]
    third_party = [line for line in result.stdout.splitlines() if line.startswith("THIRDPARTY=")][0]
    assert leaked == "LEAKED=", leaked
    assert third_party == "THIRDPARTY=", third_party
