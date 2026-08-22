"""The documentation must not teach anything that does not work.

Documentation rots faster than code, and it rots silently: nothing fails when a
page keeps describing a function that was renamed a month ago, so the first
person to find out is a reader following instructions that do not work. These
tests run what the pages claim.

Two kinds of check. Every code block that is meant to be runnable is executed
against the installed package. Every internal link is resolved. Neither needs a
graphics processor.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"

# Blocks that legitimately cannot run in a test process. Each entry is a
# fragment that identifies the block, with the reason it is exempt — an
# exemption list nobody has to justify is a list that grows until it covers
# everything.
UNRUNNABLE = {
    # Needs sample clips that are not in version control.
    "face.mp4": "reads a sample video that is downloaded, not committed",
    "baby.mp4": "reads a sample video that is downloaded, not committed",
    "guitar.mp4": "reads a clip the reader supplies",
    "machine.mp4": "reads a clip the reader supplies",
    "long.mp4": "reads a clip the reader supplies",
    "clip.mp4": "reads a clip the reader supplies",
    # Needs hardware or a package that may not be present.
    "import torch": "PyTorch is not a dependency of this project",
    "from vidmag.cuda import ops": "needs the compiled CUDA extension",
    "from vidmag.cuda.array import": "needs the compiled CUDA extension",
    # Illustrative rather than executable.
    "my_operations": "a placeholder for the reader's own code",
}


def _python_blocks(path: Path) -> list[tuple[int, str]]:
    """Every fenced Python block in a page, with the line it starts on."""
    blocks: list[tuple[int, str]] = []
    lines = path.read_text().splitlines()
    body: list[str] = []
    inside, start = False, 0
    for number, line in enumerate(lines, start=1):
        if not inside and re.match(r"^```(python|py)\s*$", line.strip()):
            inside, start, body = True, number, []
        elif inside and line.strip() == "```":
            blocks.append((start, "\n".join(body)))
            inside = False
        elif inside:
            body.append(line)
    return blocks


def _pages() -> list[Path]:
    return sorted(p for p in DOCS.rglob("*.md") if "dev" not in p.parts)


def _exemption(code: str) -> str | None:
    for marker, reason in UNRUNNABLE.items():
        if marker in code:
            return reason
    return None


@pytest.mark.parametrize("page", _pages(), ids=lambda p: str(p.relative_to(DOCS)))
def test_every_runnable_example_actually_runs(page: Path):
    """Execute the examples a reader would copy.

    A page that shows a call which no longer exists is worse than a page that
    shows nothing: the reader trusts it and loses time proving it wrong.
    """
    for line, code in _python_blocks(page):
        reason = _exemption(code)
        if reason:
            continue
        try:
            exec(compile(code, f"{page.name}:{line}", "exec"), {"__name__": "__doc__"})
        except Exception as exc:
            pytest.fail(
                f"{page.relative_to(DOCS)} line {line}: the example fails with "
                f"{type(exc).__name__}: {exc}\n\n{code}"
            )


@pytest.mark.parametrize("page", _pages(), ids=lambda p: str(p.relative_to(DOCS)))
def test_internal_links_point_at_pages_that_exist(page: Path):
    """A broken link inside the site.

    The site build also rejects these, but it only runs where the site
    generator is installed; this runs everywhere the suite does.
    """
    text = page.read_text()
    broken = []
    for target in re.findall(r"\]\(([^)]+)\)", text):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        resolved = (page.parent / target.split("#")[0]).resolve()
        if not resolved.exists():
            broken.append(target)
    assert not broken, f"{page.relative_to(DOCS)} links to missing pages: {broken}"


def test_every_page_is_in_the_navigation():
    """A page nobody links to is a page nobody reads.

    Skipped where the site generator is not installed, since its configuration
    file is what is being read. The strict site build enforces the same thing
    wherever it runs, so this is a convenience, not the only guard.
    """
    yaml = pytest.importorskip(
        "yaml", reason="needs the site generator's dependencies: pip install .[docs]"
    )

    config = yaml.safe_load(
        (DOCS.parent / "mkdocs.yml").read_text().replace("!!python/name:", "")
    )

    listed: set[str] = set()

    def walk(entry) -> None:
        if isinstance(entry, str):
            listed.add(entry)
        elif isinstance(entry, dict):
            for value in entry.values():
                walk(value)
        elif isinstance(entry, list):
            for value in entry:
                walk(value)

    walk(config["nav"])
    actual = {str(p.relative_to(DOCS)) for p in _pages()}
    missing = sorted(actual - listed)
    assert not missing, f"pages not reachable from the navigation: {missing}"
