"""Load and sync a research-papers corpus (papers.csv + markdown) from GitHub."""

import csv
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# In-corpus citation links: `](../<year>/<id>.md)` or same-directory `](<id>.md)`.
CITATION_LINK_RE = re.compile(r"\]\((?:\.\./\d{4}/)?([^/()\s]+)\.md\)")


@dataclass
class Paper:
    """One paper's metadata, markdown location, and in-corpus citation edges."""

    paper_id: str
    title: str
    authors: str
    submitted: str
    url: str
    abstract: str
    md_path: Path | None = None
    cites: list[str] = field(default_factory=list)
    cited_by: list[str] = field(default_factory=list)


@dataclass
class Corpus:
    """A cloned corpus repo and its loaded papers, keyed by paper id."""

    name: str
    repo_url: str
    clone_dir: Path
    papers: dict[str, Paper] = field(default_factory=dict)

    def sync(self) -> None:
        """Clone the corpus repo if absent, otherwise fast-forward pull."""
        if (self.clone_dir / ".git").exists():
            subprocess.run(
                ["git", "-C", str(self.clone_dir), "pull", "--ff-only"],
                check=True,
                capture_output=True,
            )
        else:
            self.clone_dir.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--depth", "1", self.repo_url, str(self.clone_dir)],
                check=True,
                capture_output=True,
            )
        logging.info("synced %s corpus at %s", self.name, self.clone_dir)
