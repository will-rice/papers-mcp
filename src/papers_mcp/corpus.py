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
    markdown: str = ""
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
        """Clone the corpus repo if absent, otherwise hard-reset to the latest origin HEAD.

        A reset-to-fetched-ref mirror (rather than `pull --ff-only`) is immune to the
        upstream repo ever force-pushing, which would otherwise fail every refresh forever.
        """
        if (self.clone_dir / ".git").exists():
            self._git(["-C", str(self.clone_dir), "fetch", "--depth", "1", "origin", "HEAD"])
            self._git(["-C", str(self.clone_dir), "reset", "--hard", "FETCH_HEAD"])
        else:
            self.clone_dir.parent.mkdir(parents=True, exist_ok=True)
            self._git(["clone", "--depth", "1", self.repo_url, str(self.clone_dir)])
        logging.info("synced %s corpus at %s", self.name, self.clone_dir)

    def _git(self, args: list[str]) -> None:
        """Run a git command, surfacing its stderr on failure instead of swallowing it."""
        try:
            subprocess.run(["git", *args], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"git {args} failed for {self.name}: {exc.stderr.strip()}") from exc

    def load(self) -> None:
        """Load papers.csv, locate corpus markdown files, and build the citation graph."""
        papers: dict[str, Paper] = {}
        with (self.clone_dir / "papers.csv").open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                papers[row["arxiv_id"]] = Paper(
                    paper_id=row["arxiv_id"],
                    title=row["title"],
                    authors=row["authors"],
                    submitted=row["submitted"],
                    url=row["url"],
                    abstract=row["abstract"],
                )

        for md_path in sorted(self.clone_dir.glob("papers/*/*.md")):
            paper = papers.get(md_path.stem)  # skips per-year README.md files
            if paper:
                paper.md_path = md_path

        for paper in papers.values():
            if paper.md_path is None:
                continue
            paper.markdown = paper.md_path.read_text(encoding="utf-8")
            for cited_id in CITATION_LINK_RE.findall(paper.markdown):
                if (
                    cited_id != paper.paper_id
                    and cited_id in papers
                    and cited_id not in paper.cites
                ):
                    paper.cites.append(cited_id)
        for paper in papers.values():
            for cited_id in paper.cites:
                papers[cited_id].cited_by.append(paper.paper_id)

        self.papers = papers
        logging.info("loaded %d papers for %s corpus", len(papers), self.name)
