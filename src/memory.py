"""
Markdown-based opponent memory for the bargaining agent.

Each opponent gets a separate .md file in the memory/opponents/ directory.
Files contain a summary and dated lessons learned from past games.
When lessons exceed a threshold, the LLM consolidates them into an updated summary.
"""

import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

MEMORY_DIR = Path(
    os.environ.get("AGENT_MEMORY_DIR", Path(__file__).parent.parent / "memory")
)
OPPONENTS_DIR = MEMORY_DIR / "opponents"

MAX_LESSONS = 5  # when exceeded, trigger consolidation
KEEP_AFTER_CONSOLIDATION = 2  # keep this many newest lessons after consolidation


LESSON_PROMPT = """You just finished a bargaining game. Analyze what happened and extract ONE concise lesson for future games against this opponent type.

## Game Result
{game_summary}

## Your Previous Lessons About This Opponent
{existing_lessons}

## Instructions
Write exactly ONE lesson in 1-2 sentences. Focus on:
- What strategy worked or failed and WHY
- What you should do differently next time
- Specific actionable insight (not vague advice)

Do NOT repeat lessons you already know. If nothing new was learned, write "No new lesson."

Reply with ONLY the lesson text, nothing else."""


CONSOLIDATION_PROMPT = """You have accumulated too many lessons about opponent "{opp_key}". Consolidate them into an updated summary and keep only the {keep} most important lessons.

## Current Summary
{current_summary}

## All Lessons (oldest first)
{all_lessons}

## Instructions
1. Write an updated **Summary** (2-3 sentences max) that captures the key strategic insights.
2. Select the {keep} most valuable, non-redundant lessons to keep. Prefer recent and actionable ones.

Respond in EXACTLY this format (no extra text):
SUMMARY: <your updated summary>
LESSON: <lesson 1>
LESSON: <lesson 2>"""


def _sanitize_filename(name: str) -> str:
    """Convert an opponent name to a safe filename by replacing special characters."""
    return re.sub(r"[^\w\-]", "_", name).strip("_").lower()


class MarkdownMemory:
    """Per-opponent memory stored as Markdown files with lessons and summary."""

    def __init__(self, memory_dir: Path = OPPONENTS_DIR):
        """Initialize memory storage with the given directory, creating it if needed."""
        self._dir = memory_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path_for(self, opp_key: str) -> Path:
        """Return the file path for a given opponent key."""
        return self._dir / f"{_sanitize_filename(opp_key)}.md"

    def read(self, opp_key: str) -> str | None:
        """Read entire memory file for an opponent. Returns None if no memory exists."""
        path = self._path_for(opp_key)
        if not path.exists():
            return None
        with self._lock:
            return path.read_text(encoding="utf-8")

    def read_for_prompt(self, opp_key: str) -> str:
        """Read memory formatted for injection into agent's system prompt."""
        content = self.read(opp_key)
        if not content:
            return f"No past experience with opponent '{opp_key}'."
        return f"## Memory: past experience with '{opp_key}'\n{content}"

    def _parse_file(self, content: str) -> tuple[str, list[str]]:
        """Parse a memory file into (summary, [lessons])."""
        summary = ""
        lessons = []

        summary_match = re.search(r"## Summary\n(.*?)(?=\n## |\Z)", content, re.DOTALL)
        if summary_match:
            summary = summary_match.group(1).strip()

        lesson_matches = re.findall(r"^- \[.*?\] (.+)$", content, re.MULTILINE)
        lessons = lesson_matches

        return summary, lessons

    def _parse_lessons_with_dates(self, content: str) -> list[str]:
        """Parse full lesson lines including dates."""
        return re.findall(r"^- \[.+$", content, re.MULTILINE)

    def add_lesson(
        self, opp_key: str, lesson: str, games: int = 0, agreements: int = 0
    ) -> bool:
        """
        Add a lesson to opponent's memory file.
        Returns True if consolidation is needed (lessons > MAX_LESSONS).
        """
        if not lesson or lesson.strip().lower() == "no new lesson.":
            return False

        path = self._path_for(opp_key)
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        with self._lock:
            if path.exists():
                content = path.read_text(encoding="utf-8")
                # Update stats in header
                content = self._update_header_stats(content, games, agreements)
                # Append lesson
                if "## Lessons" in content:
                    content = content.rstrip() + f"\n- [{date_str}] {lesson}\n"
                else:
                    content = (
                        content.rstrip() + f"\n\n## Lessons\n- [{date_str}] {lesson}\n"
                    )
            else:
                content = (
                    f"# Opponent: {opp_key}\n"
                    f"Games: {games} | Agreements: {agreements}\n\n"
                    f"## Summary\nNo summary yet.\n\n"
                    f"## Lessons\n- [{date_str}] {lesson}\n"
                )

            path.write_text(content, encoding="utf-8")

            # Check if consolidation needed
            lesson_lines = self._parse_lessons_with_dates(content)
            return len(lesson_lines) > MAX_LESSONS

    def add_lesson_and_increment(
        self, opp_key: str, lesson: str, is_agreement: bool = False
    ) -> bool:
        """Atomically read current stats, increment them, and add a lesson.

        This avoids the TOCTOU race of separate read() + add_lesson() calls.
        Returns True if consolidation is needed (lessons > MAX_LESSONS).
        """
        if not lesson or lesson.strip().lower() == "no new lesson.":
            return False

        path = self._path_for(opp_key)
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        with self._lock:
            if path.exists():
                content = path.read_text(encoding="utf-8")
                # Parse current stats
                games, agreements = 1, (1 if is_agreement else 0)
                m = re.search(r"Games: (\d+) \| Agreements: (\d+)", content)
                if m:
                    games = int(m.group(1)) + 1
                    agreements = int(m.group(2)) + (1 if is_agreement else 0)
                content = self._update_header_stats(content, games, agreements)
                # Append lesson
                if "## Lessons" in content:
                    content = content.rstrip() + f"\n- [{date_str}] {lesson}\n"
                else:
                    content = (
                        content.rstrip() + f"\n\n## Lessons\n- [{date_str}] {lesson}\n"
                    )
            else:
                games = 1
                agreements = 1 if is_agreement else 0
                content = (
                    f"# Opponent: {opp_key}\n"
                    f"Games: {games} | Agreements: {agreements}\n\n"
                    f"## Summary\nNo summary yet.\n\n"
                    f"## Lessons\n- [{date_str}] {lesson}\n"
                )

            path.write_text(content, encoding="utf-8")

            lesson_lines = self._parse_lessons_with_dates(content)
            return len(lesson_lines) > MAX_LESSONS

    def _update_header_stats(self, content: str, games: int, agreements: int) -> str:
        """Update the stats line in the header."""
        if games > 0:
            stats_pattern = r"Games: \d+ \| Agreements: \d+"
            new_stats = f"Games: {games} | Agreements: {agreements}"
            if re.search(stats_pattern, content):
                content = re.sub(stats_pattern, new_stats, content)
        return content

    def apply_consolidation(
        self, opp_key: str, new_summary: str, kept_lessons: list[str]
    ) -> None:
        """Apply LLM consolidation result: replace summary and lessons."""
        path = self._path_for(opp_key)
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        with self._lock:
            if path.exists():
                content = path.read_text(encoding="utf-8")
                # Extract header (everything before ## Summary)
                header_match = re.match(r"(.*?)(?=## Summary)", content, re.DOTALL)
                header = (
                    header_match.group(1)
                    if header_match
                    else f"# Opponent: {opp_key}\n\n"
                )
            else:
                header = f"# Opponent: {opp_key}\n\n"

            # Rebuild file
            lessons_text = "\n".join(
                f"- [{date_str}] {lesson}" for lesson in kept_lessons
            )
            new_content = (
                f"{header.rstrip()}\n\n"
                f"## Summary\n{new_summary}\n\n"
                f"## Lessons\n{lessons_text}\n"
            )

            path.write_text(new_content, encoding="utf-8")

    def build_lesson_prompt(self, opp_key: str, game_summary: str) -> str:
        """Build the prompt for LLM to generate a lesson after a game."""
        content = self.read(opp_key)
        if content:
            _, lessons = self._parse_file(content)
            existing = (
                "\n".join(f"- {lesson}" for lesson in lessons)
                if lessons
                else "None yet."
            )
        else:
            existing = "None yet (first game against this opponent)."

        return LESSON_PROMPT.format(
            game_summary=game_summary,
            existing_lessons=existing,
        )

    def build_consolidation_prompt(self, opp_key: str) -> str | None:
        """Build the prompt for LLM to consolidate lessons. Returns None if not needed."""
        content = self.read(opp_key)
        if not content:
            return None

        summary, _ = self._parse_file(content)
        lesson_lines = self._parse_lessons_with_dates(content)

        if len(lesson_lines) <= MAX_LESSONS:
            return None

        return CONSOLIDATION_PROMPT.format(
            opp_key=opp_key,
            current_summary=summary or "No summary yet.",
            all_lessons="\n".join(lesson_lines),
            keep=KEEP_AFTER_CONSOLIDATION,
        )

    def parse_consolidation_response(self, response: str) -> tuple[str, list[str]]:
        """Parse the LLM's consolidation response into (summary, [lessons])."""
        summary = ""
        lessons = []

        summary_match = re.search(
            r"SUMMARY:\s*(.+?)(?=\nLESSON:|\Z)", response, re.DOTALL
        )
        if summary_match:
            summary = summary_match.group(1).strip()

        lesson_matches = re.findall(r"LESSON:\s*(.+)", response)
        lessons = [item.strip() for item in lesson_matches if item.strip()]

        return summary, lessons


# Global instance
markdown_memory = MarkdownMemory()
