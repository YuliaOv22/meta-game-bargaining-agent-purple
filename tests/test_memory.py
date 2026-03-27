"""Unit tests for the markdown memory module."""
import sys
from pathlib import Path

# Add src to path so we can import memory
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import tempfile
import shutil
from memory import MarkdownMemory


@pytest.fixture
def mem(tmp_path):
    """Create a MarkdownMemory instance with a temp directory."""
    return MarkdownMemory(memory_dir=tmp_path)


class TestReadWrite:
    """Tests for basic read/write operations on opponent memory files."""

    def test_read_nonexistent(self, mem):
        """Reading a non-existent opponent should return None."""
        assert mem.read("unknown_opp") is None

    def test_read_for_prompt_nonexistent(self, mem):
        """Prompt text for unknown opponent should indicate no past experience."""
        result = mem.read_for_prompt("unknown_opp")
        assert "No past experience" in result

    def test_add_lesson_creates_file(self, mem):
        """Adding a lesson for a new opponent should create the memory file."""
        mem.add_lesson("tough", "They never accept. Just take their offer.", games=1, agreements=0)
        content = mem.read("tough")
        assert content is not None
        assert "tough" in content.lower()
        assert "They never accept" in content
        assert "## Lessons" in content
        assert "## Summary" in content

    def test_add_lesson_appends(self, mem):
        """Adding multiple lessons should append each one."""
        mem.add_lesson("soft", "Lesson one.", games=1, agreements=1)
        mem.add_lesson("soft", "Lesson two.", games=2, agreements=2)
        content = mem.read("soft")
        assert "Lesson one." in content
        assert "Lesson two." in content

    def test_add_lesson_skips_no_new_lesson(self, mem):
        """Lessons with text 'No new lesson.' should be skipped."""
        mem.add_lesson("opp", "First lesson.", games=1, agreements=0)
        result = mem.add_lesson("opp", "No new lesson.", games=2, agreements=0)
        assert result is False
        content = mem.read("opp")
        assert content.count("- [") == 1  # only one lesson

    def test_stats_updated(self, mem):
        """Game and agreement counts should be written to the header."""
        mem.add_lesson("opp", "Lesson.", games=5, agreements=3)
        content = mem.read("opp")
        assert "Games: 5" in content
        assert "Agreements: 3" in content

    def test_stats_increment(self, mem):
        """Stats should reflect the latest values passed to add_lesson."""
        mem.add_lesson("opp", "Lesson 1.", games=1, agreements=1)
        mem.add_lesson("opp", "Lesson 2.", games=2, agreements=1)
        content = mem.read("opp")
        assert "Games: 2" in content
        assert "Agreements: 1" in content


class TestConsolidation:
    """Tests for the lesson consolidation lifecycle."""

    def test_needs_consolidation_when_over_threshold(self, mem):
        """Consolidation should be triggered when lessons exceed MAX_LESSONS."""
        for i in range(5):
            mem.add_lesson("opp", f"Lesson {i}.", games=i + 1, agreements=i)
        # 5 lessons, threshold is 5 — should NOT need consolidation yet
        result = mem.add_lesson("opp", "Lesson 6.", games=6, agreements=5)
        assert result is True  # 6 > MAX_LESSONS(5), needs consolidation

    def test_apply_consolidation(self, mem):
        """After consolidation, only the new summary and kept lessons should remain."""
        for i in range(6):
            mem.add_lesson("opp", f"Lesson {i}.", games=i + 1, agreements=i)

        mem.apply_consolidation("opp", "This opponent is tough and rarely accepts.", ["Key lesson 1", "Key lesson 2"])
        content = mem.read("opp")
        assert "This opponent is tough" in content
        assert "Key lesson 1" in content
        assert "Key lesson 2" in content
        # Old lessons should be gone
        assert "Lesson 0." not in content
        assert "Lesson 5." not in content

    def test_build_consolidation_prompt(self, mem):
        """Consolidation prompt should include all lessons and the expected format."""
        for i in range(7):
            mem.add_lesson("opp", f"Lesson {i}.", games=i + 1, agreements=i)

        prompt = mem.build_consolidation_prompt("opp")
        assert prompt is not None
        assert "opp" in prompt
        assert "Lesson 0." in prompt
        assert "SUMMARY:" in prompt

    def test_no_consolidation_needed(self, mem):
        """No consolidation prompt should be generated when lessons are below threshold."""
        mem.add_lesson("opp", "Just one.", games=1, agreements=0)
        assert mem.build_consolidation_prompt("opp") is None


class TestLessonPrompt:
    """Tests for building LLM lesson-extraction prompts."""

    def test_build_lesson_prompt_first_game(self, mem):
        """First game prompt should indicate no prior lessons."""
        prompt = mem.build_lesson_prompt("new_opp", "Outcome: timeout\nPayoff: 0")
        assert "first game" in prompt.lower() or "None yet" in prompt
        assert "timeout" in prompt

    def test_build_lesson_prompt_with_history(self, mem):
        """Prompt with existing lessons should include them for context."""
        mem.add_lesson("opp", "They are aggressive.", games=1, agreements=0)
        prompt = mem.build_lesson_prompt("opp", "Outcome: agreement\nPayoff: 200")
        assert "aggressive" in prompt
        assert "agreement" in prompt


class TestParseConsolidation:
    """Tests for parsing LLM consolidation responses."""

    def test_parse_response(self, mem):
        """Parse a well-formed consolidation response into summary and lessons."""
        response = "SUMMARY: Opponent is balanced.\nLESSON: Accept early if > 1.5x BATNA.\nLESSON: Don't concede too fast."
        summary, lessons = mem.parse_consolidation_response(response)
        assert summary == "Opponent is balanced."
        assert len(lessons) == 2
        assert "Accept early" in lessons[0]
        assert "concede" in lessons[1]

    def test_parse_empty(self, mem):
        """Parsing an empty string should return empty summary and lessons."""
        summary, lessons = mem.parse_consolidation_response("")
        assert summary == ""
        assert lessons == []


class TestFilenameSafety:
    """Tests for safe filename generation from opponent names."""

    def test_special_characters(self, mem):
        """Opponent names with special characters should be sanitized for filesystem use."""
        mem.add_lesson("opp/with:special<chars>", "Lesson.", games=1, agreements=0)
        content = mem.read("opp/with:special<chars>")
        assert content is not None
        assert "Lesson." in content
