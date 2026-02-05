"""Tests for serialization utilities."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from llm_archive_analysis.models import Archive, Conversation, CostEstimate, Message
from llm_archive_analysis.serializers import write_jsonl, write_summary_report


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def sample_archive():
    """Create a sample archive for testing."""
    messages = [
        Message(id="msg-1", role="user", content="Hello"),
        Message(id="msg-2", role="assistant", content="Hi there!"),
        Message(id="msg-3", role="user", content="How are you?"),
        Message(id="msg-4", role="assistant", content="I'm doing well, thanks!"),
    ]
    conversations = [
        Conversation(
            id="conv-1",
            title="Test Conversation 1",
            started_at=datetime(2024, 1, 1, 12, 0, 0),
            messages=messages[:2],
        ),
        Conversation(
            id="conv-2",
            title="Test Conversation 2",
            started_at=datetime(2024, 1, 2, 14, 0, 0),
            messages=messages[2:],
        ),
    ]
    return Archive(
        source="chatgpt",
        ingested_at=datetime(2024, 1, 3, 10, 0, 0),
        conversations=conversations,
    )


@pytest.fixture
def sample_cost_estimates():
    """Create sample cost estimates."""
    return [
        CostEstimate(
            model_name="gpt-4",
            currency="USD",
            input_cost=0.03,
            output_cost=0.06,
            total_cost=0.09,
            pricing_source="built-in",
            input_tokens=1000,
            output_tokens=1000,
        ),
        CostEstimate(
            model_name="gpt-3.5-turbo",
            currency="USD",
            input_cost=0.001,
            output_cost=0.003,
            total_cost=0.004,
            pricing_source="built-in",
            input_tokens=2000,
            output_tokens=2000,
        ),
    ]


class TestWriteJsonl:
    """Tests for write_jsonl function."""

    def test_write_empty_list(self, temp_output_dir):
        """Test writing an empty list."""
        output_path = temp_output_dir / "empty.jsonl"
        write_jsonl([], output_path)

        assert output_path.exists()
        with open(output_path, "r") as f:
            content = f.read()
            assert content == ""

    def test_write_messages(self, temp_output_dir):
        """Test writing messages to JSONL."""
        messages = [
            Message(id="msg-1", role="user", content="Hello"),
            Message(id="msg-2", role="assistant", content="Hi"),
        ]
        output_path = temp_output_dir / "messages.jsonl"
        write_jsonl(messages, output_path)

        assert output_path.exists()

        # Read and verify
        with open(output_path, "r") as f:
            lines = f.readlines()
            assert len(lines) == 2

            # Verify each line is valid JSON
            for line in lines:
                data = json.loads(line)
                assert "id" in data
                assert "role" in data
                assert "content" in data

    def test_write_conversations(self, temp_output_dir, sample_archive):
        """Test writing conversations to JSONL."""
        output_path = temp_output_dir / "conversations.jsonl"
        write_jsonl(sample_archive.conversations, output_path)

        assert output_path.exists()

        with open(output_path, "r") as f:
            lines = f.readlines()
            assert len(lines) == 2

            # Verify structure
            conv_data = json.loads(lines[0])
            assert "id" in conv_data
            assert "title" in conv_data
            assert "messages" in conv_data

    def test_write_cost_estimates(self, temp_output_dir, sample_cost_estimates):
        """Test writing cost estimates to JSONL."""
        output_path = temp_output_dir / "costs.jsonl"
        write_jsonl(sample_cost_estimates, output_path)

        assert output_path.exists()

        with open(output_path, "r") as f:
            lines = f.readlines()
            assert len(lines) == 2

            # Verify structure
            cost_data = json.loads(lines[0])
            assert "model_name" in cost_data
            assert "total_cost" in cost_data
            assert "input_tokens" in cost_data

    def test_jsonl_format(self, temp_output_dir):
        """Test that output is proper JSONL (one JSON object per line)."""
        messages = [
            Message(id=f"msg-{i}", role="user", content=f"Message {i}")
            for i in range(5)
        ]
        output_path = temp_output_dir / "test.jsonl"
        write_jsonl(messages, output_path)

        with open(output_path, "r") as f:
            lines = f.readlines()
            assert len(lines) == 5

            # Each line should end with newline and be valid JSON
            for line in lines:
                assert line.endswith("\n")
                data = json.loads(line)
                assert isinstance(data, dict)


class TestWriteSummaryReport:
    """Tests for write_summary_report function."""

    def test_write_summary_basic(self, temp_output_dir, sample_archive, sample_cost_estimates):
        """Test writing a basic summary report."""
        output_path = temp_output_dir / "summary.md"
        write_summary_report(sample_archive, sample_cost_estimates, output_path)

        assert output_path.exists()

        with open(output_path, "r") as f:
            content = f.read()

            # Check for key sections
            assert "# LLM Archive Analysis Report" in content
            assert "**Provider:** chatgpt" in content
            assert "**Total conversations:** 2" in content
            assert "**Total messages:** 4" in content

    def test_summary_with_costs(self, temp_output_dir, sample_archive, sample_cost_estimates):
        """Test that cost information is included in summary."""
        output_path = temp_output_dir / "summary.md"
        write_summary_report(sample_archive, sample_cost_estimates, output_path)

        with open(output_path, "r") as f:
            content = f.read()

            # Check for cost section
            assert "## Cost Estimates" in content
            assert "Total estimated cost" in content
            assert "gpt-4" in content
            assert "gpt-3.5-turbo" in content

    def test_summary_no_costs(self, temp_output_dir, sample_archive):
        """Test summary report when no cost estimates are available."""
        output_path = temp_output_dir / "summary.md"
        write_summary_report(sample_archive, [], output_path)

        with open(output_path, "r") as f:
            content = f.read()

            # Should indicate no cost data
            assert "No usage data available" in content

    def test_summary_conversation_list(self, temp_output_dir, sample_archive, sample_cost_estimates):
        """Test that conversations are listed in summary."""
        output_path = temp_output_dir / "summary.md"
        write_summary_report(sample_archive, sample_cost_estimates, output_path)

        with open(output_path, "r") as f:
            content = f.read()

            # Check for conversations section
            assert "## Conversations" in content
            assert "Test Conversation 1" in content
            assert "Test Conversation 2" in content

    def test_summary_many_conversations(self, temp_output_dir, sample_cost_estimates):
        """Test that only first 10 conversations are listed."""
        conversations = [
            Conversation(id=f"conv-{i}", title=f"Conversation {i}", messages=[])
            for i in range(15)
        ]
        archive = Archive(
            source="claude",
            ingested_at=datetime.now(),
            conversations=conversations,
        )

        output_path = temp_output_dir / "summary.md"
        write_summary_report(archive, sample_cost_estimates, output_path)

        with open(output_path, "r") as f:
            content = f.read()

            # Should show first 10 and indicate more
            assert "Conversation 0" in content
            assert "Conversation 9" in content
            assert "... and 5 more conversations" in content

    def test_summary_markdown_table(self, temp_output_dir, sample_archive, sample_cost_estimates):
        """Test that cost table is formatted as markdown."""
        output_path = temp_output_dir / "summary.md"
        write_summary_report(sample_archive, sample_cost_estimates, output_path)

        with open(output_path, "r") as f:
            content = f.read()

            # Check for markdown table structure
            assert "| Model |" in content
            assert "|-------|" in content
            assert "| gpt-4 |" in content
