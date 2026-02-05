"""Tests for archive parsers."""

from datetime import datetime
from pathlib import Path

import pytest

from llm_archive_analysis.parsers import (
    parse_archive,
    parse_chatgpt_export,
    parse_claude_export,
)


@pytest.fixture
def fixtures_dir():
    """Get the fixtures directory path."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def chatgpt_sample_file(fixtures_dir):
    """Path to ChatGPT sample file."""
    return fixtures_dir / "chatgpt_sample.json"


@pytest.fixture
def claude_sample_file(fixtures_dir):
    """Path to Claude sample file."""
    return fixtures_dir / "claude_sample.json"


class TestChatGPTParser:
    """Tests for ChatGPT parser."""

    def test_parse_chatgpt_export(self, chatgpt_sample_file):
        """Test parsing a ChatGPT export file."""
        archive = parse_chatgpt_export(chatgpt_sample_file)

        assert archive.source == "chatgpt"
        assert isinstance(archive.ingested_at, datetime)
        assert len(archive.conversations) == 2

    def test_chatgpt_conversation_parsing(self, chatgpt_sample_file):
        """Test that conversations are parsed correctly."""
        archive = parse_chatgpt_export(chatgpt_sample_file)

        # Check first conversation
        conv1 = archive.conversations[0]
        assert conv1.id == "conv-123"
        assert conv1.title == "Test Conversation"
        assert conv1.started_at is not None
        assert len(conv1.messages) == 4

        # Check second conversation
        conv2 = archive.conversations[1]
        assert conv2.id == "conv-456"
        assert conv2.title == "Python Help"

    def test_chatgpt_message_parsing(self, chatgpt_sample_file):
        """Test that messages are parsed correctly."""
        archive = parse_chatgpt_export(chatgpt_sample_file)
        conv = archive.conversations[0]

        # Check first message
        msg1 = conv.messages[0]
        assert msg1.id == "msg-1"
        assert msg1.role == "user"
        assert "Hello, how are you?" in msg1.content
        assert msg1.created_at is not None

        # Check second message (assistant with model)
        msg2 = conv.messages[1]
        assert msg2.id == "msg-2"
        assert msg2.role == "assistant"
        assert msg2.model_usage is not None
        assert msg2.model_usage.model_name == "gpt-4"

    def test_chatgpt_model_metadata(self, chatgpt_sample_file):
        """Test that model metadata is extracted."""
        archive = parse_chatgpt_export(chatgpt_sample_file)

        # Find assistant messages with model metadata
        assistant_msgs = [
            msg
            for conv in archive.conversations
            for msg in conv.messages
            if msg.role == "assistant" and msg.model_usage
        ]

        assert len(assistant_msgs) > 0
        assert any(msg.model_usage.model_name == "gpt-4" for msg in assistant_msgs)
        assert any(msg.model_usage.model_name == "gpt-3.5-turbo" for msg in assistant_msgs)


class TestClaudeParser:
    """Tests for Claude parser."""

    def test_parse_claude_export(self, claude_sample_file):
        """Test parsing a Claude export file."""
        archive = parse_claude_export(claude_sample_file)

        assert archive.source == "claude"
        assert isinstance(archive.ingested_at, datetime)
        assert len(archive.conversations) == 2

    def test_claude_conversation_parsing(self, claude_sample_file):
        """Test that conversations are parsed correctly."""
        archive = parse_claude_export(claude_sample_file)

        # Check first conversation
        conv1 = archive.conversations[0]
        assert conv1.id == "claude-conv-789"
        assert conv1.title == "AI Discussion"
        assert conv1.started_at is not None
        assert len(conv1.messages) == 4

        # Check second conversation
        conv2 = archive.conversations[1]
        assert conv2.id == "claude-conv-012"
        assert conv2.title == "Code Review"

    def test_claude_message_parsing(self, claude_sample_file):
        """Test that messages are parsed correctly."""
        archive = parse_claude_export(claude_sample_file)
        conv = archive.conversations[0]

        # Check first message (human)
        msg1 = conv.messages[0]
        assert msg1.id == "msg-claude-1"
        assert msg1.role == "user"
        assert "What is artificial intelligence?" in msg1.content
        assert msg1.created_at is not None

        # Check second message (assistant with model)
        msg2 = conv.messages[1]
        assert msg2.id == "msg-claude-2"
        assert msg2.role == "assistant"
        assert msg2.model_usage is not None
        assert msg2.model_usage.model_name == "claude-3-opus"

    def test_claude_sender_role_mapping(self, claude_sample_file):
        """Test that sender roles are mapped correctly."""
        archive = parse_claude_export(claude_sample_file)
        conv = archive.conversations[0]

        human_msgs = [msg for msg in conv.messages if msg.role == "user"]
        assistant_msgs = [msg for msg in conv.messages if msg.role == "assistant"]

        assert len(human_msgs) == 2
        assert len(assistant_msgs) == 2

    def test_claude_model_metadata(self, claude_sample_file):
        """Test that model metadata is extracted."""
        archive = parse_claude_export(claude_sample_file)

        # Find assistant messages with model metadata
        assistant_msgs = [
            msg
            for conv in archive.conversations
            for msg in conv.messages
            if msg.role == "assistant" and msg.model_usage
        ]

        assert len(assistant_msgs) > 0
        assert any(msg.model_usage.model_name == "claude-3-opus" for msg in assistant_msgs)
        assert any(msg.model_usage.model_name == "claude-3-5-sonnet" for msg in assistant_msgs)


class TestParseArchive:
    """Tests for the generic parse_archive function."""

    def test_parse_archive_chatgpt(self, chatgpt_sample_file):
        """Test parsing with provider='chatgpt'."""
        archive = parse_archive(chatgpt_sample_file, "chatgpt")
        assert archive.source == "chatgpt"
        assert len(archive.conversations) > 0

    def test_parse_archive_claude(self, claude_sample_file):
        """Test parsing with provider='claude'."""
        archive = parse_archive(claude_sample_file, "claude")
        assert archive.source == "claude"
        assert len(archive.conversations) > 0

    def test_parse_archive_invalid_provider(self, chatgpt_sample_file):
        """Test that invalid provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown provider"):
            parse_archive(chatgpt_sample_file, "invalid")
