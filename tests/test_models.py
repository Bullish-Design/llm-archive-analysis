"""Tests for Pydantic models."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from llm_archive_analysis.models import (
    Archive,
    Conversation,
    CostEstimate,
    Message,
    ModelUsage,
)


class TestModelUsage:
    """Tests for ModelUsage model."""

    def test_model_usage_creation(self):
        """Test creating a ModelUsage instance."""
        usage = ModelUsage(
            model_name="gpt-4",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )
        assert usage.model_name == "gpt-4"
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150

    def test_model_usage_defaults(self):
        """Test ModelUsage with default values."""
        usage = ModelUsage(model_name="claude-3-opus")
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0


class TestMessage:
    """Tests for Message model."""

    def test_message_creation(self):
        """Test creating a Message instance."""
        msg = Message(
            id="msg-1",
            role="user",
            content="Hello world",
            created_at=datetime.now(),
        )
        assert msg.id == "msg-1"
        assert msg.role == "user"
        assert msg.content == "Hello world"
        assert msg.created_at is not None
        assert msg.model_usage is None

    def test_message_with_model_usage(self):
        """Test Message with model usage."""
        usage = ModelUsage(model_name="gpt-4", input_tokens=10, output_tokens=20)
        msg = Message(
            id="msg-1",
            role="assistant",
            content="Response",
            model_usage=usage,
        )
        assert msg.model_usage is not None
        assert msg.model_usage.model_name == "gpt-4"

    def test_message_role_validation(self):
        """Test that only valid roles are accepted."""
        valid_roles = ["system", "user", "assistant", "tool"]
        for role in valid_roles:
            msg = Message(id="msg-1", role=role, content="test")
            assert msg.role == role

        # Invalid role should raise ValidationError
        with pytest.raises(ValidationError):
            Message(id="msg-1", role="invalid", content="test")


class TestConversation:
    """Tests for Conversation model."""

    def test_conversation_creation(self):
        """Test creating a Conversation instance."""
        conv = Conversation(
            id="conv-1",
            title="Test Conversation",
            started_at=datetime.now(),
            messages=[],
        )
        assert conv.id == "conv-1"
        assert conv.title == "Test Conversation"
        assert conv.started_at is not None
        assert len(conv.messages) == 0

    def test_conversation_with_messages(self):
        """Test Conversation with messages."""
        messages = [
            Message(id="msg-1", role="user", content="Hello"),
            Message(id="msg-2", role="assistant", content="Hi there"),
        ]
        conv = Conversation(id="conv-1", messages=messages)
        assert len(conv.messages) == 2
        assert conv.messages[0].role == "user"
        assert conv.messages[1].role == "assistant"

    def test_conversation_defaults(self):
        """Test Conversation with default values."""
        conv = Conversation(id="conv-1")
        assert conv.title is None
        assert conv.started_at is None
        assert conv.messages == []


class TestArchive:
    """Tests for Archive model."""

    def test_archive_creation(self):
        """Test creating an Archive instance."""
        archive = Archive(
            source="chatgpt",
            ingested_at=datetime.now(),
            conversations=[],
        )
        assert archive.source == "chatgpt"
        assert archive.ingested_at is not None
        assert len(archive.conversations) == 0

    def test_archive_with_conversations(self):
        """Test Archive with conversations."""
        conv = Conversation(id="conv-1", title="Test")
        archive = Archive(
            source="claude",
            ingested_at=datetime.now(),
            conversations=[conv],
        )
        assert len(archive.conversations) == 1
        assert archive.conversations[0].id == "conv-1"

    def test_archive_source_validation(self):
        """Test that only valid sources are accepted."""
        for source in ["chatgpt", "claude"]:
            archive = Archive(source=source, ingested_at=datetime.now())
            assert archive.source == source

        # Invalid source should raise ValidationError
        with pytest.raises(ValidationError):
            Archive(source="invalid", ingested_at=datetime.now())


class TestCostEstimate:
    """Tests for CostEstimate model."""

    def test_cost_estimate_creation(self):
        """Test creating a CostEstimate instance."""
        estimate = CostEstimate(
            model_name="gpt-4",
            currency="USD",
            input_cost=0.03,
            output_cost=0.06,
            total_cost=0.09,
            pricing_source="api",
            input_tokens=1000,
            output_tokens=1000,
        )
        assert estimate.model_name == "gpt-4"
        assert estimate.currency == "USD"
        assert estimate.input_cost == 0.03
        assert estimate.output_cost == 0.06
        assert estimate.total_cost == 0.09
        assert estimate.pricing_source == "api"

    def test_cost_estimate_defaults(self):
        """Test CostEstimate with default values."""
        estimate = CostEstimate(model_name="claude-3-opus")
        assert estimate.currency == "USD"
        assert estimate.input_cost == 0.0
        assert estimate.output_cost == 0.0
        assert estimate.total_cost == 0.0
        assert estimate.pricing_source == ""
        assert estimate.input_tokens == 0
        assert estimate.output_tokens == 0

    def test_cost_estimate_serialization(self):
        """Test that CostEstimate can be serialized."""
        estimate = CostEstimate(
            model_name="gpt-4",
            input_cost=1.5,
            output_cost=3.0,
            total_cost=4.5,
        )
        data = estimate.model_dump()
        assert data["model_name"] == "gpt-4"
        assert data["input_cost"] == 1.5
        assert data["output_cost"] == 3.0
        assert data["total_cost"] == 4.5
