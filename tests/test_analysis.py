"""Tests for cost and usage analysis."""

from datetime import datetime

import pytest

from llm_archive_analysis.analysis import analyze_archive, normalize_model_name
from llm_archive_analysis.models import Archive, Conversation, Message, ModelUsage


class TestNormalizeModelName:
    """Tests for model name normalization."""

    def test_gpt4_variants(self):
        """Test GPT-4 model name normalization."""
        assert normalize_model_name("gpt-4") == "gpt-4"
        assert normalize_model_name("GPT-4") == "gpt-4"
        assert normalize_model_name("gpt-4-0613") == "gpt-4"
        assert normalize_model_name("gpt-4-32k") == "gpt-4"

    def test_gpt4o_variants(self):
        """Test GPT-4o model name normalization."""
        assert normalize_model_name("gpt-4o") == "gpt-4o"
        assert normalize_model_name("GPT-4O") == "gpt-4o"
        assert normalize_model_name("gpt-4o-2024-05-13") == "gpt-4o"

    def test_gpt4_turbo_variants(self):
        """Test GPT-4 Turbo model name normalization."""
        assert normalize_model_name("gpt-4-turbo") == "gpt-4-turbo"
        assert normalize_model_name("GPT-4-TURBO") == "gpt-4-turbo"
        assert normalize_model_name("gpt-4-turbo-preview") == "gpt-4-turbo"

    def test_gpt35_variants(self):
        """Test GPT-3.5 model name normalization."""
        assert normalize_model_name("gpt-3.5-turbo") == "gpt-3.5-turbo"
        assert normalize_model_name("GPT-3.5-Turbo") == "gpt-3.5-turbo"
        assert normalize_model_name("gpt-3.5-turbo-16k") == "gpt-3.5-turbo"

    def test_claude_opus_variants(self):
        """Test Claude Opus model name normalization."""
        assert normalize_model_name("claude-3-opus") == "claude-3-opus"
        assert normalize_model_name("Claude-3-Opus") == "claude-3-opus"
        assert normalize_model_name("claude-3-opus-20240229") == "claude-3-opus"

    def test_claude_sonnet_variants(self):
        """Test Claude Sonnet model name normalization."""
        assert normalize_model_name("claude-3-sonnet") == "claude-3-sonnet"
        assert normalize_model_name("Claude-3-Sonnet") == "claude-3-sonnet"
        assert normalize_model_name("claude-3-sonnet-20240229") == "claude-3-sonnet"

    def test_claude_35_sonnet_variants(self):
        """Test Claude 3.5 Sonnet model name normalization."""
        assert normalize_model_name("claude-3-5-sonnet") == "claude-3-5-sonnet"
        assert normalize_model_name("claude-3.5-sonnet") == "claude-3-5-sonnet"
        assert normalize_model_name("Claude-3-5-Sonnet") == "claude-3-5-sonnet"

    def test_claude_haiku_variants(self):
        """Test Claude Haiku model name normalization."""
        assert normalize_model_name("claude-3-haiku") == "claude-3-haiku"
        assert normalize_model_name("Claude-3-Haiku") == "claude-3-haiku"

    def test_unknown_model(self):
        """Test that unknown model names are returned as-is."""
        unknown_model = "unknown-model-xyz"
        assert normalize_model_name(unknown_model) == unknown_model


class TestAnalyzeArchive:
    """Tests for archive analysis."""

    def test_analyze_empty_archive(self):
        """Test analyzing an archive with no conversations."""
        archive = Archive(
            source="chatgpt",
            ingested_at=datetime.now(),
            conversations=[],
        )
        estimates = analyze_archive(archive)
        assert estimates == []

    def test_analyze_archive_no_usage(self):
        """Test analyzing an archive with no model usage data."""
        messages = [
            Message(id="msg-1", role="user", content="Hello"),
            Message(id="msg-2", role="assistant", content="Hi"),
        ]
        conv = Conversation(id="conv-1", messages=messages)
        archive = Archive(
            source="chatgpt",
            ingested_at=datetime.now(),
            conversations=[conv],
        )
        estimates = analyze_archive(archive)
        assert estimates == []

    def test_analyze_archive_with_usage(self):
        """Test analyzing an archive with model usage data."""
        usage1 = ModelUsage(
            model_name="gpt-4",
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1500,
        )
        usage2 = ModelUsage(
            model_name="gpt-4",
            input_tokens=2000,
            output_tokens=1000,
            total_tokens=3000,
        )

        messages = [
            Message(
                id="msg-1",
                role="assistant",
                content="Response 1",
                model_usage=usage1,
            ),
            Message(
                id="msg-2",
                role="assistant",
                content="Response 2",
                model_usage=usage2,
            ),
        ]
        conv = Conversation(id="conv-1", messages=messages)
        archive = Archive(
            source="chatgpt",
            ingested_at=datetime.now(),
            conversations=[conv],
        )

        estimates = analyze_archive(archive)

        assert len(estimates) == 1
        estimate = estimates[0]
        assert estimate.model_name == "gpt-4"
        assert estimate.input_tokens == 3000
        assert estimate.output_tokens == 1500
        assert estimate.currency == "USD"
        assert estimate.total_cost > 0

    def test_analyze_archive_multiple_models(self):
        """Test analyzing an archive with multiple models."""
        messages = [
            Message(
                id="msg-1",
                role="assistant",
                content="Response 1",
                model_usage=ModelUsage(
                    model_name="gpt-4",
                    input_tokens=1000,
                    output_tokens=500,
                ),
            ),
            Message(
                id="msg-2",
                role="assistant",
                content="Response 2",
                model_usage=ModelUsage(
                    model_name="gpt-3.5-turbo",
                    input_tokens=2000,
                    output_tokens=1000,
                ),
            ),
        ]
        conv = Conversation(id="conv-1", messages=messages)
        archive = Archive(
            source="chatgpt",
            ingested_at=datetime.now(),
            conversations=[conv],
        )

        estimates = analyze_archive(archive)

        assert len(estimates) == 2
        model_names = {est.model_name for est in estimates}
        assert "gpt-4" in model_names
        assert "gpt-3.5-turbo" in model_names

    def test_cost_calculation(self):
        """Test that cost is calculated correctly."""
        # GPT-4 pricing: $0.03/1k input, $0.06/1k output
        usage = ModelUsage(
            model_name="gpt-4",
            input_tokens=1000,
            output_tokens=1000,
            total_tokens=2000,
        )
        messages = [
            Message(
                id="msg-1",
                role="assistant",
                content="Response",
                model_usage=usage,
            ),
        ]
        conv = Conversation(id="conv-1", messages=messages)
        archive = Archive(
            source="chatgpt",
            ingested_at=datetime.now(),
            conversations=[conv],
        )

        estimates = analyze_archive(archive)

        assert len(estimates) == 1
        estimate = estimates[0]
        # 1000 tokens * $0.03/1000 = $0.03 input
        # 1000 tokens * $0.06/1000 = $0.06 output
        # Total = $0.09
        assert estimate.input_cost == pytest.approx(0.03, rel=1e-6)
        assert estimate.output_cost == pytest.approx(0.06, rel=1e-6)
        assert estimate.total_cost == pytest.approx(0.09, rel=1e-6)

    def test_unknown_model_zero_cost(self):
        """Test that unknown models have zero cost."""
        usage = ModelUsage(
            model_name="unknown-model",
            input_tokens=1000,
            output_tokens=1000,
        )
        messages = [
            Message(
                id="msg-1",
                role="assistant",
                content="Response",
                model_usage=usage,
            ),
        ]
        conv = Conversation(id="conv-1", messages=messages)
        archive = Archive(
            source="chatgpt",
            ingested_at=datetime.now(),
            conversations=[conv],
        )

        estimates = analyze_archive(archive)

        assert len(estimates) == 1
        estimate = estimates[0]
        assert estimate.input_cost == 0.0
        assert estimate.output_cost == 0.0
        assert estimate.total_cost == 0.0
