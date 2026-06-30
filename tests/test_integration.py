#!/usr/bin/env python3
"""
Integration tests for the complete QA Voice Agent system.

These tests verify end-to-end functionality with real API calls
and comprehensive system behavior.
"""

import pytest
import asyncio
import sys
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load environment variables
load_dotenv()

from qa_agent import QAAgent
from model_comparison import ModelComparator
from evaluation import QAEvaluator


@pytest.mark.integration
@pytest.mark.api_required
class TestSystemIntegration:
    """Integration tests requiring API access."""

    @pytest.fixture(autouse=True)
    def check_api_key(self):
        """Ensure OpenAI API key is available for integration tests."""
        if not os.getenv('OPENAI_API_KEY'):
            pytest.skip("OpenAI API key required for integration tests")

    @pytest.mark.asyncio
    async def test_complete_qa_workflow(self):
        """Test the complete QA analysis workflow."""
        qa_agent = QAAgent(default_model="gpt-3.5-turbo")

        # Nike order status transcript
        transcript = """
        AI Agent: Hi, thanks for calling Nike. Am I speaking with Sarah?
        Customer: Yes.
        AI Agent: Hi Sarah, thanks for calling. I can access your account and help answer any questions. What can I help you with today?
        Customer: I'd like to know the status of my order.
        AI Agent: Is this about your order for the Air Max 270 sneakers in black?
        Customer: Yes, that's right.
        AI Agent: I've pulled up your order. It looks like your order is currently on its way to you and will be delivered by FedEx. You should receive it by tomorrow at 3 PM. Did you have any other questions?
        Customer: The package is being delivered, but they sent it to the wrong address.
        AI Agent: I'm sorry to hear your package was delivered to the wrong address. If you're unable to locate the package, please allow 5 business days after the delivery date, then contact our customer service team for further assistance. Is there anything else I can help you with?
        Customer: No, thank you.
        AI Agent: Great, thanks for calling Sarah. Have a great day.
        """

        analysis = await qa_agent.analyze_transcript(
            transcript=transcript,
            transcript_id="integration_test_001",
            model="gpt-3.5-turbo"
        )

        # Verify analysis completeness
        assert analysis.transcript_id == "integration_test_001"
        assert analysis.primary_classification is not None
        assert analysis.intent is not None
        assert analysis.confidence_score is not None
        assert analysis.analysis_duration is not None
        assert analysis.cost is not None

        # Verify classification makes sense
        assert "automated" in analysis.primary_classification.lower()
        assert "order" in analysis.intent.lower()
        assert analysis.confidence_score > 0

    @pytest.mark.asyncio
    async def test_sample_transcripts_analysis(self):
        """Test analysis of all sample transcripts from PDF."""
        qa_agent = QAAgent(default_model="gpt-3.5-turbo")

        # Load sample data
        data_file = Path(__file__).parent.parent / "data" / "sample_transcripts.json"
        assert data_file.exists(), "Sample data file not found"

        with open(data_file, 'r', encoding='utf-8') as f:
            sample_data = json.load(f)

        assert len(sample_data) == 5, "Expected 5 sample transcripts"

        results = []
        for sample in sample_data:
            analysis = await qa_agent.analyze_transcript(
                transcript=sample['transcript'],
                transcript_id=str(sample['id']),
                model="gpt-3.5-turbo"
            )

            # Verify basic analysis structure
            assert analysis.primary_classification is not None
            assert analysis.intent is not None

            # Check if classification type matches expectation (automated vs escalated)
            expected_type = "automated" if "automated" in sample['outcome'].lower() else "escalated"
            actual_type = "automated" if "automated" in analysis.primary_classification.lower() else "escalated"

            results.append({
                'id': sample['id'],
                'expected': sample['outcome'],
                'predicted': analysis.primary_classification,
                'type_match': expected_type == actual_type,
                'cost': analysis.cost
            })

        # Verify overall performance
        type_accuracy = sum(1 for r in results if r['type_match']) / len(results)
        total_cost = sum(r['cost'] for r in results if r['cost'])

        assert type_accuracy >= 0.6, f"Type accuracy too low: {type_accuracy:.1%}"
        assert total_cost < 0.05, f"Total cost too high: ${total_cost:.4f}"

    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test batch processing functionality."""
        qa_agent = QAAgent(default_model="gpt-3.5-turbo")

        batch_transcripts = [
            {
                "id": "batch_001",
                "transcript": "Customer: Order status? Agent: Your order shipped yesterday. Customer: Thank you."
            },
            {
                "id": "batch_002",
                "transcript": "Customer: I want to return this. Agent: I'll transfer you to returns. Agent: No agents available."
            }
        ]

        results = await qa_agent.batch_analyze(
            transcripts=batch_transcripts,
            model="gpt-3.5-turbo",
            max_concurrent=2
        )

        assert len(results) == 2
        assert all(hasattr(r, 'primary_classification') for r in results)
        assert all(r.transcript_id in ['batch_001', 'batch_002'] for r in results)

    @pytest.mark.asyncio
    async def test_export_functionality(self):
        """Test export capabilities."""
        qa_agent = QAAgent(default_model="gpt-3.5-turbo")

        analysis = await qa_agent.analyze_transcript(
            transcript="Customer: Hi. Agent: Hello, how can I help? Customer: Order status. Agent: Your order shipped.",
            transcript_id="export_test"
        )

        # Test JSON export
        json_export = qa_agent.export_analysis(analysis, format="json")
        parsed_json = json.loads(json_export)
        assert parsed_json["transcript_id"] == "export_test"

        # Test CSV export
        csv_export = qa_agent.export_analysis(analysis, format="csv")
        assert "export_test" in csv_export
        assert "," in csv_export  # CSV should have comma separators

    @pytest.mark.asyncio
    async def test_different_call_types(self):
        """Test classification of different call types."""
        qa_agent = QAAgent(default_model="gpt-3.5-turbo")

        test_cases = [
            {
                "name": "successful_automated",
                "transcript": "Customer: Order status? Agent: Your order is shipped, arriving tomorrow. Customer: Perfect, thanks!",
                "expected_type": "automated"
            },
            {
                "name": "escalated_unsuccessful",
                "transcript": "Customer: I need help with sizing. Agent: Let me transfer you. Agent: No agents available, email us.",
                "expected_type": "escalated"
            }
        ]

        for test_case in test_cases:
            analysis = await qa_agent.analyze_transcript(
                transcript=test_case["transcript"],
                transcript_id=f"type_test_{test_case['name']}"
            )

            classification = analysis.primary_classification.lower()
            expected = test_case["expected_type"]

            assert expected in classification, f"Expected {expected} in {classification}"

    @pytest.mark.asyncio
    async def test_model_comparison_basic(self):
        """Test basic model comparison functionality."""
        comparator = ModelComparator()

        # Simple transcript for comparison
        transcript = "Customer: Order status? Agent: Your order shipped. Customer: Thanks."

        # Test with available models (just OpenAI for integration test)
        available_models = ["gpt-3.5-turbo"]

        comparison_result = await comparator.compare_models_on_transcript(
            transcript=transcript,
            transcript_id="comparison_test",
            models=available_models,
            ground_truth="Automated - Successful"
        )

        assert comparison_result.transcript_id == "comparison_test"
        assert "gpt-3.5-turbo" in comparison_result.model_results
        assert comparison_result.model_results["gpt-3.5-turbo"].primary_classification is not None

    @pytest.mark.asyncio
    async def test_evaluation_system_basic(self):
        """Test basic evaluation system functionality."""
        evaluator = QAEvaluator()

        test_data = [
            {
                'id': 'eval_001',
                'transcript': 'Customer: Order status? Agent: Your order shipped yesterday.',
                'expected_outcome': 'Automated - Successful',
                'intent': 'Order Status'
            }
        ]

        metrics = await evaluator.evaluate_with_ground_truth(
            test_data=test_data,
            model="gpt-3.5-turbo",
            detailed=False
        )

        assert metrics.total_evaluations == 1
        assert metrics.success_rate > 0
        assert metrics.classification_accuracy >= 0
        assert metrics.business_value_score >= 0


@pytest.mark.slow
class TestPerformanceIntegration:
    """Performance and stress tests."""

    @pytest.mark.asyncio
    async def test_concurrent_processing(self):
        """Test system under concurrent load."""
        qa_agent = QAAgent(default_model="gpt-3.5-turbo")

        # Create multiple concurrent tasks
        transcripts = [
            f"Customer: Test {i}. Agent: Response {i}."
            for i in range(5)
        ]

        tasks = [
            qa_agent.analyze_transcript(transcript, f"concurrent_{i}")
            for i, transcript in enumerate(transcripts)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify all completed successfully
        successful = [r for r in results if not isinstance(r, Exception)]
        assert len(successful) >= 4, "Most concurrent requests should succeed"

    @pytest.mark.asyncio
    async def test_cost_efficiency(self):
        """Test that costs remain reasonable."""
        qa_agent = QAAgent(default_model="gpt-3.5-turbo")

        # Analyze a typical transcript
        analysis = await qa_agent.analyze_transcript(
            transcript="Customer: Order status? Agent: Your order shipped yesterday. Customer: Thank you.",
            transcript_id="cost_test"
        )

        # Verify cost is reasonable (should be under $0.01 for typical analysis)
        assert analysis.cost is not None
        assert analysis.cost < 0.01, f"Cost too high: ${analysis.cost:.4f}"

    @pytest.mark.asyncio
    async def test_response_time(self):
        """Test that response times are reasonable."""
        qa_agent = QAAgent(default_model="gpt-3.5-turbo")

        analysis = await qa_agent.analyze_transcript(
            transcript="Customer: Hi. Agent: Hello. Customer: Order status. Agent: Shipped.",
            transcript_id="timing_test"
        )

        # Verify response time is reasonable (should be under 30 seconds)
        assert analysis.analysis_duration is not None
        assert analysis.analysis_duration < 30, f"Response time too slow: {analysis.analysis_duration:.2f}s"


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_empty_transcript(self):
        """Test handling of empty transcript."""
        qa_agent = QAAgent(default_model="gpt-3.5-turbo")

        analysis = await qa_agent.analyze_transcript(
            transcript="",
            transcript_id="empty_test"
        )

        # Should not crash, should return some analysis
        assert analysis.transcript_id == "empty_test"
        assert analysis.primary_classification is not None

    @pytest.mark.asyncio
    async def test_malformed_transcript(self):
        """Test handling of malformed transcript."""
        qa_agent = QAAgent(default_model="gpt-3.5-turbo")

        # Transcript with unusual formatting
        analysis = await qa_agent.analyze_transcript(
            transcript="Customer Agent Customer Agent Random text!!!",
            transcript_id="malformed_test"
        )

        # Should handle gracefully
        assert analysis.transcript_id == "malformed_test"
        assert analysis.primary_classification is not None