"""
Pytest configuration and shared fixtures.
"""

import pytest
import asyncio
import sys
from pathlib import Path

# Add src to Python path for tests
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_transcript():
    """Sample transcript for testing."""
    return """
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


@pytest.fixture
def sample_transcripts():
    """Multiple sample transcripts for testing."""
    return [
        {
            "id": "test_001",
            "transcript": """
            AI Agent: Hi, thanks for calling Nike. Am I speaking with Sarah?
            Customer: Yes.
            AI Agent: I'd like to know the status of my order.
            AI Agent: Your order is shipped and will arrive tomorrow.
            Customer: Thank you.
            """,
            "outcome": "Automated - Successful",
            "intent": "Order Status"
        },
        {
            "id": "test_002",
            "transcript": """
            AI Agent: Hi, thanks for calling Nike. Am I speaking with Mike?
            Customer: Yes.
            Customer: I want to return something.
            AI Agent: I'll connect you with our returns team.
            AI Agent: Unfortunately no agents are available.
            """,
            "outcome": "Escalated - Unsuccessful",
            "intent": "Return Request"
        }
    ]


@pytest.fixture
def mock_model_response():
    """Mock model response for testing."""
    from utils.models import ModelResponse

    return ModelResponse(
        content='{"primary_classification": "Automated - Successful", "intent": "Order Status", "resolution_status": "Customer inquiry resolved", "confidence_score": 0.9, "success_indicators": ["Quick resolution"], "failure_indicators": [], "recommendations": ["Monitor for similar cases"], "customer_satisfaction_signals": "Positive"}',
        provider="test",
        model="test-model",
        tokens_used=150,
        response_time=2.0,
        cost=0.001
    )


@pytest.fixture
def sample_transcript_analysis():
    """Sample transcript analysis for testing."""
    from qa_agent import TranscriptAnalysis

    return TranscriptAnalysis(
        transcript_id="test_123",
        primary_classification="Automated - Successful",
        intent="Order Status",
        resolution_status="Customer inquiry resolved successfully",
        confidence_score=0.9,
        success_indicators=["Quick response", "Accurate information"],
        failure_indicators=[],
        recommendations=["Continue monitoring similar cases"],
        customer_satisfaction_signals="Customer expressed satisfaction",
        analysis_duration=2.5,
        tokens_used=150,
        cost=0.001
    )


@pytest.fixture
def evaluation_test_data():
    """Test data for evaluation testing."""
    return [
        {
            'id': 'eval_001',
            'transcript': 'Customer wants order status, agent provides tracking info',
            'expected_outcome': 'Automated - Successful',
            'intent': 'Order Status',
            'expected_resolution': 'Resolved'
        },
        {
            'id': 'eval_002',
            'transcript': 'Customer has complex return issue, escalated to human',
            'expected_outcome': 'Escalated - Partially Successful',
            'intent': 'Return Request',
            'expected_resolution': 'Partially Resolved'
        }
    ]


@pytest.fixture
def mock_comparison_result():
    """Mock comparison result for testing."""
    from model_comparison import ComparisonResult
    from qa_agent import TranscriptAnalysis

    return ComparisonResult(
        transcript_id="comparison_test",
        ground_truth="Automated - Successful",
        model_results={
            "model1": TranscriptAnalysis(
                transcript_id="comparison_test",
                primary_classification="Automated - Successful",
                intent="Order Status",
                confidence_score=0.9
            ),
            "model2": TranscriptAnalysis(
                transcript_id="comparison_test",
                primary_classification="Automated - Partially Successful",
                intent="Order Status",
                confidence_score=0.8
            )
        },
        performance_metrics={
            "model1": {
                "response_time": 2.0,
                "tokens_used": 100,
                "cost": 0.001,
                "confidence_score": 0.9,
                "success": True,
                "classification_accuracy": 1.0
            },
            "model2": {
                "response_time": 1.5,
                "tokens_used": 80,
                "cost": 0.0008,
                "confidence_score": 0.8,
                "success": True,
                "classification_accuracy": 0.0
            }
        }
    )


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Custom markers for different test categories
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "api_required: mark test as requiring API keys"
    )


# Pytest collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names and classes."""
    for item in items:
        # Add markers based on test class names
        if "Integration" in str(item.cls):
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.api_required)
        elif "Unit" in str(item.cls):
            item.add_marker(pytest.mark.unit)

        # Add slow marker to performance tests
        if any(keyword in item.name.lower() for keyword in ["performance", "stress", "concurrent", "benchmark"]):
            item.add_marker(pytest.mark.slow)

        # Add api_required marker to tests that need real API calls
        if any(keyword in item.name.lower() for keyword in ["integration", "api", "real_model", "live"]):
            item.add_marker(pytest.mark.api_required)