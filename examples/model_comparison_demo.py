#!/usr/bin/env python3
"""
Model comparison demonstration script.

This script shows how to compare different AI models on QA analysis tasks,
evaluating their accuracy, speed, cost, and reliability.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model_comparison import model_comparator
from utils.config import settings, validate_api_keys


async def compare_models_on_single_transcript():
    """Compare multiple models on a single transcript."""
    print("=== Single Transcript Model Comparison ===\n")

    # Sample transcript
    transcript = """
    AI Agent: Hi, thanks for calling. Am I speaking with Jessica?
    Customer: Yes.
    AI Agent: Hi Jessica, thanks for calling. I can access your account and help answer any questions. What can I help you with today?
    Customer: I submitted a return and haven't received a refund.
    AI Agent: Are you checking on a return you've already submitted, or would you like to start a new return?
    Customer: Checking on the status.
    AI Agent: I found your order. Your order number is 8421357, and I see it was fulfilled. Would you like me to text you a link where you can view more information?
    Customer: No. I returned 2 items and haven't received a refund.
    AI Agent: I see you returned 2 items from your order, and I apologize you haven't received your refund yet. Please allow up to 14 business days for the refund to process after we receive the returned items.
    Customer: It has been 14 days.
    AI Agent: I apologize it's been 14 days and you still haven't received your refund. I can send you a text with a link to our returns portal where you can check your return status.
    Customer: Yes, please.
    AI Agent: Great, I've just sent that to you. Was there anything else you needed help with?
    Customer: Yeah, I'd like to speak with somebody about this.
    AI Agent: I'll connect you with a team member who can help you.
    """

    # Get available models
    available_models = get_available_models()
    if not available_models:
        print("❌ No models available. Please configure API keys.")
        return

    print(f"Comparing {len(available_models)} models on transcript analysis...")
    print(f"Models: {', '.join(available_models)}\n")

    try:
        # Run comparison
        comparison_result = await model_comparator.compare_models_on_transcript(
            transcript=transcript,
            transcript_id="comparison_demo",
            models=available_models[:3],  # Limit to first 3 models to avoid high costs
            ground_truth="Escalated - Partially Successful"
        )

        print("Results:")
        print("--------")

        for model_name, analysis in comparison_result.model_results.items():
            print(f"\n**{model_name}:**")
            print(f"  Classification: {analysis.primary_classification}")
            print(f"  Intent: {analysis.intent}")
            print(f"  Confidence: {analysis.confidence_score:.2f}" if analysis.confidence_score else "  No confidence score")
            print(f"  Duration: {analysis.analysis_duration:.2f}s" if analysis.analysis_duration else "  No duration")
            print(f"  Cost: ${analysis.cost:.4f}" if analysis.cost else "  No cost info")

        # Show performance metrics
        print(f"\n**Performance Comparison:**")
        for model_name, metrics in comparison_result.performance_metrics.items():
            if not metrics.get('error'):
                print(f"\n{model_name}:")
                print(f"  Response Time: {metrics.get('response_time', 0):.2f}s")
                print(f"  Tokens Used: {metrics.get('tokens_used', 0)}")
                print(f"  Cost: ${metrics.get('cost', 0):.4f}")
                print(f"  Confidence: {metrics.get('confidence_score', 0):.2f}")

        # Show consensus
        if comparison_result.consensus_analysis:
            print(f"\n**Consensus Analysis:**")
            consensus = comparison_result.consensus_analysis
            print(f"  Classification: {consensus.primary_classification}")
            print(f"  Intent: {consensus.intent}")
            print(f"  Confidence: {consensus.confidence_score:.2f}" if consensus.confidence_score else "  No confidence score")

    except Exception as e:
        print(f"❌ Error in model comparison: {e}")


async def comprehensive_model_evaluation():
    """Run comprehensive evaluation across multiple transcripts."""
    print("\n\n=== Comprehensive Model Evaluation ===\n")

    # Load sample data
    data_file = Path(__file__).parent.parent / "data" / "sample_transcripts.json"

    if not data_file.exists():
        print(f"❌ Sample data file not found: {data_file}")
        return

    with open(data_file, 'r', encoding='utf-8') as f:
        sample_data = json.load(f)

    # Get available models (limit to 2-3 for demo to avoid high costs)
    available_models = get_available_models()[:2]

    if not available_models:
        print("❌ No models available for evaluation.")
        return

    print(f"Running comprehensive evaluation...")
    print(f"Models: {', '.join(available_models)}")
    print(f"Test cases: {len(sample_data)}\n")

    try:
        # Run comprehensive evaluation
        model_reports = await model_comparator.comprehensive_model_evaluation(
            test_transcripts=sample_data,
            models=available_models,
            include_ground_truth=True
        )

        print("Evaluation Results:")
        print("==================")

        for model_name, report in model_reports.items():
            print(f"\n**{model_name}** ({report.provider}):")
            print(f"  Classification Accuracy: {report.classification_accuracy:.1%}")
            print(f"  Average Response Time: {report.avg_response_time:.2f}s")
            print(f"  Average Cost: ${report.avg_cost_per_analysis:.4f}")
            print(f"  Success Rate: {report.success_rate:.1%}")
            print(f"  Business Value Score: {report.business_value_score:.3f}")

            if report.strengths:
                print(f"  Strengths: {', '.join(report.strengths[:2])}")  # Show first 2

            if report.weaknesses:
                print(f"  Weaknesses: {', '.join(report.weaknesses[:2])}")  # Show first 2

        # Generate comparison report
        print(f"\n" + "="*50)
        markdown_report = model_comparator.generate_comparison_report(model_reports)
        print(markdown_report)

        # Save report to file
        output_dir = Path(__file__).parent.parent / "output"
        output_dir.mkdir(exist_ok=True)

        report_file = output_dir / "model_comparison_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(markdown_report)

        print(f"\n📄 Full report saved to: {report_file}")

    except Exception as e:
        print(f"❌ Error in comprehensive evaluation: {e}")


async def model_consistency_test():
    """Test model consistency by running the same transcript multiple times."""
    print("\n\n=== Model Consistency Test ===\n")

    # Sample transcript
    transcript = """
    AI Agent: Hi, thanks for calling. Am I speaking with David?
    Customer: Yes.
    AI Agent: Hi David, thanks for calling. I can access your account and help answer any questions. What can I help you with today?
    Customer: I need help with sizing for the low-top sneakers. I usually wear a 10 but heard they run big.
    AI Agent: I can help with general sizing information. Those sneakers typically run about half a size large, so many customers size down. However, for the best fit recommendation, I'd like to connect you with our product specialist.
    Customer: That would be great.
    AI Agent: Let me transfer you to our product team now.
    AI Agent: Thanks for calling. Our customer service team is experiencing higher than normal volume. We are available to answer all inquiries by email at this time. Please reach out to us directly at help@example.com. We appreciate your understanding. Have a great day.
    """

    # Get first available model
    available_models = get_available_models()
    if not available_models:
        print("❌ No models available for consistency testing.")
        return

    test_model = available_models[0]
    print(f"Testing consistency of {test_model} over 3 runs...\n")

    try:
        consistency_result = await model_comparator.benchmark_model_consistency(
            model_name=test_model,
            transcript=transcript,
            num_runs=3
        )

        print("Consistency Test Results:")
        print("------------------------")
        print(f"Model: {consistency_result['model']}")
        print(f"Runs Completed: {consistency_result['runs_completed']}")
        print(f"Runs Failed: {consistency_result['runs_failed']}")
        print(f"Classification Consistency: {consistency_result['classification_consistency']:.1%}")
        print(f"Confidence Variance: {consistency_result['confidence_variance']:.3f}")
        print(f"Response Time Variance: {consistency_result['response_time_variance']:.3f}")
        print(f"Overall Consistency Score: {consistency_result['consistency_score']:.3f}")

        if consistency_result['all_classifications']:
            print(f"\nAll Classifications:")
            for i, classification in enumerate(consistency_result['all_classifications']):
                print(f"  Run {i+1}: {classification}")

        if consistency_result['confidence_scores']:
            print(f"\nConfidence Scores: {[f'{score:.2f}' for score in consistency_result['confidence_scores']]}")

        if consistency_result['response_times']:
            print(f"Response Times: {[f'{time:.2f}s' for time in consistency_result['response_times']]}")

    except Exception as e:
        print(f"❌ Error in consistency test: {e}")


def get_available_models():
    """Get list of available models based on API key configuration."""
    available_models = []

    # Check API key availability
    key_status = validate_api_keys(settings)

    if key_status.get('openai'):
        available_models.extend(['gpt-3.5-turbo', 'gpt-4'])
    if key_status.get('anthropic'):
        available_models.extend(['claude-3-haiku', 'claude-3-sonnet'])
    if key_status.get('groq'):
        available_models.extend(['llama3-70b', 'mixtral-8x7b'])
    if key_status.get('google'):
        available_models.append('gemini-pro')
    if key_status.get('mistral'):
        available_models.append('mistral-large')

    # Always include local models (assuming Ollama might be available)
    if not available_models:
        print("⚠️  No API keys found. You can:")
        print("  1. Set up API keys for cloud models")
        print("  2. Install and run Ollama for local models")
        print("  3. Try with gpt-3.5-turbo if you have OpenAI API key")

    return available_models


async def main():
    """Run all model comparison examples."""
    print("QA Voice Agent - Model Comparison Demo")
    print("=====================================")

    # Check available models
    available_models = get_available_models()

    if not available_models:
        print("\n❌ No models available for comparison.")
        print("\nTo use this demo, you need to:")
        print("  1. Set API keys as environment variables (OPENAI_API_KEY, etc.)")
        print("  2. Or install Ollama and pull local models")
        return

    print(f"\n✅ Found {len(available_models)} available models: {', '.join(available_models[:5])}")

    if len(available_models) > 5:
        print(f"    (and {len(available_models) - 5} more...)")

    print("\n⚠️  Note: This demo may incur API costs. Models are limited to reduce expenses.")

    # Run demonstrations
    await compare_models_on_single_transcript()
    await comprehensive_model_evaluation()
    await model_consistency_test()

    print("\n🎉 Model comparison demo completed!")
    print("\nRecommendations:")
    print("  1. Review the generated report in output/model_comparison_report.md")
    print("  2. Choose the best model based on your requirements (accuracy vs speed vs cost)")
    print("  3. Run the evaluation demo for more detailed analysis")


if __name__ == "__main__":
    asyncio.run(main())