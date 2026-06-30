#!/usr/bin/env python3
"""
Evaluation demonstration script.

This script shows how to evaluate QA models against ground truth data
and generate comprehensive evaluation reports.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation import qa_evaluator
from utils.config import settings, validate_api_keys


def prepare_ground_truth_data():
    """Prepare ground truth data for evaluation."""
    # Load sample data
    data_file = Path(__file__).parent.parent / "data" / "sample_transcripts.json"

    if not data_file.exists():
        print(f"❌ Sample data file not found: {data_file}")
        return None

    with open(data_file, 'r', encoding='utf-8') as f:
        sample_data = json.load(f)

    # Convert to evaluation format with ground truth
    evaluation_data = []
    for item in sample_data:
        evaluation_data.append({
            'id': str(item['id']),
            'transcript': item['transcript'],
            'expected_outcome': item['outcome'],
            'intent': item['intent'],
            'expected_resolution': 'Resolved' if 'successful' in item['outcome'].lower() else 'Unresolved',
            'notes': f"Original classification: {item['outcome']}"
        })

    return evaluation_data


async def single_model_evaluation():
    """Evaluate a single model against ground truth data."""
    print("=== Single Model Evaluation ===\n")

    # Get available models
    available_models = get_available_models()
    if not available_models:
        print("❌ No models available for evaluation.")
        return

    test_model = available_models[0]
    print(f"Evaluating model: {test_model}")

    # Prepare test data
    test_data = prepare_ground_truth_data()
    if not test_data:
        return

    print(f"Test cases: {len(test_data)}\n")

    try:
        # Run evaluation
        metrics = await qa_evaluator.evaluate_with_ground_truth(
            test_data=test_data,
            model=test_model,
            detailed=True
        )

        print("Evaluation Results:")
        print("==================")
        print(f"Total Evaluations: {metrics.total_evaluations}")
        print(f"Success Rate: {metrics.success_rate:.1%}")
        print(f"Classification Accuracy: {metrics.classification_accuracy:.1%}")
        print(f"Intent Accuracy: {metrics.intent_accuracy:.1%}")
        print(f"Average Response Time: {metrics.avg_response_time:.2f}s")
        print(f"Total Cost: ${metrics.total_cost:.4f}")
        print(f"Business Value Score: {metrics.business_value_score:.3f}")
        print(f"Confidence Score: {metrics.confidence_score:.2f}")
        print(f"Completeness Score: {metrics.completeness_score:.1%}")
        print(f"Recommendation Quality: {metrics.recommendation_quality:.1%}")

        # Generate detailed report
        report = qa_evaluator.generate_evaluation_report(
            metrics=metrics,
            model_name=test_model,
            test_description=f"Evaluation on {len(test_data)} Nike customer service transcripts"
        )

        print("\n" + "="*60)
        print(report)

        # Save report to file
        output_dir = Path(__file__).parent.parent / "output"
        output_dir.mkdir(exist_ok=True)

        report_file = output_dir / f"evaluation_report_{test_model.replace('/', '_')}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\n📄 Detailed report saved to: {report_file}")

    except Exception as e:
        print(f"❌ Error in evaluation: {e}")


async def cross_validation_demo():
    """Demonstrate cross-validation across multiple models."""
    print("\n\n=== Cross-Validation Demo ===\n")

    # Get available models (limit to 2 to avoid high costs)
    available_models = get_available_models()[:2]

    if len(available_models) < 2:
        print("❌ Need at least 2 models for cross-validation. Available:", available_models)
        return

    # Prepare test data
    test_data = prepare_ground_truth_data()
    if not test_data:
        return

    print(f"Running 3-fold cross-validation...")
    print(f"Models: {', '.join(available_models)}")
    print(f"Test cases: {len(test_data)}\n")

    try:
        # Run cross-validation
        cv_results = await qa_evaluator.cross_validate_models(
            test_data=test_data,
            models=available_models,
            k_folds=3
        )

        print("Cross-Validation Results:")
        print("========================")

        for model_name, metrics in cv_results.items():
            print(f"\n**{model_name}:**")
            print(f"  Classification Accuracy: {metrics.classification_accuracy:.1%}")
            print(f"  Success Rate: {metrics.success_rate:.1%}")
            print(f"  Average Response Time: {metrics.avg_response_time:.2f}s")
            print(f"  Business Value Score: {metrics.business_value_score:.3f}")
            print(f"  Confidence Score: {metrics.confidence_score:.2f}")

        # Find best performing model
        best_model = max(cv_results.keys(), key=lambda m: cv_results[m].business_value_score)
        best_score = cv_results[best_model].business_value_score

        print(f"\n🏆 **Best Performing Model:** {best_model}")
        print(f"   Business Value Score: {best_score:.3f}")

        # Export results to CSV
        output_dir = Path(__file__).parent.parent / "output"
        output_dir.mkdir(exist_ok=True)

        csv_file = output_dir / "cross_validation_results.csv"
        qa_evaluator.export_metrics_to_csv(cv_results, csv_file)

        print(f"\n📊 Results exported to: {csv_file}")

    except Exception as e:
        print(f"❌ Error in cross-validation: {e}")


async def create_ground_truth_template():
    """Create a template for manual ground truth annotation."""
    print("\n\n=== Ground Truth Template Creation ===\n")

    # Load sample data
    data_file = Path(__file__).parent.parent / "data" / "sample_transcripts.json"

    if not data_file.exists():
        print(f"❌ Sample data file not found: {data_file}")
        return

    with open(data_file, 'r', encoding='utf-8') as f:
        sample_data = json.load(f)

    # Create template
    template = qa_evaluator.create_ground_truth_template(sample_data)

    # Save template
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    template_file = output_dir / "ground_truth_template.json"
    with open(template_file, 'w', encoding='utf-8') as f:
        json.dump(template, f, indent=2, ensure_ascii=False)

    print(f"✅ Ground truth template created: {template_file}")
    print(f"   Template contains {len(template)} transcripts for annotation")
    print("\nTo use this template:")
    print("  1. Fill in the 'expected_outcome', 'intent', and other fields")
    print("  2. Save the completed file")
    print("  3. Use it with qa_evaluator.load_ground_truth_from_json()")


def accuracy_analysis_demo():
    """Demonstrate accuracy analysis techniques."""
    print("\n\n=== Accuracy Analysis Demo ===\n")

    # Show how classification matching works
    test_cases = [
        ("Automated - Successful", "Automated - Successful", True),
        ("Automated - Successful", "automated - successful", True),
        ("Escalated - Partially Successful", "Escalated - Unsuccessful", False),
        ("Escalated - Partially Successful", "escalated - partially successful", True),
    ]

    print("Classification Matching Examples:")
    print("================================")

    for predicted, expected, should_match in test_cases:
        # Use the evaluator's internal matching logic
        evaluator = qa_evaluator
        actual_match = evaluator._classifications_match(predicted, expected)

        status = "✅" if actual_match == should_match else "❌"
        print(f"{status} Predicted: '{predicted}' vs Expected: '{expected}' -> {actual_match}")

    # Show intent matching
    print(f"\nIntent Matching Examples:")
    print("========================")

    intent_cases = [
        ("Order Status", "Order Status", True),
        ("order tracking", "Order Status", True),
        ("Return Question", "Product Information", False),
        ("refund request", "Return Status", True),
    ]

    for predicted, expected, should_match in intent_cases:
        evaluator = qa_evaluator
        actual_match = evaluator._intents_match(predicted, expected)

        status = "✅" if actual_match == should_match else "❌"
        print(f"{status} Predicted: '{predicted}' vs Expected: '{expected}' -> {actual_match}")


def get_available_models():
    """Get list of available models based on API key configuration."""
    available_models = []

    # Check API key availability
    key_status = validate_api_keys(settings)

    if key_status.get('openai'):
        available_models.extend(['gpt-3.5-turbo'])  # Start with cheaper model
    if key_status.get('anthropic'):
        available_models.extend(['claude-3-haiku'])  # Cheaper Anthropic model
    if key_status.get('groq'):
        available_models.extend(['llama3-70b'])  # Fast and affordable
    if key_status.get('google'):
        available_models.append('gemini-pro')
    if key_status.get('mistral'):
        available_models.append('mistral-large')

    return available_models


async def main():
    """Run all evaluation examples."""
    print("QA Voice Agent - Evaluation Demo")
    print("===============================")

    # Check available models
    available_models = get_available_models()

    if not available_models:
        print("\n❌ No models available for evaluation.")
        print("\nTo use this demo, you need to:")
        print("  1. Set API keys as environment variables")
        print("     - OPENAI_API_KEY for OpenAI models")
        print("     - ANTHROPIC_API_KEY for Anthropic models")
        print("     - GROQ_API_KEY for Groq models")
        print("     - etc.")
        print("  2. Or install Ollama and pull local models")
        return

    print(f"\n✅ Found {len(available_models)} available models: {', '.join(available_models)}")

    print("\n⚠️  Note: This demo may incur API costs.")

    # Run demonstrations
    await single_model_evaluation()
    await cross_validation_demo()
    await create_ground_truth_template()
    accuracy_analysis_demo()

    print("\n🎉 Evaluation demo completed!")
    print("\nNext steps:")
    print("  1. Review evaluation reports in the output/ directory")
    print("  2. Use the ground truth template to create your own test dataset")
    print("  3. Run evaluations on your production data")
    print("  4. Use cross-validation to select the best model for your use case")


if __name__ == "__main__":
    asyncio.run(main())