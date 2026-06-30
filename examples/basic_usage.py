#!/usr/bin/env python3
"""
Basic usage example for QA Voice Agent.

This script demonstrates how to analyze customer service call transcripts
using the QA Voice Agent system.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qa_agent import qa_agent
from utils.config import settings


async def analyze_single_transcript():
    """Example: Analyze a single transcript."""
    print("=== Single Transcript Analysis ===\n")

    # Sample transcript (from our test data)
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

    try:
        # Analyze the transcript
        analysis = await qa_agent.analyze_transcript(
            transcript=transcript,
            transcript_id="example_001",
            model="gpt-3.5-turbo"  # Using a more affordable model for demo
        )

        # Display results
        print(f"Transcript ID: {analysis.transcript_id}")
        print(f"Classification: {analysis.primary_classification}")
        print(f"Intent: {analysis.intent}")
        print(f"Resolution Status: {analysis.resolution_status}")
        print(f"Confidence Score: {analysis.confidence_score:.2f}" if analysis.confidence_score else "No confidence score")

        if analysis.success_indicators:
            print(f"\nSuccess Indicators:")
            for indicator in analysis.success_indicators:
                print(f"  • {indicator}")

        if analysis.failure_indicators:
            print(f"\nFailure Indicators:")
            for indicator in analysis.failure_indicators:
                print(f"  • {indicator}")

        if analysis.recommendations:
            print(f"\nRecommendations:")
            for rec in analysis.recommendations:
                print(f"  • {rec}")

        print(f"\nAnalysis Duration: {analysis.analysis_duration:.2f}s" if analysis.analysis_duration else "")
        print(f"Tokens Used: {analysis.tokens_used}" if analysis.tokens_used else "")
        print(f"Cost: ${analysis.cost:.4f}" if analysis.cost else "")

    except Exception as e:
        print(f"Error analyzing transcript: {e}")


async def analyze_sample_data():
    """Example: Analyze all sample transcripts from the data file."""
    print("\n\n=== Sample Data Analysis ===\n")

    # Load sample data
    data_file = Path(__file__).parent.parent / "data" / "sample_transcripts.json"

    if not data_file.exists():
        print(f"Sample data file not found: {data_file}")
        return

    with open(data_file, 'r', encoding='utf-8') as f:
        sample_data = json.load(f)

    print(f"Analyzing {len(sample_data)} sample transcripts...\n")

    # Analyze each transcript
    for i, transcript_data in enumerate(sample_data):
        print(f"--- Analysis {i+1}: {transcript_data['intent']} ---")

        try:
            analysis = await qa_agent.analyze_transcript(
                transcript=transcript_data['transcript'],
                transcript_id=str(transcript_data['id']),
                model="gpt-3.5-turbo"
            )

            print(f"Expected: {transcript_data['outcome']}")
            print(f"Predicted: {analysis.primary_classification}")
            print(f"Intent: {analysis.intent}")
            print(f"Confidence: {analysis.confidence_score:.2f}" if analysis.confidence_score else "No confidence score")

            # Check if prediction matches expectation
            if analysis.primary_classification and transcript_data['outcome'].lower() in analysis.primary_classification.lower():
                print("✅ Classification matches expected outcome")
            else:
                print("❌ Classification differs from expected outcome")

            print()

        except Exception as e:
            print(f"❌ Error analyzing transcript {transcript_data['id']}: {e}")
            print()


async def batch_analysis_example():
    """Example: Batch analysis of multiple transcripts."""
    print("\n\n=== Batch Analysis Example ===\n")

    # Load sample data
    data_file = Path(__file__).parent.parent / "data" / "sample_transcripts.json"

    if not data_file.exists():
        print(f"Sample data file not found: {data_file}")
        return

    with open(data_file, 'r', encoding='utf-8') as f:
        sample_data = json.load(f)

    # Prepare data for batch analysis
    transcripts = [
        {
            'id': str(item['id']),
            'transcript': item['transcript']
        }
        for item in sample_data
    ]

    print(f"Running batch analysis on {len(transcripts)} transcripts...")

    try:
        # Run batch analysis
        results = await qa_agent.batch_analyze(
            transcripts=transcripts,
            model="gpt-3.5-turbo",
            max_concurrent=3  # Limit concurrent requests
        )

        print(f"Completed batch analysis. Results:")
        print(f"Total transcripts: {len(transcripts)}")
        print(f"Successful analyses: {len([r for r in results if not isinstance(r, Exception)])}")
        print(f"Failed analyses: {len([r for r in results if isinstance(r, Exception)])}")

        # Show summary of results
        classifications = {}
        for result in results:
            if not isinstance(result, Exception) and result.primary_classification:
                classification = result.primary_classification
                classifications[classification] = classifications.get(classification, 0) + 1

        print(f"\nClassification Summary:")
        for classification, count in classifications.items():
            print(f"  {classification}: {count}")

    except Exception as e:
        print(f"Error in batch analysis: {e}")


async def export_analysis_example():
    """Example: Export analysis results in different formats."""
    print("\n\n=== Export Analysis Example ===\n")

    # Analyze a sample transcript
    transcript = """
    AI Agent: Hi, thanks for calling Nike. Am I speaking with Mike?
    Customer: Yes.
    AI Agent: Hi Mike, welcome back. I can access your account and help answer any questions. What do you need help with today?
    Customer: Return order.
    AI Agent: Are you checking on the status of a return, or would you like to start a new return?
    Customer: Checking on status of return.
    """

    try:
        analysis = await qa_agent.analyze_transcript(
            transcript=transcript,
            transcript_id="export_example",
            model="gpt-3.5-turbo"
        )

        # Export as JSON
        json_export = qa_agent.export_analysis(analysis, format="json")
        print("JSON Export:")
        print(json_export[:500] + "..." if len(json_export) > 500 else json_export)

        # Export as CSV
        csv_export = qa_agent.export_analysis(analysis, format="csv")
        print(f"\nCSV Export (first 200 chars):")
        print(csv_export[:200] + "..." if len(csv_export) > 200 else csv_export)

        # Save to files
        output_dir = Path(__file__).parent.parent / "output"
        output_dir.mkdir(exist_ok=True)

        with open(output_dir / "sample_analysis.json", 'w', encoding='utf-8') as f:
            f.write(json_export)

        with open(output_dir / "sample_analysis.csv", 'w', encoding='utf-8') as f:
            f.write(csv_export)

        print(f"\nExported analysis to:")
        print(f"  - {output_dir / 'sample_analysis.json'}")
        print(f"  - {output_dir / 'sample_analysis.csv'}")

    except Exception as e:
        print(f"Error in export example: {e}")


async def main():
    """Run all examples."""
    print("QA Voice Agent - Basic Usage Examples")
    print("=====================================")

    # Check if we have any API keys configured
    api_keys_configured = any([
        settings.openai_api_key,
        settings.anthropic_api_key,
        settings.groq_api_key,
        settings.google_api_key,
        settings.mistral_api_key
    ])

    if not api_keys_configured:
        print("\n⚠️  Warning: No API keys found in environment variables.")
        print("To use cloud models, set one or more of the following environment variables:")
        print("  - OPENAI_API_KEY")
        print("  - ANTHROPIC_API_KEY")
        print("  - GROQ_API_KEY")
        print("  - GOOGLE_API_KEY")
        print("  - MISTRAL_API_KEY")
        print("\nAlternatively, ensure Ollama is running locally for local model support.")
        print("Continuing with examples (may fail without proper API keys)...\n")

    # Run examples
    await analyze_single_transcript()
    await analyze_sample_data()
    await batch_analysis_example()
    await export_analysis_example()

    print("\n🎉 All examples completed!")
    print("\nNext steps:")
    print("  1. Set up API keys for cloud models")
    print("  2. Try model comparison: python examples/model_comparison_demo.py")
    print("  3. Run evaluation: python examples/evaluation_demo.py")


if __name__ == "__main__":
    asyncio.run(main())