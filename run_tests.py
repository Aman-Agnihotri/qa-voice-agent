#!/usr/bin/env python3
"""
Test runner for QA Voice Agent system.

Provides convenient commands for running different types of tests.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and display results."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {cmd}")
    print('='*60)

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    return result.returncode == 0


def main():
    """Main test runner."""
    if len(sys.argv) < 2:
        print("QA Voice Agent - Test Runner")
        print("=" * 40)
        print("Usage: python run_tests.py <test_type>")
        print("\nAvailable test types:")
        print("  unit        - Fast unit tests (no API required)")
        print("  integration - Integration tests (requires API keys)")
        print("  all         - All tests")
        print("  fast        - Only fast tests (excludes slow tests)")
        print("  api         - Only tests that require API keys")
        print("  coverage    - Run tests with coverage report")
        print("\nExamples:")
        print("  python run_tests.py unit")
        print("  python run_tests.py integration")
        print("  python run_tests.py fast")
        return

    test_type = sys.argv[1].lower()

    # Change to project directory
    os.chdir(Path(__file__).parent)

    # Check if API key is available for integration tests
    has_api_key = bool(os.getenv('OPENAI_API_KEY'))

    commands = {
        'unit': ('pytest -m unit', 'Unit Tests (no API required)'),
        'integration': ('pytest -m integration', 'Integration Tests (API required)'),
        'all': ('pytest', 'All Tests'),
        'fast': ('pytest -m "not slow"', 'Fast Tests Only'),
        'api': ('pytest -m api_required', 'API-dependent Tests'),
        'coverage': ('pytest --cov=src --cov-report=html --cov-report=term', 'Tests with Coverage'),
        'quick': ('pytest tests/test_unit.py::TestQAAgentUnit::test_export_analysis_json -v', 'Quick Smoke Test')
    }

    if test_type not in commands:
        print(f"Unknown test type: {test_type}")
        print(f"Available types: {', '.join(commands.keys())}")
        return

    cmd, description = commands[test_type]

    # Warn about API requirements
    if test_type in ['integration', 'api', 'all'] and not has_api_key:
        print("[WARNING] Integration tests require OPENAI_API_KEY environment variable")
        print("   These tests will be skipped if the API key is not found.")
        print("   Set your API key in the .env file or as an environment variable.")
        print()

    # Run the tests
    success = run_command(cmd, description)

    if success:
        print(f"\n[SUCCESS] {description} completed successfully!")
    else:
        print(f"\n[FAILED] {description} failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()