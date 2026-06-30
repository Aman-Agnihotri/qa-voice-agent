"""
QA Voice Agent - AI-powered Quality Assurance for Customer Support Calls

This package provides comprehensive analysis of customer service call transcripts,
including multi-model comparison, evaluation metrics, and actionable insights.
"""

__version__ = "1.0.0"
__author__ = "QA Voice Agent Team"

from .qa_agent import qa_agent, QAAgent, TranscriptAnalysis
from .model_comparison import model_comparator, ModelComparator
from .evaluation import qa_evaluator, QAEvaluator

__all__ = [
    "qa_agent",
    "QAAgent",
    "TranscriptAnalysis",
    "model_comparator",
    "ModelComparator",
    "qa_evaluator",
    "QAEvaluator",
]