"""
Model comparison framework for evaluating different AI models on QA tasks.

This module provides comprehensive comparison capabilities across multiple
models and providers, focusing on accuracy, speed, cost, and reliability.
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import statistics

try:
    from .qa_agent import QAAgent, TranscriptAnalysis
    from .utils import model_manager, ModelResponse, ModelMetrics, settings
    from .prompts import COMPARATIVE_ANALYSIS_PROMPT
except ImportError:
    from qa_agent import QAAgent, TranscriptAnalysis
    from utils import model_manager, ModelResponse, ModelMetrics, settings
    from prompts import COMPARATIVE_ANALYSIS_PROMPT


@dataclass
class ComparisonResult:
    """Results from comparing multiple models on the same transcript."""
    transcript_id: str
    ground_truth: Optional[str] = None
    model_results: Dict[str, TranscriptAnalysis] = None
    performance_metrics: Dict[str, Dict[str, float]] = None
    consensus_analysis: Optional[TranscriptAnalysis] = None
    comparison_timestamp: str = None

    def __post_init__(self):
        if self.comparison_timestamp is None:
            self.comparison_timestamp = datetime.now().isoformat()
        if self.model_results is None:
            self.model_results = {}
        if self.performance_metrics is None:
            self.performance_metrics = {}


@dataclass
class ModelPerformanceReport:
    """Comprehensive performance report for a model."""
    model_name: str
    provider: str

    # Accuracy metrics
    classification_accuracy: float = 0.0
    intent_accuracy: float = 0.0
    overall_accuracy: float = 0.0

    # Performance metrics
    avg_response_time: float = 0.0
    avg_tokens_used: float = 0.0
    avg_cost_per_analysis: float = 0.0

    # Reliability metrics
    success_rate: float = 0.0
    error_count: int = 0
    consistency_score: float = 0.0

    # Quality metrics
    avg_confidence_score: float = 0.0
    recommendation_quality: float = 0.0
    detail_completeness: float = 0.0

    # Business metrics
    cost_effectiveness: float = 0.0
    business_value_score: float = 0.0

    # Sample outputs
    sample_analyses: List[TranscriptAnalysis] = None
    strengths: List[str] = None
    weaknesses: List[str] = None

    def __post_init__(self):
        if self.sample_analyses is None:
            self.sample_analyses = []
        if self.strengths is None:
            self.strengths = []
        if self.weaknesses is None:
            self.weaknesses = []


class ModelComparator:
    """
    Comprehensive model comparison framework.

    Evaluates multiple models across various metrics including accuracy,
    speed, cost, reliability, and business value.
    """

    def __init__(self):
        self.qa_agent = QAAgent()
        self.model_manager = model_manager

        # Standard test models for comparison
        self.default_models = [
            "gpt-4",
            "gpt-3.5-turbo",
            "claude-3-sonnet",
            "claude-3-haiku",
            "llama3-70b",  # Groq
            "gemini-pro",
            "mistral-large",
        ]

    async def compare_models_on_transcript(
        self,
        transcript: str,
        transcript_id: str,
        models: Optional[List[str]] = None,
        ground_truth: Optional[str] = None
    ) -> ComparisonResult:
        """
        Compare multiple models on a single transcript.

        Args:
            transcript: The call transcript to analyze
            transcript_id: Identifier for the transcript
            models: List of model names to compare (uses defaults if None)
            ground_truth: Known correct classification for accuracy measurement

        Returns:
            ComparisonResult with analysis from all models
        """
        models_to_test = models or self._get_available_models()

        # Run analysis with all models in parallel
        tasks = [
            self.qa_agent.analyze_transcript(transcript, transcript_id, model)
            for model in models_to_test
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        model_results = {}
        performance_metrics = {}

        for model, result in zip(models_to_test, results):
            if isinstance(result, Exception):
                # Handle failed analysis
                performance_metrics[model] = {
                    "error": True,
                    "error_message": str(result),
                    "success": False
                }
            else:
                model_results[model] = result
                performance_metrics[model] = self._calculate_performance_metrics(result, ground_truth)

        # Generate consensus analysis
        consensus = self._generate_consensus_analysis(model_results)

        return ComparisonResult(
            transcript_id=transcript_id,
            ground_truth=ground_truth,
            model_results=model_results,
            performance_metrics=performance_metrics,
            consensus_analysis=consensus
        )

    async def comprehensive_model_evaluation(
        self,
        test_transcripts: List[Dict[str, Any]],
        models: Optional[List[str]] = None,
        include_ground_truth: bool = True
    ) -> Dict[str, ModelPerformanceReport]:
        """
        Conduct comprehensive evaluation of models across multiple transcripts.

        Args:
            test_transcripts: List of transcript data with 'transcript', 'id', and optionally 'expected_outcome'
            models: Models to evaluate
            include_ground_truth: Whether to include ground truth comparisons

        Returns:
            Dictionary mapping model names to performance reports
        """
        models_to_test = models or self._get_available_models()

        print(f"Starting comprehensive evaluation of {len(models_to_test)} models on {len(test_transcripts)} transcripts...")

        # Run comparisons for all transcripts
        comparison_tasks = []
        for transcript_data in test_transcripts:
            ground_truth = transcript_data.get('expected_outcome') if include_ground_truth else None

            task = self.compare_models_on_transcript(
                transcript=transcript_data['transcript'],
                transcript_id=transcript_data['id'],
                models=models_to_test,
                ground_truth=ground_truth
            )
            comparison_tasks.append(task)

        comparison_results = await asyncio.gather(*comparison_tasks, return_exceptions=True)

        # Process results and generate reports
        model_reports = {}

        for model in models_to_test:
            # Collect all results for this model
            model_analyses = []
            model_performance_data = []

            for result in comparison_results:
                if isinstance(result, Exception):
                    continue

                if model in result.model_results:
                    model_analyses.append(result.model_results[model])

                if model in result.performance_metrics:
                    model_performance_data.append(result.performance_metrics[model])

            # Generate comprehensive report
            report = self._generate_performance_report(model, model_analyses, model_performance_data)
            model_reports[model] = report

        return model_reports

    def _get_available_models(self) -> List[str]:
        """Get list of available models based on API key availability."""
        from .utils.config import validate_api_keys

        available_models = []
        key_status = validate_api_keys(settings)

        # Add models based on available API keys
        if key_status.get('openai'):
            available_models.extend(['gpt-4', 'gpt-3.5-turbo'])
        if key_status.get('anthropic'):
            available_models.extend(['claude-3-sonnet', 'claude-3-haiku'])
        if key_status.get('groq'):
            available_models.extend(['llama3-70b', 'mixtral-8x7b'])
        if key_status.get('google'):
            available_models.extend(['gemini-pro'])
        if key_status.get('mistral'):
            available_models.extend(['mistral-large'])
        if key_status.get('ollama'):
            available_models.extend(['llama3', 'qwen2'])

        # Fallback to a subset if no API keys available
        if not available_models:
            print("Warning: No API keys found. Consider setting up API keys or using local models.")
            available_models = ['llama3']  # Assume Ollama is available locally

        return available_models

    def _calculate_performance_metrics(
        self,
        analysis: TranscriptAnalysis,
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """Calculate performance metrics for a single analysis."""
        metrics = {
            "response_time": analysis.analysis_duration or 0,
            "tokens_used": analysis.tokens_used or 0,
            "cost": analysis.cost or 0,
            "confidence_score": analysis.confidence_score or 0,
            "success": 1.0,  # Successfully completed analysis
        }

        # Accuracy metrics if ground truth available
        if ground_truth and analysis.primary_classification:
            classification_match = 1.0 if ground_truth.lower() in analysis.primary_classification.lower() else 0.0
            metrics["classification_accuracy"] = classification_match

        # Quality metrics
        metrics["detail_completeness"] = self._calculate_completeness_score(analysis)
        metrics["recommendation_quality"] = self._calculate_recommendation_quality(analysis)

        return metrics

    def _calculate_completeness_score(self, analysis: TranscriptAnalysis) -> float:
        """Calculate how complete and detailed the analysis is."""
        score = 0.0
        max_score = 7.0

        # Check for presence of key components
        if analysis.primary_classification:
            score += 1.0
        if analysis.intent:
            score += 1.0
        if analysis.resolution_status:
            score += 1.0
        if analysis.success_indicators and len(analysis.success_indicators) > 0:
            score += 1.0
        if analysis.failure_indicators and len(analysis.failure_indicators) > 0:
            score += 1.0
        if analysis.customer_satisfaction_signals:
            score += 1.0
        if analysis.recommendations and len(analysis.recommendations) > 0:
            score += 1.0

        return score / max_score

    def _calculate_recommendation_quality(self, analysis: TranscriptAnalysis) -> float:
        """Calculate quality of recommendations provided."""
        if not analysis.recommendations:
            return 0.0

        # Simple scoring based on number and length of recommendations
        num_recommendations = len(analysis.recommendations)
        avg_length = sum(len(rec) for rec in analysis.recommendations) / num_recommendations

        # Score based on number (1-5 is good) and average length (50-200 chars is good)
        num_score = min(num_recommendations / 5.0, 1.0)
        length_score = min(max(avg_length - 20, 0) / 180.0, 1.0)

        return (num_score + length_score) / 2.0

    def _generate_consensus_analysis(
        self,
        model_results: Dict[str, TranscriptAnalysis]
    ) -> Optional[TranscriptAnalysis]:
        """Generate consensus analysis from multiple model results."""
        if not model_results:
            return None

        analyses = list(model_results.values())

        # Find most common classification
        classifications = [a.primary_classification for a in analyses if a.primary_classification]
        most_common_classification = max(set(classifications), key=classifications.count) if classifications else None

        # Find most common intent
        intents = [a.intent for a in analyses if a.intent]
        most_common_intent = max(set(intents), key=intents.count) if intents else None

        # Average confidence score
        confidence_scores = [a.confidence_score for a in analyses if a.confidence_score is not None]
        avg_confidence = statistics.mean(confidence_scores) if confidence_scores else None

        # Combine recommendations
        all_recommendations = []
        for analysis in analyses:
            if analysis.recommendations:
                all_recommendations.extend(analysis.recommendations)

        # Remove duplicates while preserving order
        unique_recommendations = list(dict.fromkeys(all_recommendations))

        return TranscriptAnalysis(
            primary_classification=most_common_classification,
            intent=most_common_intent,
            confidence_score=avg_confidence,
            recommendations=unique_recommendations[:5],  # Top 5 unique recommendations
            resolution_status="Consensus analysis from multiple models"
        )

    def _generate_performance_report(
        self,
        model_name: str,
        analyses: List[TranscriptAnalysis],
        performance_data: List[Dict[str, float]]
    ) -> ModelPerformanceReport:
        """Generate comprehensive performance report for a model."""

        if not analyses or not performance_data:
            return ModelPerformanceReport(
                model_name=model_name,
                provider="unknown",
                error_count=len(performance_data)
            )

        # Calculate aggregate metrics
        successful_analyses = [p for p in performance_data if p.get("success", False)]

        if not successful_analyses:
            return ModelPerformanceReport(
                model_name=model_name,
                provider="unknown",
                error_count=len(performance_data),
                success_rate=0.0
            )

        # Performance metrics
        avg_response_time = statistics.mean([p["response_time"] for p in successful_analyses])
        avg_tokens_used = statistics.mean([p["tokens_used"] for p in successful_analyses if p["tokens_used"] > 0])
        avg_cost = statistics.mean([p["cost"] for p in successful_analyses if p["cost"] > 0])

        # Quality metrics
        avg_confidence = statistics.mean([p["confidence_score"] for p in successful_analyses if p["confidence_score"] > 0])
        avg_completeness = statistics.mean([p["detail_completeness"] for p in successful_analyses])
        avg_recommendation_quality = statistics.mean([p["recommendation_quality"] for p in successful_analyses])

        # Accuracy metrics (if available)
        accuracy_scores = [p["classification_accuracy"] for p in successful_analyses if "classification_accuracy" in p]
        classification_accuracy = statistics.mean(accuracy_scores) if accuracy_scores else 0.0

        # Success rate
        success_rate = len(successful_analyses) / len(performance_data)

        # Cost effectiveness (quality per dollar)
        cost_effectiveness = (avg_confidence * avg_completeness) / (avg_cost + 0.001) if avg_cost > 0 else 0

        # Business value score (combination of accuracy, speed, and cost)
        speed_score = max(0, 1 - (avg_response_time / 60))  # Normalize to 0-1, 60s = 0
        business_value = (classification_accuracy * 0.4 + speed_score * 0.3 + avg_confidence * 0.3)

        # Identify strengths and weaknesses
        strengths, weaknesses = self._analyze_strengths_weaknesses(
            avg_response_time, avg_cost, classification_accuracy, avg_confidence, success_rate
        )

        return ModelPerformanceReport(
            model_name=model_name,
            provider=analyses[0].model_used if analyses else "unknown",
            classification_accuracy=classification_accuracy,
            overall_accuracy=classification_accuracy,  # Same for now
            avg_response_time=avg_response_time,
            avg_tokens_used=avg_tokens_used,
            avg_cost_per_analysis=avg_cost,
            success_rate=success_rate,
            error_count=len(performance_data) - len(successful_analyses),
            avg_confidence_score=avg_confidence,
            recommendation_quality=avg_recommendation_quality,
            detail_completeness=avg_completeness,
            cost_effectiveness=cost_effectiveness,
            business_value_score=business_value,
            sample_analyses=analyses[:3],  # First 3 as samples
            strengths=strengths,
            weaknesses=weaknesses
        )

    def _analyze_strengths_weaknesses(
        self,
        response_time: float,
        cost: float,
        accuracy: float,
        confidence: float,
        success_rate: float
    ) -> Tuple[List[str], List[str]]:
        """Analyze model strengths and weaknesses based on metrics."""
        strengths = []
        weaknesses = []

        # Response time analysis
        if response_time < 5:
            strengths.append("Very fast response times")
        elif response_time > 30:
            weaknesses.append("Slow response times")

        # Cost analysis
        if cost < 0.01:
            strengths.append("Low cost per analysis")
        elif cost > 0.05:
            weaknesses.append("High cost per analysis")

        # Accuracy analysis
        if accuracy > 0.9:
            strengths.append("High classification accuracy")
        elif accuracy < 0.7:
            weaknesses.append("Low classification accuracy")

        # Confidence analysis
        if confidence > 0.8:
            strengths.append("High confidence in predictions")
        elif confidence < 0.6:
            weaknesses.append("Low confidence in predictions")

        # Reliability analysis
        if success_rate > 0.95:
            strengths.append("Very reliable (high success rate)")
        elif success_rate < 0.9:
            weaknesses.append("Reliability issues (low success rate)")

        return strengths, weaknesses

    def generate_comparison_report(
        self,
        model_reports: Dict[str, ModelPerformanceReport],
        output_format: str = "markdown"
    ) -> str:
        """Generate comprehensive comparison report across all models."""

        if output_format.lower() == "json":
            return json.dumps({name: asdict(report) for name, report in model_reports.items()}, indent=2, default=str)

        # Markdown format
        report = []
        report.append("# Model Comparison Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\nModels evaluated: {len(model_reports)}")

        # Executive Summary
        report.append("\n## Executive Summary")

        # Find best performing model overall
        best_model = max(model_reports.keys(), key=lambda m: model_reports[m].business_value_score)
        best_score = model_reports[best_model].business_value_score

        report.append(f"\n**Recommended Model:** {best_model} (Business Value Score: {best_score:.3f})")

        # Key insights
        fastest_model = min(model_reports.keys(), key=lambda m: model_reports[m].avg_response_time)
        cheapest_model = min(model_reports.keys(), key=lambda m: model_reports[m].avg_cost_per_analysis)
        most_accurate = max(model_reports.keys(), key=lambda m: model_reports[m].classification_accuracy)

        report.append(f"\n**Fastest Model:** {fastest_model} ({model_reports[fastest_model].avg_response_time:.2f}s avg)")
        report.append(f"**Most Cost-Effective:** {cheapest_model} (${model_reports[cheapest_model].avg_cost_per_analysis:.4f} per analysis)")
        report.append(f"**Most Accurate:** {most_accurate} ({model_reports[most_accurate].classification_accuracy:.1%} accuracy)")

        # Detailed Results
        report.append("\n## Detailed Results")

        # Performance table
        report.append("\n### Performance Metrics")
        report.append("| Model | Response Time | Cost | Accuracy | Confidence | Success Rate | Business Value |")
        report.append("|-------|---------------|------|----------|------------|--------------|----------------|")

        for model_name, model_report in sorted(model_reports.items(), key=lambda x: x[1].business_value_score, reverse=True):
            report.append(
                f"| {model_name} | {model_report.avg_response_time:.2f}s | "
                f"${model_report.avg_cost_per_analysis:.4f} | {model_report.classification_accuracy:.1%} | "
                f"{model_report.avg_confidence_score:.2f} | {model_report.success_rate:.1%} | "
                f"{model_report.business_value_score:.3f} |"
            )

        # Individual model details
        report.append("\n### Individual Model Analysis")

        for model_name, model_report in model_reports.items():
            report.append(f"\n#### {model_name}")
            report.append(f"**Provider:** {model_report.provider}")

            if model_report.strengths:
                report.append(f"**Strengths:** {', '.join(model_report.strengths)}")

            if model_report.weaknesses:
                report.append(f"**Weaknesses:** {', '.join(model_report.weaknesses)}")

            report.append(f"**Key Metrics:**")
            report.append(f"- Response Time: {model_report.avg_response_time:.2f}s")
            report.append(f"- Cost per Analysis: ${model_report.avg_cost_per_analysis:.4f}")
            report.append(f"- Classification Accuracy: {model_report.classification_accuracy:.1%}")
            report.append(f"- Success Rate: {model_report.success_rate:.1%}")

        # Recommendations
        report.append("\n## Recommendations")

        report.append(f"\n**For Production Use:** {best_model}")
        report.append(f"- Best overall business value with good balance of accuracy, speed, and cost")

        if fastest_model != best_model:
            report.append(f"\n**For High-Volume/Real-Time:** {fastest_model}")
            report.append(f"- Fastest response times for real-time applications")

        if cheapest_model != best_model:
            report.append(f"\n**For Cost-Sensitive Applications:** {cheapest_model}")
            report.append(f"- Most cost-effective option for large-scale processing")

        return "\n".join(report)

    async def benchmark_model_consistency(
        self,
        model_name: str,
        transcript: str,
        num_runs: int = 5
    ) -> Dict[str, Any]:
        """Test model consistency by running the same transcript multiple times."""

        tasks = [
            self.qa_agent.analyze_transcript(transcript, f"consistency_test_{i}", model_name)
            for i in range(num_runs)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        successful_results = [r for r in results if not isinstance(r, Exception)]

        if not successful_results:
            return {"error": "All runs failed", "consistency_score": 0.0}

        # Calculate consistency metrics
        classifications = [r.primary_classification for r in successful_results]
        confidence_scores = [r.confidence_score for r in successful_results if r.confidence_score is not None]

        # Classification consistency
        most_common_classification = max(set(classifications), key=classifications.count)
        classification_consistency = classifications.count(most_common_classification) / len(classifications)

        # Confidence score variance
        confidence_variance = statistics.variance(confidence_scores) if len(confidence_scores) > 1 else 0

        # Response time consistency
        response_times = [r.analysis_duration for r in successful_results if r.analysis_duration]
        time_variance = statistics.variance(response_times) if len(response_times) > 1 else 0

        return {
            "model": model_name,
            "runs_completed": len(successful_results),
            "runs_failed": len(results) - len(successful_results),
            "classification_consistency": classification_consistency,
            "confidence_variance": confidence_variance,
            "response_time_variance": time_variance,
            "consistency_score": classification_consistency * (1 - min(confidence_variance, 1.0)),
            "all_classifications": classifications,
            "confidence_scores": confidence_scores,
            "response_times": response_times
        }


# Default model comparator instance
model_comparator = ModelComparator()