"""
Evaluation and metrics system for QA Voice Agent.

This module provides comprehensive evaluation capabilities including
accuracy assessment, performance metrics, and business value calculations.
"""

import json
import asyncio
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd

try:
    from .qa_agent import QAAgent, TranscriptAnalysis
    from .model_comparison import ModelComparator, ModelPerformanceReport
except ImportError:
    from qa_agent import QAAgent, TranscriptAnalysis
    from model_comparison import ModelComparator, ModelPerformanceReport


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for QA analysis."""

    # Accuracy Metrics
    classification_accuracy: float = 0.0
    intent_accuracy: float = 0.0
    resolution_accuracy: float = 0.0

    # Quality Metrics
    completeness_score: float = 0.0
    confidence_score: float = 0.0
    recommendation_quality: float = 0.0

    # Performance Metrics
    avg_response_time: float = 0.0
    total_cost: float = 0.0
    success_rate: float = 0.0

    # Business Metrics
    actionability_score: float = 0.0
    business_value_score: float = 0.0

    # Statistical Metrics
    consistency_score: float = 0.0
    reliability_score: float = 0.0

    # Metadata
    total_evaluations: int = 0
    evaluation_timestamp: str = None

    def __post_init__(self):
        if self.evaluation_timestamp is None:
            self.evaluation_timestamp = datetime.now().isoformat()


@dataclass
class GroundTruthData:
    """Ground truth data for evaluation."""
    transcript_id: str
    expected_classification: str
    expected_intent: str
    expected_resolution_status: Optional[str] = None
    success_indicators: Optional[List[str]] = None
    failure_indicators: Optional[List[str]] = None
    notes: Optional[str] = None


class QAEvaluator:
    """
    Comprehensive evaluation system for QA Voice Agent.

    Provides various evaluation capabilities including accuracy assessment,
    performance benchmarking, and business value analysis.
    """

    def __init__(self):
        self.qa_agent = QAAgent()
        self.model_comparator = ModelComparator()

    async def evaluate_with_ground_truth(
        self,
        test_data: List[Dict[str, Any]],
        model: str = "gpt-4",
        detailed: bool = True
    ) -> EvaluationMetrics:
        """
        Evaluate QA agent performance against ground truth data.

        Args:
            test_data: List of test cases with 'transcript', 'id', and ground truth data
            model: Model to evaluate
            detailed: Whether to perform detailed analysis

        Returns:
            Comprehensive evaluation metrics
        """
        print(f"Evaluating model {model} on {len(test_data)} test cases...")

        # Prepare test cases
        evaluation_tasks = []
        ground_truth_cases = []

        for case in test_data:
            # Extract transcript and ground truth
            transcript = case.get('transcript', '')
            transcript_id = case.get('id', f"test_{len(evaluation_tasks)}")

            ground_truth = GroundTruthData(
                transcript_id=transcript_id,
                expected_classification=case.get('expected_outcome', case.get('outcome', '')),
                expected_intent=case.get('intent', ''),
                expected_resolution_status=case.get('expected_resolution'),
                success_indicators=case.get('success_indicators'),
                failure_indicators=case.get('failure_indicators'),
                notes=case.get('notes')
            )

            ground_truth_cases.append(ground_truth)

            # Create analysis task
            task = self.qa_agent.analyze_transcript(
                transcript=transcript,
                transcript_id=transcript_id,
                model=model
            )
            evaluation_tasks.append(task)

        # Run all analyses
        start_time = datetime.now()
        results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)
        total_time = (datetime.now() - start_time).total_seconds()

        # Calculate metrics
        successful_results = []
        failed_results = []

        for result, ground_truth in zip(results, ground_truth_cases):
            if isinstance(result, Exception):
                failed_results.append((result, ground_truth))
            else:
                successful_results.append((result, ground_truth))

        # Calculate comprehensive metrics
        metrics = self._calculate_evaluation_metrics(successful_results, failed_results, total_time)

        if detailed:
            # Add detailed analysis
            await self._add_detailed_evaluation_metrics(metrics, successful_results, model)

        return metrics

    def _calculate_evaluation_metrics(
        self,
        successful_results: List[Tuple[TranscriptAnalysis, GroundTruthData]],
        failed_results: List[Tuple[Exception, GroundTruthData]],
        total_time: float
    ) -> EvaluationMetrics:
        """Calculate core evaluation metrics."""

        total_cases = len(successful_results) + len(failed_results)
        success_rate = len(successful_results) / total_cases if total_cases > 0 else 0

        if not successful_results:
            return EvaluationMetrics(
                success_rate=success_rate,
                total_evaluations=total_cases,
                avg_response_time=total_time / total_cases if total_cases > 0 else 0
            )

        # Accuracy calculations
        classification_matches = 0
        intent_matches = 0
        resolution_matches = 0

        # Quality calculations
        completeness_scores = []
        confidence_scores = []
        recommendation_scores = []

        # Performance calculations
        response_times = []
        costs = []

        for analysis, ground_truth in successful_results:
            # Accuracy assessment
            if self._classifications_match(analysis.primary_classification, ground_truth.expected_classification):
                classification_matches += 1

            if self._intents_match(analysis.intent, ground_truth.expected_intent):
                intent_matches += 1

            if ground_truth.expected_resolution_status and analysis.resolution_status:
                if self._resolutions_match(analysis.resolution_status, ground_truth.expected_resolution_status):
                    resolution_matches += 1

            # Quality assessment
            completeness_scores.append(self._calculate_completeness(analysis))
            if analysis.confidence_score is not None:
                confidence_scores.append(analysis.confidence_score)
            recommendation_scores.append(self._calculate_recommendation_quality(analysis))

            # Performance metrics
            if analysis.analysis_duration:
                response_times.append(analysis.analysis_duration)
            if analysis.cost:
                costs.append(analysis.cost)

        # Calculate averages
        num_successful = len(successful_results)

        classification_accuracy = classification_matches / num_successful
        intent_accuracy = intent_matches / num_successful
        resolution_accuracy = resolution_matches / len([r for r in successful_results if r[1].expected_resolution_status])

        avg_completeness = statistics.mean(completeness_scores) if completeness_scores else 0
        avg_confidence = statistics.mean(confidence_scores) if confidence_scores else 0
        avg_recommendation_quality = statistics.mean(recommendation_scores) if recommendation_scores else 0

        avg_response_time = statistics.mean(response_times) if response_times else 0
        total_cost = sum(costs) if costs else 0

        # Business value calculation
        business_value = (classification_accuracy * 0.4 + avg_confidence * 0.3 + avg_completeness * 0.3)

        # Actionability score (based on presence and quality of recommendations)
        actionability = avg_recommendation_quality

        return EvaluationMetrics(
            classification_accuracy=classification_accuracy,
            intent_accuracy=intent_accuracy,
            resolution_accuracy=resolution_accuracy,
            completeness_score=avg_completeness,
            confidence_score=avg_confidence,
            recommendation_quality=avg_recommendation_quality,
            avg_response_time=avg_response_time,
            total_cost=total_cost,
            success_rate=success_rate,
            actionability_score=actionability,
            business_value_score=business_value,
            total_evaluations=total_cases
        )

    async def _add_detailed_evaluation_metrics(
        self,
        metrics: EvaluationMetrics,
        successful_results: List[Tuple[TranscriptAnalysis, GroundTruthData]],
        model: str
    ):
        """Add detailed evaluation metrics including consistency and reliability."""

        if not successful_results:
            return

        # Consistency analysis - run same transcript multiple times
        sample_transcript = successful_results[0][0]
        if hasattr(sample_transcript, 'transcript_id'):
            # Get the original transcript (this is a simplified approach)
            consistency_data = await self.model_comparator.benchmark_model_consistency(
                model_name=model,
                transcript="Sample transcript for consistency testing",  # Would need actual transcript
                num_runs=3
            )
            metrics.consistency_score = consistency_data.get('consistency_score', 0)

        # Reliability score (combination of success rate and consistency)
        metrics.reliability_score = (metrics.success_rate + metrics.consistency_score) / 2

    def _classifications_match(self, predicted: str, expected: str) -> bool:
        """Check if classifications match (fuzzy matching)."""
        if not predicted or not expected:
            return False

        predicted_lower = predicted.lower()
        expected_lower = expected.lower()

        # Extract key components
        predicted_parts = predicted_lower.split(' - ')
        expected_parts = expected_lower.split(' - ')

        # Check if main category matches (automated vs escalated)
        pred_category = predicted_parts[0] if predicted_parts else predicted_lower
        exp_category = expected_parts[0] if expected_parts else expected_lower

        category_match = ('automated' in pred_category and 'automated' in exp_category) or \
                        ('escalated' in pred_category and 'escalated' in exp_category)

        if not category_match:
            return False

        # Check subcategory if available
        if len(predicted_parts) > 1 and len(expected_parts) > 1:
            pred_sub = predicted_parts[1].strip()
            exp_sub = expected_parts[1].strip()

            return ('successful' in pred_sub and 'successful' in exp_sub) or \
                   ('unsuccessful' in pred_sub and 'unsuccessful' in exp_sub) or \
                   ('partially' in pred_sub and 'partially' in exp_sub)

        return True

    def _intents_match(self, predicted: str, expected: str) -> bool:
        """Check if intents match (fuzzy matching)."""
        if not predicted or not expected:
            return predicted == expected

        predicted_lower = predicted.lower()
        expected_lower = expected.lower()

        # Define intent keywords
        intent_keywords = {
            'order': ['order', 'status', 'tracking'],
            'return': ['return', 'exchange', 'refund'],
            'product': ['product', 'question', 'information'],
            'membership': ['membership', 'account', 'subscription'],
            'billing': ['billing', 'payment', 'charge'],
            'support': ['support', 'help', 'assistance']
        }

        # Find matching intent categories
        pred_category = None
        exp_category = None

        for category, keywords in intent_keywords.items():
            if any(keyword in predicted_lower for keyword in keywords):
                pred_category = category
            if any(keyword in expected_lower for keyword in keywords):
                exp_category = category

        return pred_category == exp_category

    def _resolutions_match(self, predicted: str, expected: str) -> bool:
        """Check if resolution descriptions match conceptually."""
        if not predicted or not expected:
            return predicted == expected

        predicted_lower = predicted.lower()
        expected_lower = expected.lower()

        # Check for key resolution indicators
        resolution_indicators = ['resolved', 'unresolved', 'partial', 'complete', 'successful', 'failed']

        pred_indicators = [ind for ind in resolution_indicators if ind in predicted_lower]
        exp_indicators = [ind for ind in resolution_indicators if ind in expected_lower]

        return bool(set(pred_indicators) & set(exp_indicators))

    def _calculate_completeness(self, analysis: TranscriptAnalysis) -> float:
        """Calculate how complete the analysis is."""
        score = 0.0
        total_components = 8.0

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
        if analysis.confidence_score is not None:
            score += 1.0

        return score / total_components

    def _calculate_recommendation_quality(self, analysis: TranscriptAnalysis) -> float:
        """Calculate quality of recommendations."""
        if not analysis.recommendations:
            return 0.0

        # Quality factors
        num_recommendations = len(analysis.recommendations)
        avg_length = sum(len(rec) for rec in analysis.recommendations) / num_recommendations

        # Scoring
        quantity_score = min(num_recommendations / 3.0, 1.0)  # 3 recommendations is optimal
        quality_score = min(max(avg_length - 20, 0) / 100.0, 1.0)  # 20-120 chars is good

        return (quantity_score + quality_score) / 2.0

    async def cross_validate_models(
        self,
        test_data: List[Dict[str, Any]],
        models: List[str],
        k_folds: int = 3
    ) -> Dict[str, EvaluationMetrics]:
        """
        Perform k-fold cross-validation across multiple models.

        Args:
            test_data: Test dataset
            models: List of models to evaluate
            k_folds: Number of folds for cross-validation

        Returns:
            Dictionary mapping model names to average evaluation metrics
        """
        print(f"Starting {k_folds}-fold cross-validation for {len(models)} models...")

        # Split data into folds
        fold_size = len(test_data) // k_folds
        folds = [test_data[i:i + fold_size] for i in range(0, len(test_data), fold_size)]

        # If there's a remainder, add it to the last fold
        if len(folds) > k_folds:
            folds[k_folds - 1].extend(folds[k_folds])
            folds = folds[:k_folds]

        model_metrics = {model: [] for model in models}

        # Run cross-validation
        for fold_idx in range(k_folds):
            print(f"Processing fold {fold_idx + 1}/{k_folds}...")

            test_fold = folds[fold_idx]

            # Evaluate each model on this fold
            for model in models:
                try:
                    metrics = await self.evaluate_with_ground_truth(test_fold, model, detailed=False)
                    model_metrics[model].append(metrics)
                except Exception as e:
                    print(f"Error evaluating {model} on fold {fold_idx + 1}: {e}")

        # Calculate average metrics for each model
        averaged_metrics = {}
        for model, metrics_list in model_metrics.items():
            if metrics_list:
                averaged_metrics[model] = self._average_metrics(metrics_list)
            else:
                averaged_metrics[model] = EvaluationMetrics()

        return averaged_metrics

    def _average_metrics(self, metrics_list: List[EvaluationMetrics]) -> EvaluationMetrics:
        """Calculate average metrics across multiple evaluations."""
        if not metrics_list:
            return EvaluationMetrics()

        # Calculate means for all numeric fields
        return EvaluationMetrics(
            classification_accuracy=statistics.mean([m.classification_accuracy for m in metrics_list]),
            intent_accuracy=statistics.mean([m.intent_accuracy for m in metrics_list]),
            resolution_accuracy=statistics.mean([m.resolution_accuracy for m in metrics_list]),
            completeness_score=statistics.mean([m.completeness_score for m in metrics_list]),
            confidence_score=statistics.mean([m.confidence_score for m in metrics_list]),
            recommendation_quality=statistics.mean([m.recommendation_quality for m in metrics_list]),
            avg_response_time=statistics.mean([m.avg_response_time for m in metrics_list]),
            total_cost=sum([m.total_cost for m in metrics_list]),
            success_rate=statistics.mean([m.success_rate for m in metrics_list]),
            actionability_score=statistics.mean([m.actionability_score for m in metrics_list]),
            business_value_score=statistics.mean([m.business_value_score for m in metrics_list]),
            consistency_score=statistics.mean([m.consistency_score for m in metrics_list]),
            reliability_score=statistics.mean([m.reliability_score for m in metrics_list]),
            total_evaluations=sum([m.total_evaluations for m in metrics_list])
        )

    def generate_evaluation_report(
        self,
        metrics: EvaluationMetrics,
        model_name: str,
        test_description: str = ""
    ) -> str:
        """Generate a comprehensive evaluation report."""

        report = []
        report.append(f"# Evaluation Report - {model_name}")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if test_description:
            report.append(f"\n**Test Description:** {test_description}")

        report.append(f"\n**Total Evaluations:** {metrics.total_evaluations}")

        # Executive Summary
        report.append("\n## Executive Summary")

        # Overall grade
        overall_score = metrics.business_value_score
        if overall_score >= 0.9:
            grade = "A (Excellent)"
        elif overall_score >= 0.8:
            grade = "B (Good)"
        elif overall_score >= 0.7:
            grade = "C (Satisfactory)"
        elif overall_score >= 0.6:
            grade = "D (Needs Improvement)"
        else:
            grade = "F (Poor)"

        report.append(f"\n**Overall Grade:** {grade} (Score: {overall_score:.3f})")

        # Key metrics
        report.append(f"\n**Key Performance Indicators:**")
        report.append(f"- Classification Accuracy: {metrics.classification_accuracy:.1%}")
        report.append(f"- Success Rate: {metrics.success_rate:.1%}")
        report.append(f"- Average Response Time: {metrics.avg_response_time:.2f}s")
        report.append(f"- Average Cost: ${metrics.total_cost / max(metrics.total_evaluations, 1):.4f} per analysis")

        # Detailed Metrics
        report.append("\n## Detailed Metrics")

        # Accuracy section
        report.append("\n### Accuracy Metrics")
        report.append(f"- **Classification Accuracy:** {metrics.classification_accuracy:.1%}")
        report.append(f"- **Intent Recognition Accuracy:** {metrics.intent_accuracy:.1%}")
        report.append(f"- **Resolution Assessment Accuracy:** {metrics.resolution_accuracy:.1%}")

        # Quality section
        report.append("\n### Quality Metrics")
        report.append(f"- **Analysis Completeness:** {metrics.completeness_score:.1%}")
        report.append(f"- **Average Confidence:** {metrics.confidence_score:.1%}")
        report.append(f"- **Recommendation Quality:** {metrics.recommendation_quality:.1%}")

        # Performance section
        report.append("\n### Performance Metrics")
        report.append(f"- **Average Response Time:** {metrics.avg_response_time:.2f} seconds")
        report.append(f"- **Success Rate:** {metrics.success_rate:.1%}")
        report.append(f"- **Total Cost:** ${metrics.total_cost:.4f}")

        # Business Value section
        report.append("\n### Business Value")
        report.append(f"- **Business Value Score:** {metrics.business_value_score:.3f}")
        report.append(f"- **Actionability Score:** {metrics.actionability_score:.3f}")
        report.append(f"- **Reliability Score:** {metrics.reliability_score:.3f}")

        # Recommendations
        report.append("\n## Recommendations")

        if metrics.classification_accuracy < 0.8:
            report.append("- **Improve Classification Accuracy**: Consider prompt engineering or model fine-tuning")

        if metrics.avg_response_time > 10:
            report.append("- **Optimize Response Time**: Consider using faster models or caching strategies")

        if metrics.recommendation_quality < 0.7:
            report.append("- **Enhance Recommendations**: Improve prompt templates for more actionable insights")

        if metrics.success_rate < 0.95:
            report.append("- **Improve Reliability**: Add error handling and fallback mechanisms")

        # Conclusion
        report.append("\n## Conclusion")

        if overall_score >= 0.8:
            report.append("The model demonstrates strong performance across key metrics and is suitable for production use.")
        elif overall_score >= 0.7:
            report.append("The model shows good performance with some areas for improvement. Consider addressing the recommendations above.")
        else:
            report.append("The model requires significant improvement before production deployment. Focus on the key areas identified in the recommendations.")

        return "\n".join(report)

    def export_metrics_to_csv(self, metrics_dict: Dict[str, EvaluationMetrics], filename: str):
        """Export evaluation metrics to CSV file."""
        data = []
        for model_name, metrics in metrics_dict.items():
            row = {
                'model': model_name,
                **asdict(metrics)
            }
            data.append(row)

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Metrics exported to {filename}")

    def load_ground_truth_from_json(self, filepath: str) -> List[Dict[str, Any]]:
        """Load ground truth data from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def create_ground_truth_template(self, transcripts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create ground truth template for manual annotation."""
        template = []
        for transcript in transcripts:
            template.append({
                'id': transcript.get('id', ''),
                'transcript': transcript.get('transcript', ''),
                'expected_outcome': '',  # To be filled manually
                'intent': '',  # To be filled manually
                'expected_resolution': '',  # To be filled manually
                'success_indicators': [],  # To be filled manually
                'failure_indicators': [],  # To be filled manually
                'notes': ''  # To be filled manually
            })
        return template


# Default evaluator instance
qa_evaluator = QAEvaluator()