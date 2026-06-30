"""
Main QA Agent for analyzing customer service call transcripts.

This module provides the core functionality for analyzing call transcripts
and providing detailed insights beyond binary classification.
"""

import json
import asyncio
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    from .prompts import (
        PRIMARY_CLASSIFICATION_PROMPT,
        INTENT_RECOGNITION_PROMPT,
        SUCCESS_METRICS_PROMPT,
        FAILURE_ANALYSIS_PROMPT,
    )
    from .utils import model_manager, ModelResponse, settings, AnalysisMode
except ImportError:
    # Handle direct execution
    from prompts import (
        PRIMARY_CLASSIFICATION_PROMPT,
        INTENT_RECOGNITION_PROMPT,
        SUCCESS_METRICS_PROMPT,
        FAILURE_ANALYSIS_PROMPT,
    )
    from utils import model_manager, ModelResponse, settings, AnalysisMode


@dataclass
class TranscriptAnalysis:
    """Structured analysis result for a call transcript."""

    # Basic information
    transcript_id: Optional[str] = None
    analysis_timestamp: str = None
    model_used: str = None

    # Primary classification
    primary_classification: str = None
    intent: str = None
    resolution_status: str = None

    # Analysis details
    success_indicators: List[str] = None
    failure_indicators: List[str] = None
    customer_satisfaction_signals: str = None
    recommendations: List[str] = None
    confidence_score: float = None

    # Detailed metrics (for detailed analysis mode)
    issue_resolution_score: Optional[float] = None
    information_accuracy_score: Optional[float] = None
    process_efficiency_score: Optional[float] = None
    customer_experience_score: Optional[float] = None
    escalation_appropriateness_score: Optional[float] = None

    # Additional insights
    intent_details: Optional[Dict[str, Any]] = None
    failure_analysis: Optional[Dict[str, Any]] = None

    # Metadata
    analysis_duration: Optional[float] = None
    tokens_used: Optional[int] = None
    cost: Optional[float] = None

    def __post_init__(self):
        if self.analysis_timestamp is None:
            self.analysis_timestamp = datetime.now().isoformat()
        if self.success_indicators is None:
            self.success_indicators = []
        if self.failure_indicators is None:
            self.failure_indicators = []
        if self.recommendations is None:
            self.recommendations = []


class QAAgent:
    """
    Main QA Agent for analyzing customer service call transcripts.

    Provides comprehensive analysis beyond simple binary classification,
    including success metrics, failure analysis, and actionable recommendations.
    """

    def __init__(self, default_model: str = "gpt-4"):
        self.default_model = default_model
        self.model_manager = model_manager

    async def analyze_transcript(
        self,
        transcript: str,
        transcript_id: Optional[str] = None,
        model: Optional[str] = None,
        analysis_mode: AnalysisMode = AnalysisMode.DETAILED
    ) -> TranscriptAnalysis:
        """
        Analyze a single call transcript.

        Args:
            transcript: The call transcript text
            transcript_id: Optional identifier for the transcript
            model: Model to use for analysis (uses default if not specified)
            analysis_mode: Level of analysis detail (quick, detailed, comparative)

        Returns:
            TranscriptAnalysis object with comprehensive analysis results
        """
        start_time = datetime.now()
        model_name = model or self.default_model

        try:
            # Primary classification analysis
            primary_response = await self._get_primary_classification(transcript, model_name)
            analysis = self._parse_primary_response(primary_response)

            # Set basic metadata
            analysis.transcript_id = transcript_id
            analysis.model_used = model_name
            analysis.tokens_used = primary_response.tokens_used
            analysis.cost = primary_response.cost

            if analysis_mode in [AnalysisMode.DETAILED, AnalysisMode.COMPARATIVE]:
                # Additional detailed analysis
                await self._add_detailed_analysis(transcript, analysis, model_name)

            # Calculate analysis duration
            analysis.analysis_duration = (datetime.now() - start_time).total_seconds()

            return analysis

        except Exception as e:
            # Return error analysis
            return TranscriptAnalysis(
                transcript_id=transcript_id,
                model_used=model_name,
                primary_classification="Error - Analysis Failed",
                intent="Analysis Error",
                resolution_status=f"Failed to analyze transcript: {str(e)}",
                confidence_score=0.0,
                failure_indicators=[f"Analysis error: {str(e)}"],
                analysis_duration=(datetime.now() - start_time).total_seconds()
            )

    async def _get_primary_classification(self, transcript: str, model: str) -> ModelResponse:
        """Get primary classification for the transcript."""
        prompt = PRIMARY_CLASSIFICATION_PROMPT.format(transcript=transcript)
        return await self.model_manager.generate(model, prompt)

    def _parse_primary_response(self, response: ModelResponse) -> TranscriptAnalysis:
        """Parse the primary classification response into structured data."""
        try:
            # Try to parse as JSON first
            if response.content.strip().startswith('{'):
                data = json.loads(response.content)
                return TranscriptAnalysis(
                    primary_classification=data.get("primary_classification"),
                    intent=data.get("intent"),
                    resolution_status=data.get("resolution_status"),
                    success_indicators=data.get("success_indicators", []),
                    failure_indicators=data.get("failure_indicators", []),
                    customer_satisfaction_signals=data.get("customer_satisfaction_signals"),
                    recommendations=data.get("recommendations", []),
                    confidence_score=data.get("confidence_score")
                )
            else:
                # Parse structured text response
                return self._parse_text_response(response.content)

        except json.JSONDecodeError:
            # Fallback to text parsing
            return self._parse_text_response(response.content)

    def _parse_text_response(self, content: str) -> TranscriptAnalysis:
        """Parse text response when JSON parsing fails."""
        lines = content.split('\n')
        analysis = TranscriptAnalysis()

        # Simple text parsing logic
        for line in lines:
            line = line.strip()
            if 'classification' in line.lower():
                analysis.primary_classification = line.split(':', 1)[-1].strip()
            elif 'intent' in line.lower():
                analysis.intent = line.split(':', 1)[-1].strip()
            elif 'resolution' in line.lower():
                analysis.resolution_status = line.split(':', 1)[-1].strip()
            elif 'confidence' in line.lower():
                try:
                    confidence_str = line.split(':', 1)[-1].strip()
                    analysis.confidence_score = float(confidence_str)
                except:
                    analysis.confidence_score = 0.5

        # Set defaults if not found
        if not analysis.primary_classification:
            analysis.primary_classification = "Automated - Partially Successful"
        if not analysis.intent:
            analysis.intent = "General Support"
        if not analysis.resolution_status:
            analysis.resolution_status = "Analysis completed but detailed resolution status unclear"
        if analysis.confidence_score is None:
            analysis.confidence_score = 0.5

        return analysis

    async def _add_detailed_analysis(self, transcript: str, analysis: TranscriptAnalysis, model: str):
        """Add detailed analysis including intent recognition, metrics, and failure analysis."""

        # Run multiple analysis types in parallel
        tasks = [
            self._get_intent_analysis(transcript, model),
            self._get_success_metrics(transcript, model),
            self._get_failure_analysis(transcript, model)
        ]

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Process intent analysis
        if not isinstance(responses[0], Exception):
            intent_data = self._parse_intent_response(responses[0])
            analysis.intent_details = intent_data

        # Process success metrics
        if not isinstance(responses[1], Exception):
            metrics_data = self._parse_metrics_response(responses[1])
            analysis.issue_resolution_score = metrics_data.get("issue_resolution")
            analysis.information_accuracy_score = metrics_data.get("information_accuracy")
            analysis.process_efficiency_score = metrics_data.get("process_efficiency")
            analysis.customer_experience_score = metrics_data.get("customer_experience")
            analysis.escalation_appropriateness_score = metrics_data.get("escalation_appropriateness")

        # Process failure analysis
        if not isinstance(responses[2], Exception):
            failure_data = self._parse_failure_response(responses[2])
            analysis.failure_analysis = failure_data

    async def _get_intent_analysis(self, transcript: str, model: str) -> ModelResponse:
        """Get detailed intent analysis."""
        prompt = INTENT_RECOGNITION_PROMPT.format(transcript=transcript)
        return await self.model_manager.generate(model, prompt)

    async def _get_success_metrics(self, transcript: str, model: str) -> ModelResponse:
        """Get success metrics analysis."""
        prompt = SUCCESS_METRICS_PROMPT.format(transcript=transcript)
        return await self.model_manager.generate(model, prompt)

    async def _get_failure_analysis(self, transcript: str, model: str) -> ModelResponse:
        """Get failure analysis."""
        prompt = FAILURE_ANALYSIS_PROMPT.format(transcript=transcript)
        return await self.model_manager.generate(model, prompt)

    def _parse_intent_response(self, response: ModelResponse) -> Dict[str, Any]:
        """Parse intent analysis response."""
        # Simple parsing - in production, this would be more sophisticated
        return {
            "raw_analysis": response.content,
            "timestamp": datetime.now().isoformat()
        }

    def _parse_metrics_response(self, response: ModelResponse) -> Dict[str, float]:
        """Parse success metrics response."""
        content = response.content
        metrics = {}

        # Extract numeric scores from the response
        import re
        score_pattern = r'(\w+.*?):\s*(\d+(?:\.\d+)?)'
        matches = re.findall(score_pattern, content, re.IGNORECASE)

        for metric, score in matches:
            metric_key = metric.lower().replace(' ', '_').replace('-', '_')
            try:
                score_value = float(score)
                if 'resolution' in metric_key:
                    metrics['issue_resolution'] = score_value
                elif 'accuracy' in metric_key:
                    metrics['information_accuracy'] = score_value
                elif 'efficiency' in metric_key:
                    metrics['process_efficiency'] = score_value
                elif 'experience' in metric_key:
                    metrics['customer_experience'] = score_value
                elif 'escalation' in metric_key:
                    metrics['escalation_appropriateness'] = score_value
            except ValueError:
                continue

        return metrics

    def _parse_failure_response(self, response: ModelResponse) -> Dict[str, Any]:
        """Parse failure analysis response."""
        return {
            "raw_analysis": response.content,
            "timestamp": datetime.now().isoformat()
        }

    async def batch_analyze(
        self,
        transcripts: List[Dict[str, Any]],
        model: Optional[str] = None,
        analysis_mode: AnalysisMode = AnalysisMode.DETAILED,
        max_concurrent: int = 5
    ) -> List[TranscriptAnalysis]:
        """
        Analyze multiple transcripts in batch.

        Args:
            transcripts: List of transcript dictionaries with 'id' and 'transcript' keys
            model: Model to use for analysis
            analysis_mode: Level of analysis detail
            max_concurrent: Maximum concurrent analyses

        Returns:
            List of TranscriptAnalysis results
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def analyze_single(transcript_data):
            async with semaphore:
                return await self.analyze_transcript(
                    transcript=transcript_data.get("transcript", ""),
                    transcript_id=transcript_data.get("id"),
                    model=model,
                    analysis_mode=analysis_mode
                )

        tasks = [analyze_single(t) for t in transcripts]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def export_analysis(self, analysis: TranscriptAnalysis, format: str = "json") -> str:
        """Export analysis results in various formats."""
        data = asdict(analysis)

        if format.lower() == "json":
            return json.dumps(data, indent=2, default=str)
        elif format.lower() == "csv":
            # Flatten the data for CSV export
            flattened = self._flatten_dict(data)
            import csv
            import io
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=flattened.keys())
            writer.writeheader()
            writer.writerow(flattened)
            return output.getvalue()
        else:
            return str(data)

    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary for CSV export."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                items.append((new_key, '; '.join(map(str, v)) if v else ''))
            else:
                items.append((new_key, v))
        return dict(items)


# Default QA Agent instance
qa_agent = QAAgent()