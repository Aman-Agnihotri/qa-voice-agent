"""
Prompt templates for QA Voice Agent classification tasks.

This module contains carefully crafted prompts for analyzing customer service call transcripts
and providing detailed insights beyond binary classification.
"""

# Primary Classification Prompt
PRIMARY_CLASSIFICATION_PROMPT = """
You are an expert QA analyst for an e-commerce customer support voice agent system. Your task is to analyze call transcripts and provide detailed classification beyond simple "automated vs escalated" metrics.

CLASSIFICATION CATEGORIES:

1. AUTOMATED CALLS:
   - Successful: Customer's issue was fully resolved by the AI agent without escalation
   - Partially Successful: Customer received some help but call ended with unresolved elements
   - Unsuccessful: Customer hung up or call ended with issue completely unresolved

2. ESCALATED CALLS:
   - Partially Successful: AI agent provided useful help before escalating to human support
   - Unsuccessful: Customer immediately wanted human help OR AI failed to provide any useful assistance

ANALYSIS FRAMEWORK:
- Intent Recognition: What was the customer trying to accomplish?
- Issue Resolution: Was the core problem solved?
- Customer Satisfaction Indicators: Tone, explicit feedback, call completion
- Agent Performance: Did the AI provide accurate information and appropriate responses?
- Escalation Necessity: Was escalation appropriate and timely?

TRANSCRIPT TO ANALYZE:
{transcript}

REQUIRED OUTPUT FORMAT:
{{
  "primary_classification": "Automated - Successful/Partially Successful/Unsuccessful OR Escalated - Partially Successful/Unsuccessful",
  "intent": "Brief description of customer's main intent",
  "resolution_status": "Detailed explanation of what was/wasn't resolved",
  "success_indicators": ["List of positive outcomes and good AI responses"],
  "failure_indicators": ["List of issues, missed opportunities, or problems"],
  "customer_satisfaction_signals": "Analysis of customer tone and satisfaction cues",
  "recommendations": ["Specific suggestions for improving similar calls"],
  "confidence_score": "0.0-1.0 confidence in this classification"
}}

Provide thorough analysis focusing on practical insights that could improve the voice agent system.
"""

# Intent Recognition Prompt (for multi-step analysis)
INTENT_RECOGNITION_PROMPT = """
Analyze this customer service call transcript to identify the customer's primary intent and any secondary intents.

COMMON INTENTS:
- Order Status/Tracking
- Returns & Exchanges
- Refund Requests
- Product Information/Questions
- Billing/Payment Issues
- Membership/Account Management
- Technical Support
- Store Information
- General Complaints
- Other (specify)

TRANSCRIPT:
{transcript}

Identify:
1. Primary Intent: Main reason for calling
2. Secondary Intents: Additional issues mentioned
3. Intent Evolution: How the intent changed during the call
4. Complexity Level: Simple, Moderate, or Complex

Provide specific quotes from the transcript to support your analysis.
"""

# Success Metrics Prompt
SUCCESS_METRICS_PROMPT = """
Evaluate this customer service call transcript across key success metrics:

METRICS TO EVALUATE:
1. Issue Resolution (0-10): How completely was the customer's problem solved?
2. Information Accuracy (0-10): Was the information provided correct and helpful?
3. Process Efficiency (0-10): How smoothly did the interaction flow?
4. Customer Experience (0-10): How satisfied was the customer likely to be?
5. Escalation Appropriateness (0-10): If escalated, was it necessary and timely?

TRANSCRIPT:
{transcript}

For each metric, provide:
- Score (0-10)
- Justification with specific examples from the transcript
- Key factors that influenced the score
- Suggestions for improvement

Also identify:
- Critical success moments
- Major failure points
- Missed opportunities
"""

# Failure Analysis Prompt
FAILURE_ANALYSIS_PROMPT = """
Conduct a detailed failure analysis of this customer service call transcript.

Focus on identifying:
1. ROOT CAUSES of any failures or suboptimal outcomes
2. SPECIFIC MOMENTS where things went wrong
3. MISSED OPPORTUNITIES for better service
4. SYSTEMIC ISSUES that could affect other calls

FAILURE CATEGORIES:
- Technical/System Issues
- Agent Knowledge Gaps
- Communication Problems
- Process/Policy Issues
- Customer Expectation Mismatches

TRANSCRIPT:
{transcript}

Provide:
1. Failure severity assessment (Critical/Major/Minor/None)
2. Detailed timeline of where things went wrong
3. Impact analysis on customer experience
4. Specific recommendations for prevention
5. Training implications for the AI system

Be constructive and focus on actionable insights.
"""

# Comparative Analysis Prompt (for model comparison)
COMPARATIVE_ANALYSIS_PROMPT = """
You are comparing AI model performance on customer service call analysis.

EVALUATION CRITERIA:
1. Accuracy: How correct are the classifications and insights?
2. Consistency: Do similar calls get similar analysis?
3. Depth: How detailed and useful are the insights?
4. Actionability: How practical are the recommendations?
5. Business Relevance: How well does the analysis serve business needs?

TRANSCRIPT:
{transcript}

EXPECTED OUTCOME: {expected_outcome}

Provide analysis that demonstrates:
- Clear reasoning for your classification
- Specific evidence from the transcript
- Practical recommendations
- Confidence levels for your assessments

This will be used to compare against other models, so focus on:
- Thoroughness
- Accuracy
- Practical value
- Clear reasoning
"""

# Template for custom prompts based on specific business needs
CUSTOM_BUSINESS_PROMPT = """
Analyze this customer service call with focus on specific business metrics:

BUSINESS PRIORITIES:
{business_priorities}

SPECIFIC ANALYSIS REQUIREMENTS:
{analysis_requirements}

TRANSCRIPT:
{transcript}

Provide analysis that directly addresses the specified business priorities and requirements.
Include quantitative assessments where possible and actionable recommendations.
"""