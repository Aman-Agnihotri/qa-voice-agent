"""
Model integration utilities for various AI providers.
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

# API clients - make imports optional
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    from mistralai.client import MistralClient
except ImportError:
    MistralClient = None

import aiohttp

try:
    from .config import settings, ModelProvider, get_model_config
except ImportError:
    from config import settings, ModelProvider, get_model_config


@dataclass
class ModelResponse:
    """Standardized response from any model."""
    content: str
    provider: str
    model: str
    tokens_used: Optional[int] = None
    response_time: Optional[float] = None
    cost: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ModelMetrics:
    """Performance metrics for model comparison."""
    provider: str
    model: str
    avg_response_time: float
    total_tokens: int
    total_cost: float
    success_rate: float
    error_count: int


class BaseModelClient(ABC):
    """Abstract base class for model clients."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.config = get_model_config(model_name)
        self.provider = self.config["provider"]

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate a response from the model."""
        pass

    def calculate_cost(self, tokens: int) -> float:
        """Calculate cost based on token usage."""
        cost_per_1k = self.config.get("cost_per_1k_tokens", 0)
        return (tokens / 1000) * cost_per_1k


class OpenAIClient(BaseModelClient):
    """OpenAI API client."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        if openai is None:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        self.client = openai.AsyncOpenAI(api_key=settings.openai_api_key)

    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        start_time = time.time()

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get("max_tokens", self.config["max_tokens"]),
                temperature=kwargs.get("temperature", self.config["temperature"]),
            )

            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            response_time = time.time() - start_time
            cost = self.calculate_cost(tokens_used)

            return ModelResponse(
                content=content,
                provider=self.provider.value,
                model=self.model_name,
                tokens_used=tokens_used,
                response_time=response_time,
                cost=cost,
                metadata={"finish_reason": response.choices[0].finish_reason}
            )

        except Exception as e:
            return ModelResponse(
                content=f"Error: {str(e)}",
                provider=self.provider.value,
                model=self.model_name,
                response_time=time.time() - start_time,
                metadata={"error": str(e)}
            )


class AnthropicClient(BaseModelClient):
    """Anthropic API client."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        start_time = time.time()

        try:
            response = await self.client.messages.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get("max_tokens", self.config["max_tokens"]),
                temperature=kwargs.get("temperature", self.config["temperature"]),
            )

            content = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            response_time = time.time() - start_time
            cost = self.calculate_cost(tokens_used)

            return ModelResponse(
                content=content,
                provider=self.provider.value,
                model=self.model_name,
                tokens_used=tokens_used,
                response_time=response_time,
                cost=cost,
                metadata={"stop_reason": response.stop_reason}
            )

        except Exception as e:
            return ModelResponse(
                content=f"Error: {str(e)}",
                provider=self.provider.value,
                model=self.model_name,
                response_time=time.time() - start_time,
                metadata={"error": str(e)}
            )


class GroqClient(BaseModelClient):
    """Groq API client."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = Groq(api_key=settings.groq_api_key)

    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        start_time = time.time()

        try:
            # Groq client is synchronous, so we'll run it in a thread
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=kwargs.get("max_tokens", self.config["max_tokens"]),
                    temperature=kwargs.get("temperature", self.config["temperature"]),
                )
            )

            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            response_time = time.time() - start_time
            cost = self.calculate_cost(tokens_used)

            return ModelResponse(
                content=content,
                provider=self.provider.value,
                model=self.model_name,
                tokens_used=tokens_used,
                response_time=response_time,
                cost=cost,
                metadata={"finish_reason": response.choices[0].finish_reason}
            )

        except Exception as e:
            return ModelResponse(
                content=f"Error: {str(e)}",
                provider=self.provider.value,
                model=self.model_name,
                response_time=time.time() - start_time,
                metadata={"error": str(e)}
            )


class GoogleClient(BaseModelClient):
    """Google Generative AI client."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        genai.configure(api_key=settings.google_api_key)
        self.client = genai.GenerativeModel(model_name)

    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        start_time = time.time()

        try:
            # Google client is synchronous
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=kwargs.get("max_tokens", self.config["max_tokens"]),
                        temperature=kwargs.get("temperature", self.config["temperature"]),
                    )
                )
            )

            content = response.text
            tokens_used = response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else None
            response_time = time.time() - start_time
            cost = self.calculate_cost(tokens_used) if tokens_used else 0

            return ModelResponse(
                content=content,
                provider=self.provider.value,
                model=self.model_name,
                tokens_used=tokens_used,
                response_time=response_time,
                cost=cost,
                metadata={"finish_reason": response.candidates[0].finish_reason.name if response.candidates else None}
            )

        except Exception as e:
            return ModelResponse(
                content=f"Error: {str(e)}",
                provider=self.provider.value,
                model=self.model_name,
                response_time=time.time() - start_time,
                metadata={"error": str(e)}
            )


class MistralClient(BaseModelClient):
    """Mistral AI client."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = MistralClient(api_key=settings.mistral_api_key)

    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        start_time = time.time()

        try:
            # Mistral client is synchronous
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=kwargs.get("max_tokens", self.config["max_tokens"]),
                    temperature=kwargs.get("temperature", self.config["temperature"]),
                )
            )

            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            response_time = time.time() - start_time
            cost = self.calculate_cost(tokens_used)

            return ModelResponse(
                content=content,
                provider=self.provider.value,
                model=self.model_name,
                tokens_used=tokens_used,
                response_time=response_time,
                cost=cost,
                metadata={"finish_reason": response.choices[0].finish_reason}
            )

        except Exception as e:
            return ModelResponse(
                content=f"Error: {str(e)}",
                provider=self.provider.value,
                model=self.model_name,
                response_time=time.time() - start_time,
                metadata={"error": str(e)}
            )


class OllamaClient(BaseModelClient):
    """Ollama (local models) client."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.base_url = settings.ollama_base_url

    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        start_time = time.time()

        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get("temperature", self.config["temperature"]),
                        "num_predict": kwargs.get("max_tokens", self.config["max_tokens"]),
                    }
                }

                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                ) as response:
                    result = await response.json()

                    content = result.get("response", "")
                    response_time = time.time() - start_time

                    # Ollama doesn't provide token counts in some versions
                    tokens_used = result.get("prompt_eval_count", 0) + result.get("eval_count", 0)

                    return ModelResponse(
                        content=content,
                        provider=self.provider.value,
                        model=self.model_name,
                        tokens_used=tokens_used if tokens_used > 0 else None,
                        response_time=response_time,
                        cost=0.0,  # Local model, no cost
                        metadata={"done": result.get("done")}
                    )

        except Exception as e:
            return ModelResponse(
                content=f"Error: {str(e)}",
                provider=self.provider.value,
                model=self.model_name,
                response_time=time.time() - start_time,
                metadata={"error": str(e)}
            )


class ModelManager:
    """Manages multiple model clients and provides unified interface."""

    def __init__(self):
        self.clients: Dict[str, BaseModelClient] = {}
        self.metrics: Dict[str, List[ModelResponse]] = {}

    def add_model(self, model_name: str) -> bool:
        """Add a model client."""
        try:
            config = get_model_config(model_name)
            provider = config["provider"]

            client_classes = {
                ModelProvider.OPENAI: OpenAIClient,
                ModelProvider.ANTHROPIC: AnthropicClient,
                ModelProvider.GROQ: GroqClient,
                ModelProvider.GOOGLE: GoogleClient,
                ModelProvider.MISTRAL: MistralClient,
                ModelProvider.OLLAMA: OllamaClient,
            }

            client_class = client_classes.get(provider)
            if client_class:
                self.clients[model_name] = client_class(model_name)
                self.metrics[model_name] = []
                return True
            return False

        except Exception as e:
            print(f"Failed to add model {model_name}: {e}")
            return False

    async def generate(self, model_name: str, prompt: str, **kwargs) -> ModelResponse:
        """Generate response from a specific model."""
        if model_name not in self.clients:
            self.add_model(model_name)

        client = self.clients.get(model_name)
        if not client:
            return ModelResponse(
                content="Error: Model not available",
                provider="unknown",
                model=model_name,
                metadata={"error": "Model not configured"}
            )

        response = await client.generate(prompt, **kwargs)
        self.metrics[model_name].append(response)
        return response

    async def compare_models(self, models: List[str], prompt: str, **kwargs) -> Dict[str, ModelResponse]:
        """Generate responses from multiple models for comparison."""
        tasks = [self.generate(model, prompt, **kwargs) for model in models]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        result = {}
        for model, response in zip(models, responses):
            if isinstance(response, Exception):
                result[model] = ModelResponse(
                    content=f"Error: {str(response)}",
                    provider="unknown",
                    model=model,
                    metadata={"error": str(response)}
                )
            else:
                result[model] = response

        return result

    def get_metrics(self, model_name: str) -> Optional[ModelMetrics]:
        """Get performance metrics for a model."""
        responses = self.metrics.get(model_name, [])
        if not responses:
            return None

        successful_responses = [r for r in responses if not r.metadata or "error" not in r.metadata]

        if not successful_responses:
            return ModelMetrics(
                provider=responses[0].provider,
                model=model_name,
                avg_response_time=0,
                total_tokens=0,
                total_cost=0,
                success_rate=0,
                error_count=len(responses)
            )

        avg_response_time = sum(r.response_time for r in successful_responses if r.response_time) / len(successful_responses)
        total_tokens = sum(r.tokens_used for r in successful_responses if r.tokens_used)
        total_cost = sum(r.cost for r in successful_responses if r.cost)
        success_rate = len(successful_responses) / len(responses)
        error_count = len(responses) - len(successful_responses)

        return ModelMetrics(
            provider=successful_responses[0].provider,
            model=model_name,
            avg_response_time=avg_response_time,
            total_tokens=total_tokens,
            total_cost=total_cost,
            success_rate=success_rate,
            error_count=error_count
        )

    def get_all_metrics(self) -> Dict[str, ModelMetrics]:
        """Get metrics for all models."""
        return {model: self.get_metrics(model) for model in self.metrics.keys()}


# Global model manager instance
model_manager = ModelManager()