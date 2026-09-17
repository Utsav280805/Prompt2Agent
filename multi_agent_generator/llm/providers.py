# multi_agent_generator/llm/providers.py
"""
Concrete LLM providers.

Each class here knows exactly one thing the rest of the application must not: how to
turn a :class:`~multi_agent_generator.llm.config.LLMConfig` into a request for one
vendor. They are deliberately thin - metadata plus a single ``_invoke`` - because
anything shared (credential validation, error translation, timing, token accounting)
belongs in :class:`~multi_agent_generator.llm.base.LLMProvider` and is inherited.

Two design decisions worth stating, because both were bugs before:

*Hugging Face goes through* ``huggingface_hub.InferenceClient``. That is the officially
supported client for the inference router, it is the package already named in the
``huggingface`` extra, and it accepts the token explicitly. The previous approach handed
the repo id to LiteLLM and hoped the ``HF_TOKEN`` environment variable would be picked up
by whatever layer needed it, which is what produced ``You must provide an api_key to work
with auto API`` from six frames inside the hub client.

*Route prefixes are added here, not stored.* ``LLMConfig.model`` is always the plain
model id; each LiteLLM-backed class prepends its own ``route_prefix``. Storing a prefixed
id is how ``huggingface/huggingface/Qwen/...`` happened.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from ..errors import LLMError, LLMResponseError
from .base import LLMProvider, LLMResponse

__all__ = [
    "LiteLLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "GroqProvider",
    "GoogleProvider",
    "WatsonxProvider",
    "OllamaProvider",
    "HuggingFaceProvider",
    "LocalTransformersProvider",
    "MockProvider",
]


# ======================================================================== LiteLLM base
class LiteLLMProvider(LLMProvider):
    """
    Anything reachable through LiteLLM's unified ``completion`` call.

    LiteLLM chooses its backend from the ``provider/`` prefix on the model id, so the
    only per-vendor state most providers need is ``route_prefix``.
    """

    required_packages = ("litellm",)
    #: Prefix LiteLLM uses to route this provider. Empty means "no prefix" (OpenAI).
    route_prefix: str = ""
    #: Whether to send ``api_base``. Off for vendors that manage their own endpoint.
    send_base_url: bool = False

    def routed_model(self) -> str:
        if not self.route_prefix:
            return self.config.model
        return f"{self.route_prefix}/{self.config.model}"

    def _invoke(self, messages: List[Dict[str, str]], params: Dict[str, Any]) -> LLMResponse:
        from litellm import completion

        kwargs: Dict[str, Any] = {
            "model": self.routed_model(),
            "messages": messages,
            **params,
        }
        key = self.config.api_key.reveal()
        if key:
            kwargs["api_key"] = key
        if self.send_base_url and self.config.base_url:
            kwargs["api_base"] = self.config.base_url
        kwargs.update(self._extra_kwargs())

        raw = completion(**kwargs)
        text = ""
        finish: Optional[str] = None
        try:
            choice = raw.choices[0]
            text = (choice.message.content or "") if choice.message else ""
            finish = getattr(choice, "finish_reason", None)
        except (AttributeError, IndexError, KeyError) as exc:
            raise LLMResponseError(
                f"{self.label} returned a response this client could not read.",
                detail=f"{type(exc).__name__}: {exc}",
                context={"provider": self.name},
            ) from exc

        usage = self._usage_from(raw)
        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.config.model,
            finish_reason=finish,
            **usage,
        )

    def _extra_kwargs(self) -> Dict[str, Any]:
        """Hook for vendor-specific arguments that are not sampling parameters."""
        return {}


# ============================================================================ hosted
class OpenAIProvider(LiteLLMProvider):
    name = "openai"
    label = "OpenAI"
    route_prefix = ""
    #: Honoured so that any OpenAI-compatible gateway (LiteLLM proxy, vLLM, LM Studio,
    #: OpenRouter) can be used by setting a base URL - no new provider class needed.
    send_base_url = True


class AnthropicProvider(LiteLLMProvider):
    name = "anthropic"
    label = "Anthropic"
    route_prefix = "anthropic"


class GroqProvider(LiteLLMProvider):
    name = "groq"
    label = "Groq"
    route_prefix = "groq"


class GoogleProvider(LiteLLMProvider):
    name = "google"
    label = "Google Gemini"
    # LiteLLM routes the public Gemini API under `gemini/`; `google/` is the Vertex path
    # and needs service-account credentials rather than an API key.
    route_prefix = "gemini"


class WatsonxProvider(LiteLLMProvider):
    name = "watsonx"
    label = "IBM watsonx"
    route_prefix = "watsonx"
    send_base_url = True

    def _extra_kwargs(self) -> Dict[str, Any]:
        import os

        project_id = os.getenv("WATSONX_PROJECT_ID") or self.config.extra.get("project_id")
        return {"project_id": project_id} if project_id else {}


class OllamaProvider(LiteLLMProvider):
    name = "ollama"
    label = "Ollama (local)"
    route_prefix = "ollama"
    send_base_url = True
    needs_credential = False
    local = True


# ===================================================================== Hugging Face
class HuggingFaceProvider(LLMProvider):
    """
    Hugging Face's hosted inference router.

    Uses ``huggingface_hub.InferenceClient.chat_completion``, which is the OpenAI-shaped
    entry point to the router and accepts the token as an explicit argument. Passing it
    explicitly - rather than relying on ambient environment discovery - is what makes a
    missing token a clean :class:`MissingCredentialError` raised by
    :meth:`LLMProvider.validate` before any request is built.
    """

    name = "huggingface"
    label = "Hugging Face"
    required_packages = ("huggingface_hub",)
    # The router implements the OpenAI chat schema but not the penalty parameters; it
    # answers 422 rather than ignoring them, so they are dropped rather than sent.
    drop_params = ("frequency_penalty", "presence_penalty")

    def _invoke(self, messages: List[Dict[str, str]], params: Dict[str, Any]) -> LLMResponse:
        from huggingface_hub import InferenceClient

        from ..providers import HF_ROUTER_BASE_URL

        token = self.config.api_key.reveal()
        # huggingface_hub 1.x needs ``provider`` so the call hits the inference router
        # rather than the retired api-inference host (that path is what produced the
        # generic "Hugging Face request failed" warning on analysis).
        client_kwargs: Dict[str, Any] = {"api_key": token, "token": token, "provider": "auto"}
        timeout = params.pop("timeout", None)
        if timeout:
            client_kwargs["timeout"] = float(timeout)
        # Custom endpoints only. Combining base_url with provider="auto" is rejected.
        if self.config.base_url and self.config.base_url != HF_ROUTER_BASE_URL:
            client_kwargs.pop("provider", None)
            client_kwargs["base_url"] = self.config.base_url

        client = InferenceClient(**client_kwargs)
        model = self._plain_model_id(self.config.model)

        call_kwargs: Dict[str, Any] = {"model": model, "messages": messages}
        if params.get("max_tokens") is not None:
            call_kwargs["max_tokens"] = int(params["max_tokens"])
        if params.get("temperature") is not None:
            # The router rejects temperature=0; it wants the flag off instead. Clamping
            # to a very small positive value keeps "as deterministic as possible"
            # expressible without a special case at every call site.
            temp = float(params["temperature"])
            call_kwargs["temperature"] = temp if temp > 0 else 0.01
        if params.get("top_p") is not None:
            call_kwargs["top_p"] = float(params["top_p"])

        raw = self._chat(client, call_kwargs)

        try:
            choice = raw.choices[0]
            message = getattr(choice, "message", None) or choice["message"]
            text = getattr(message, "content", None)
            if text is None and isinstance(message, dict):
                text = message.get("content")
            finish = getattr(choice, "finish_reason", None)
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise LLMResponseError(
                "Hugging Face returned a response this client could not read.",
                detail=f"{type(exc).__name__}: {exc}",
                context={"provider": self.name, "model": model},
            ) from exc

        return LLMResponse(
            text=text or "",
            provider=self.name,
            model=model,
            finish_reason=finish,
            **self._usage_from(raw),
        )

    @staticmethod
    def _plain_model_id(model: str) -> str:
        """Drop a LiteLLM-style ``huggingface/`` prefix if one leaked into the config."""
        text = (model or "").strip()
        if text.startswith("huggingface/"):
            return text[len("huggingface/") :]
        return text

    @staticmethod
    def _chat(client: Any, call_kwargs: Dict[str, Any]) -> Any:
        """Prefer chat_completion; fall back to the OpenAI-shaped client.chat API."""
        try:
            return client.chat_completion(**call_kwargs)
        except TypeError:
            chat = getattr(client, "chat", None)
            completions = getattr(chat, "completions", None) if chat is not None else None
            create = getattr(completions, "create", None)
            if create is None:
                raise
            return create(**call_kwargs)


class LocalTransformersProvider(LLMProvider):
    """
    Open weights executed in this process with ``transformers``.

    The model is loaded on first use and cached on the instance. Importing
    ``transformers`` is deliberately deferred to that first call: on Windows the import
    performs a full filesystem sweep of installed distribution metadata that can take
    minutes, and paying that cost at module import made unrelated code paths - including
    a test fixture - look like they had hung.
    """

    name = "huggingface-local"
    label = "Hugging Face (local transformers)"
    required_packages = ("transformers",)
    needs_credential = False
    local = True
    drop_params = ("frequency_penalty", "presence_penalty", "top_p")

    _pipeline: Any = None
    _tokenizer: Any = None

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        from transformers import AutoTokenizer, pipeline

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model)
        kwargs: Dict[str, Any] = {
            "task": "text-generation",
            "model": self.config.model,
            "tokenizer": self._tokenizer,
        }
        device = self.config.extra.get("device")
        if device is not None:
            kwargs["device"] = device
        self._pipeline = pipeline(**kwargs)

    def _render_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Flatten chat turns into one prompt string.

        Instruction-tuned checkpoints ship a chat template; using it means the model sees
        the format it was actually trained on, which matters far more for small models
        than for large ones. Base models without a template get a role-prefixed
        transcript.
        """
        tokenizer = self._tokenizer
        template = getattr(tokenizer, "chat_template", None) if tokenizer else None
        if template:
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass  # fall through to the plain transcript
        lines = [f"{m['role'].upper()}: {m['content']}" for m in messages]
        lines.append("ASSISTANT:")
        return "\n\n".join(lines)

    def _invoke(self, messages: List[Dict[str, str]], params: Dict[str, Any]) -> LLMResponse:
        self._ensure_loaded()
        prompt = self._render_prompt(messages)

        max_new = params.get("max_tokens")
        max_new = 512 if max_new is None else int(max_new)
        temperature = params.get("temperature")
        temperature = 0.7 if temperature is None else float(temperature)

        result = self._pipeline(
            prompt,
            max_new_tokens=max_new,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            return_full_text=False,
        )

        text = ""
        if isinstance(result, list) and result:
            first = result[0]
            text = first.get("generated_text", "") if isinstance(first, dict) else str(first)

        return LLMResponse(text=text, provider=self.name, model=self.config.model)


# ============================================================================== stub
#: Answers the stub returns when it recognises a structured-output request. Kept as data
#: so the shapes stay in sync with what the pipeline parsers expect.
_STUB_AGENT_CONFIG = {
    "process": "sequential",
    "agents": [
        {
            "name": "research_specialist",
            "role": "Research Specialist",
            "goal": "Gather accurate, well-sourced information on the requested topic",
            "backstory": "An experienced analyst who is careful about provenance.",
            "tools": ["web_search"],
            "verbose": True,
            "allow_delegation": False,
        },
        {
            "name": "content_writer",
            "role": "Content Writer",
            "goal": "Turn research findings into a clear, structured summary",
            "backstory": "A technical writer who favours plain language.",
            "tools": ["writing_tool"],
            "verbose": True,
            "allow_delegation": False,
        },
    ],
    "tasks": [
        {
            "name": "research_task",
            "description": "Research the topic supplied by the user.",
            "tools": ["web_search"],
            "agent": "research_specialist",
            "expected_output": "A set of sourced findings.",
        },
        {
            "name": "writing_task",
            "description": "Write a summary from the research findings.",
            "tools": ["writing_tool"],
            "agent": "content_writer",
            "expected_output": "A structured written summary.",
        },
    ],
}


class MockProvider(LLMProvider):
    """
    A deterministic, offline stub.

    This exists so the pipeline, the API and the frontend can be exercised end to end
    with no credentials, no network and no cost - and so the project's own test suite can
    assert on pipeline behaviour rather than on model behaviour.

    Two rules keep it honest. It sets ``simulated=True``, which is propagated into
    :class:`LLMResponse`, stored on the run record and rendered in the UI, so stub output
    is never presented as a real model's work. And it is only ever selected when asked
    for explicitly - by name, or via ``MAGEN_LLM_PROVIDER=mock`` - never as a silent
    fallback when a real provider fails.
    """

    name = "mock"
    label = "Mock (offline stub)"
    needs_credential = False
    local = True
    simulated = True

    def _invoke(self, messages: List[Dict[str, str]], params: Dict[str, Any]) -> LLMResponse:
        prompt = "\n".join(m["content"] for m in messages)
        return LLMResponse(
            text=self._respond(prompt),
            provider=self.name,
            model=self.config.model,
            simulated=True,
            prompt_tokens=len(prompt.split()),
        )

    def _respond(self, prompt: str) -> str:
        lowered = prompt.lower()

        # Framework selection asks for a small, specific JSON object.
        if '"framework"' in lowered and '"confidence"' in lowered:
            return json.dumps(
                {
                    "framework": "crewai",
                    "reason": (
                        "Offline stub response: defaulting to a multi-agent crew, which "
                        "is the safest general-purpose choice."
                    ),
                    "confidence": 0.5,
                }
            )

        # Review asks for a score and issue list.
        if '"score"' in lowered and ("issues" in lowered or "required_changes" in lowered):
            return json.dumps(
                {
                    "score": 8.0,
                    "passed": True,
                    "summary": "Offline stub review: no model was consulted.",
                    "issues": [],
                    "required_changes": [],
                }
            )

        # Anything that asks for the agent-configuration shape.
        if '"agents"' in lowered:
            config = dict(_STUB_AGENT_CONFIG)
            if '"nodes"' in lowered:
                config = {
                    "agents": [
                        {
                            "name": "research_specialist",
                            "role": "Research Specialist",
                            "goal": "Gather and verify information",
                            "tools": ["web_search"],
                        }
                    ],
                    "nodes": [
                        {
                            "name": "research",
                            "description": "Research the query",
                            "agent": "research_specialist",
                        }
                    ],
                    "edges": [{"source": "research", "target": "END"}],
                }
            return json.dumps(config, indent=2)

        # Fall back to something obviously synthetic. Echoing a trimmed slice of the
        # prompt makes stubbed transcripts readable while leaving no doubt they are stubs.
        excerpt = re.sub(r"\s+", " ", prompt).strip()[:180]
        return (
            "[offline stub] No language model was called. "
            f"The request began: {excerpt!r}"
        )
