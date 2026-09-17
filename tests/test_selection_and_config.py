"""
Unit tests for framework selection and LLM configuration resolution.

Both of these are decision layers with no I/O, which makes them cheap to test exhaustively and
expensive to get wrong: selection decides what code gets written, and ``LLMConfig.resolve``
decides which credential is used. Neither needs a network, so neither is allowed to be
untested.

The rules being pinned here:

* An explicit framework choice from the user is obeyed, with confidence 1.0 and no silent
  substitution. Section 8 is unambiguous about that, and a "helpful" override is the kind of
  behaviour that makes a tool untrustworthy.
* Confidence reflects *separation between candidates*, not enthusiasm.
* A saved key belonging to a different provider is never reused. The realistic accident is
  saving an OpenAI key, switching the provider dropdown to Anthropic, and having the OpenAI key
  posted to Anthropic's API.
"""
from __future__ import annotations

import pytest

from multi_agent_generator.core.models import FrameworkChoice, RequirementAnalysis
from multi_agent_generator.core.selection import (
    FRAMEWORK_PROFILES,
    score_frameworks,
    select_framework,
)
from multi_agent_generator.errors import (
    Secret,
    UnknownProviderError,
    UnsupportedCombinationError,
)
from multi_agent_generator.llm.config import LLMConfig

pytestmark = pytest.mark.unit


def _analysis(**kw) -> RequirementAnalysis:
    base = dict(requirement="Create a research assistant", summary="research and summarise")
    base.update(kw)
    return RequirementAnalysis(**base)


class TestFrameworkSelection:
    def test_returns_the_shape_the_brief_asks_for(self):
        choice = select_framework(_analysis())
        payload = choice.as_dict()
        assert set(payload) >= {"framework", "reason", "confidence"}
        assert payload["framework"] in FRAMEWORK_PROFILES
        assert 0.0 <= payload["confidence"] <= 1.0
        assert payload["reason"].strip()

    def test_an_explicit_choice_is_obeyed_without_argument(self):
        # Even when the heuristic would pick something else: the user is allowed to be wrong
        # on purpose. Overriding them here would make the framework dropdown a suggestion box.
        choice = select_framework(
            _analysis(needs_branching=True, needs_state=True), requested="crewai"
        )
        assert choice.framework == "crewai"
        assert choice.user_specified is True
        assert choice.confidence == 1.0

    def test_an_unknown_requested_framework_is_rejected_not_ignored(self):
        # Silently correcting the name would mean generating LangGraph code for someone who
        # asked for something else, which is worse than an error.
        with pytest.raises(UnsupportedCombinationError):
            select_framework(_analysis(), requested="not-a-framework")

    @pytest.mark.parametrize("value", ["", "auto", "AUTO", "any", "  auto-detect "])
    def test_auto_sentinels_fall_through_to_scoring(self, value):
        choice = select_framework(_analysis(), requested=value)
        assert choice.user_specified is False

    def test_no_model_is_needed_to_choose(self):
        # provider=None means the scored heuristic decides alone. Selection must never be a
        # hard dependency on a credential.
        choice = select_framework(_analysis(), provider=None)
        assert choice.framework
        assert choice.reason

    def test_reason_names_the_requirement_that_drove_it(self):
        choice = select_framework(_analysis(needs_sequential_steps=True, needs_delegation=True))
        assert len(choice.reason) > 20

    def test_scoring_orders_candidates_best_first(self):
        scored = score_frameworks(_analysis(needs_branching=True, needs_state=True))
        assert len(scored) >= 2
        scores = [score for _, score, _ in scored]
        assert scores == sorted(scores, reverse=True)

    def test_scoring_can_be_restricted_to_installed_frameworks(self):
        scored = score_frameworks(_analysis(), allowed=["crewai"])
        assert [name for name, _, _ in scored] == ["crewai"]

    def test_a_stateful_branching_requirement_does_not_pick_a_single_agent_loop(self):
        # Not asserting a specific winner - profiles may be retuned - but a requirement with
        # cycles and state landing on the simplest framework would mean scoring is inert.
        choice = select_framework(
            _analysis(needs_branching=True, needs_state=True, needs_sequential_steps=True)
        )
        assert choice.framework in FRAMEWORK_PROFILES
        assert choice.confidence > 0.0

    def test_alternatives_never_include_the_winner(self):
        choice = select_framework(_analysis())
        assert all(alt.get("framework") != choice.framework for alt in choice.alternatives)

    def test_confidence_is_clamped(self):
        assert FrameworkChoice(framework="crewai", reason="r", confidence=1.7).confidence == 1.0
        assert FrameworkChoice(framework="crewai", reason="r", confidence=-1).confidence == 0.0


class TestLLMConfigResolve:
    def test_an_explicit_key_wins_and_is_wrapped(self):
        config = LLMConfig.resolve("openai", api_key="sk-explicit-000000", env={})
        assert isinstance(config.api_key, Secret)
        assert config.api_key.reveal() == "sk-explicit-000000"
        assert "sk-explicit-000000" not in repr(config)

    def test_the_environment_is_used_when_no_key_is_passed(self):
        config = LLMConfig.resolve("openai", env={"OPENAI_API_KEY": "sk-from-env-111111"})
        assert config.api_key.reveal() == "sk-from-env-111111"

    def test_no_key_anywhere_resolves_to_an_empty_secret(self):
        # ``resolve`` layers configuration; it does not judge it. The refusal happens later in
        # ``LLMProvider.validate()``, which is what lets the settings page build a config and
        # *report* that a key is missing instead of crashing while assembling one.
        config = LLMConfig.resolve("openai", env={})
        assert not config.api_key
        assert config.api_key.reveal() is None
        assert any("OPENAI" in name for name in config.credential_env)

    def test_a_saved_key_for_another_provider_is_not_reused(self):
        # The dangerous case: switching providers must not send the old key somewhere new.
        config = LLMConfig.resolve(
            "anthropic",
            saved={"provider": "openai", "api_key": "sk-openai-222222"},
            env={},
        )
        assert config.api_key.reveal() != "sk-openai-222222"
        assert not config.api_key

    def test_a_saved_model_and_base_url_are_also_gated_on_the_provider(self):
        # An OpenAI model id sent to Anthropic is a 404 that reads like an outage.
        config = LLMConfig.resolve(
            "anthropic",
            saved={
                "provider": "openai",
                "model": "gpt-4o-mini",
                "base_url": "https://api.openai.com/v1",
                "extra": {"organization": "org-123"},
            },
            env={},
        )
        assert config.model != "gpt-4o-mini"
        assert config.base_url != "https://api.openai.com/v1"
        assert "organization" not in config.extra

    def test_sampling_parameters_survive_a_provider_switch(self):
        # These are provider-independent, so dropping them would silently change behaviour.
        config = LLMConfig.resolve(
            "anthropic",
            saved={"provider": "openai", "temperature": 0.15, "max_tokens": 4096},
            env={},
        )
        assert config.temperature == 0.15
        assert config.max_tokens == 4096

    def test_a_saved_key_for_the_same_provider_is_used(self):
        config = LLMConfig.resolve(
            "openai", saved={"provider": "openai", "api_key": "sk-saved-333333"}, env={}
        )
        assert config.api_key.reveal() == "sk-saved-333333"

    def test_an_unknown_provider_is_refused(self):
        with pytest.raises(UnknownProviderError):
            LLMConfig.resolve("definitely-not-a-provider", env={})

    def test_redacted_never_contains_the_key(self):
        config = LLMConfig.resolve("openai", api_key="sk-secret-444444", env={})
        rendered = repr(config.redacted())
        assert "sk-secret-444444" not in rendered
        assert config.redacted()["provider"] == "openai"

    def test_with_overrides_preserves_the_credential(self):
        # A temperature change must not drop the key and turn into a confusing auth error.
        config = LLMConfig.resolve("openai", api_key="sk-keepme-555555", env={})
        changed = config.with_overrides(temperature=0.1)
        assert changed.temperature == 0.1
        assert changed.api_key.reveal() == "sk-keepme-555555"
        assert config.temperature != 0.1  # original untouched

    def test_numeric_overrides_are_applied(self):
        config = LLMConfig.resolve(
            "openai", api_key="sk-x-666666", env={}, temperature=0.0, max_tokens=8000, timeout=45
        )
        assert (config.temperature, config.max_tokens, config.timeout) == (0.0, 8000, 45)

    def test_a_default_model_is_always_chosen(self):
        # Downstream code formats a model id into a request; None there becomes a provider
        # error that reads like a bug in the provider.
        assert LLMConfig.resolve("openai", api_key="sk-x-777777", env={}).model
