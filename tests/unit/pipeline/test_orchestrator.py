"""Unit tests for the five-stage pipeline orchestrator (P1-PIPE-01..03)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from transduce.backends.base import BackendCapabilities, BackendHealth, GenerationResult
from transduce.pipeline.orchestrator import (
    CompositionNotImplementedError,
    Orchestrator,
    VerificationFailedError,
)
from transduce.registry.spec import (
    BackendRequirements,
    ModeSpec,
    PreserveRule,
    VerifierProfile,
)
from transduce.registry.static import StaticRegistry
from transduce.verification.base import ScoreResult
from transduce.verification.pipeline import VerifierPipeline

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class ScriptedBackend:
    """Backend test double returning a queued sequence of generations."""

    responses: list[GenerationResult]
    name: str = "ollama"
    model: str = "qwen2.5:1.5b"
    capabilities: BackendCapabilities = field(default_factory=BackendCapabilities)
    prompts_seen: list[str] = field(default_factory=list)
    fail_after: int | None = None
    failure: BaseException | None = None

    async def generate(
        self, prompt: str, *, max_tokens: int, temperature: float
    ) -> GenerationResult:
        del max_tokens, temperature
        self.prompts_seen.append(prompt)
        if (
            self.fail_after is not None
            and self.failure is not None
            and len(self.prompts_seen) > self.fail_after
        ):
            raise self.failure
        if not self.responses:
            raise RuntimeError("ScriptedBackend exhausted")
        return self.responses.pop(0)

    async def health(self) -> BackendHealth:  # pragma: no cover — orchestrator never calls
        return BackendHealth(healthy=True)


@dataclass
class ScriptedScorer:
    """Scorer test double cycling through a queue of verdicts."""

    name: str
    queue: list[str]
    span: str | None = None
    invocations: int = 0

    def score(self, original: str, candidate: str) -> ScoreResult:
        del original, candidate
        self.invocations += 1
        verdict = self.queue.pop(0) if self.queue else "accept"
        return ScoreResult(
            name=self.name,
            value=1.0 if verdict == "accept" else 0.0,
            verdict=verdict,  # type: ignore[arg-type]
            rejection_reason=None if verdict == "accept" else f"{self.name} rejected",
            span=self.span if verdict == "reject" else None,
        )


def _spec(mode_id: str = "dejargon") -> ModeSpec:
    return ModeSpec(
        id=mode_id,
        version="0.1.0",
        description="x",
        prompt_template=(
            "Rewrite intensity={{ intensity }}"
            "{% if preserve %} preserve={{ preserve | join(',') }}{% endif %}"
            ": {{ input }}"
        ),
        preserve_defaults=(PreserveRule.ENTITIES,),
        backend_requirements=BackendRequirements(min_model_b=1.0),
        verifier_profile=VerifierProfile(),
    )


def _orchestrator(
    *,
    backend: ScriptedBackend,
    scorers: list[ScriptedScorer],
    default_max_retries: int = 3,
) -> Orchestrator:
    return Orchestrator(
        registry=StaticRegistry([_spec()]),
        backend=backend,
        verifier=VerifierPipeline(scorers),
        default_max_retries=default_max_retries,
    )


# ---------------------------------------------------------------------------
# Happy path and contract behaviour
# ---------------------------------------------------------------------------


async def test_orchestrator_happy_path_returns_accept_after_one_generation() -> None:
    backend = ScriptedBackend(
        responses=[GenerationResult(text="Reduced jargon.", tokens_in=10, tokens_out=4)]
    )
    scorers = [ScriptedScorer(name="cosine_similarity", queue=["accept"])]

    result = await _orchestrator(backend=backend, scorers=scorers).transform(
        text="We synergize verticals.",
        mode="dejargon",
        intensity=0.6,
        preserve=[],
        request_id="req-1",
    )

    assert result.transformed == "Reduced jargon."
    assert result.retries == 0
    assert result.scores.rejection_reason is None
    assert result.scores.preserved == {"entities": True, "numbers": True, "urls": True}
    assert result.cost.tokens_in_total == 10
    assert result.cost.tokens_out_total == 4
    assert result.cost.usd_total == pytest.approx(0.0)
    assert len(backend.prompts_seen) == 1


async def test_orchestrator_retry_increments_count_on_verifier_reject() -> None:
    backend = ScriptedBackend(
        responses=[
            GenerationResult(text="bad-1", tokens_in=8, tokens_out=2),
            GenerationResult(text="ok", tokens_in=8, tokens_out=2),
        ]
    )
    scorers = [ScriptedScorer(name="cosine_similarity", queue=["reject", "accept"], span="span-1")]

    result = await _orchestrator(backend=backend, scorers=scorers, default_max_retries=2).transform(
        text="hi",
        mode="dejargon",
        intensity=0.5,
        preserve=[],
        request_id="req-2",
    )

    assert result.retries == 1
    assert result.transformed == "ok"
    assert "scorer=cosine_similarity" in backend.prompts_seen[-1]
    assert "span='span-1'" in backend.prompts_seen[-1]


async def test_retry_failure_context_names_scorer() -> None:
    backend = ScriptedBackend(
        responses=[
            GenerationResult(text="bad", tokens_in=4, tokens_out=2),
            GenerationResult(text="ok", tokens_in=4, tokens_out=2),
        ]
    )
    scorers = [ScriptedScorer(name="negation_diff", queue=["reject", "accept"], span="did not")]

    await _orchestrator(backend=backend, scorers=scorers, default_max_retries=1).transform(
        text="hi",
        mode="dejargon",
        intensity=0.5,
        preserve=[],
        request_id="req-retry-scorer",
    )

    retry_prompt = backend.prompts_seen[-1]
    assert "scorer=negation_diff" in retry_prompt
    assert "do not add or remove negation cues" in retry_prompt


async def test_retry_failure_context_names_span() -> None:
    backend = ScriptedBackend(
        responses=[
            GenerationResult(text="bad", tokens_in=4, tokens_out=2),
            GenerationResult(text="ok", tokens_in=4, tokens_out=2),
        ]
    )
    scorers = [
        ScriptedScorer(
            name="entity_preservation",
            queue=["reject", "accept"],
            span="Acme Corp",
        )
    ]

    await _orchestrator(backend=backend, scorers=scorers, default_max_retries=1).transform(
        text="hi",
        mode="dejargon",
        intensity=0.5,
        preserve=[],
        request_id="req-retry-span",
    )

    retry_prompt = backend.prompts_seen[-1]
    assert "span='Acme Corp'" in retry_prompt
    assert "preserve every named entity from the source verbatim" in retry_prompt


async def test_orchestrator_max_retries_exhausted_raises_verification_failed() -> None:
    backend = ScriptedBackend(
        responses=[GenerationResult(text=str(i), tokens_in=1, tokens_out=1) for i in range(5)]
    )
    scorers = [
        ScriptedScorer(
            name="cosine_similarity",
            queue=["reject", "reject", "reject"],
            span="bad",
        )
    ]
    orchestrator = _orchestrator(backend=backend, scorers=scorers, default_max_retries=2)

    with pytest.raises(VerificationFailedError) as exc:
        await orchestrator.transform(
            text="hi",
            mode="dejargon",
            intensity=0.5,
            preserve=[],
            request_id="req-3",
        )

    assert exc.value.retries == 2
    assert exc.value.rejection_reason == "cosine_similarity rejected"
    assert exc.value.last_candidate == "2"


async def test_orchestrator_compose_chain_raises_not_implemented() -> None:
    backend = ScriptedBackend(responses=[])
    scorers = [ScriptedScorer(name="cosine_similarity", queue=["accept"])]
    orchestrator = _orchestrator(backend=backend, scorers=scorers)

    with pytest.raises(CompositionNotImplementedError, match="compose chains"):
        await orchestrator.transform(
            text="hi",
            mode=["dejargon", "register.casual"],
            intensity=0.5,
            preserve=[],
            request_id="req-4",
        )


async def test_orchestrator_retry_zero_disables_retry() -> None:
    backend = ScriptedBackend(responses=[GenerationResult(text="bad", tokens_in=1, tokens_out=1)])
    scorers = [ScriptedScorer(name="cosine_similarity", queue=["reject"])]
    orchestrator = _orchestrator(backend=backend, scorers=scorers)

    with pytest.raises(VerificationFailedError) as exc:
        await orchestrator.transform(
            text="hi",
            mode="dejargon",
            intensity=0.5,
            preserve=[],
            max_retries=0,
            request_id="req-5",
        )

    assert exc.value.retries == 0
    assert len(backend.prompts_seen) == 1


async def test_orchestrator_max_retries_above_ceiling_rejected() -> None:
    backend = ScriptedBackend(responses=[GenerationResult(text="ok", tokens_in=1, tokens_out=1)])
    scorers = [ScriptedScorer(name="cosine_similarity", queue=["accept"])]
    orchestrator = _orchestrator(backend=backend, scorers=scorers)

    with pytest.raises(ValueError, match="max_retries"):
        await orchestrator.transform(
            text="hi",
            mode="dejargon",
            intensity=0.5,
            preserve=[],
            max_retries=6,
            request_id="req-6",
        )


async def test_orchestrator_cost_aggregates_across_attempts() -> None:
    backend = ScriptedBackend(
        responses=[
            GenerationResult(text="bad", tokens_in=4, tokens_out=2),
            GenerationResult(text="ok", tokens_in=5, tokens_out=3),
        ]
    )
    scorers = [ScriptedScorer(name="cosine_similarity", queue=["reject", "accept"], span="x")]
    orchestrator = _orchestrator(backend=backend, scorers=scorers, default_max_retries=1)

    result = await orchestrator.transform(
        text="hi",
        mode="dejargon",
        intensity=0.5,
        preserve=[],
        request_id="req-7",
    )

    assert result.cost.tokens_in_total == 9
    assert result.cost.tokens_out_total == 5
    assert [a.attempt for a in result.cost.by_attempt] == [1, 2]


async def test_orchestrator_uses_mode_preserve_defaults_when_request_omits() -> None:
    backend = ScriptedBackend(responses=[GenerationResult(text="ok", tokens_in=1, tokens_out=1)])
    scorers = [ScriptedScorer(name="cosine_similarity", queue=["accept"])]
    orchestrator = _orchestrator(backend=backend, scorers=scorers)

    await orchestrator.transform(
        text="hi",
        mode="dejargon",
        intensity=0.5,
        preserve=[],
        request_id="req-8",
    )

    assert "entities" in backend.prompts_seen[0]


async def test_orchestrator_invalid_default_max_retries_rejected_at_construction() -> None:
    backend = ScriptedBackend(responses=[])
    scorers = [ScriptedScorer(name="cosine_similarity", queue=[])]

    with pytest.raises(ValueError, match="default_max_retries"):
        Orchestrator(
            registry=StaticRegistry([_spec()]),
            backend=backend,
            verifier=VerifierPipeline(scorers),
            default_max_retries=10,
        )


async def test_orchestrator_invalid_max_tokens_floor_rejected_at_construction() -> None:
    backend = ScriptedBackend(responses=[])
    scorers = [ScriptedScorer(name="cosine_similarity", queue=[])]

    with pytest.raises(ValueError, match="max_tokens_floor"):
        Orchestrator(
            registry=StaticRegistry([_spec()]),
            backend=backend,
            verifier=VerifierPipeline(scorers),
            max_tokens_floor=0,
        )


async def test_orchestrator_invalid_max_tokens_ratio_rejected_at_construction() -> None:
    backend = ScriptedBackend(responses=[])
    scorers = [ScriptedScorer(name="cosine_similarity", queue=[])]

    with pytest.raises(ValueError, match="max_tokens_ratio"):
        Orchestrator(
            registry=StaticRegistry([_spec()]),
            backend=backend,
            verifier=VerifierPipeline(scorers),
            max_tokens_ratio=0.0,
        )


async def test_orchestrator_diff_attached_on_accept() -> None:
    backend = ScriptedBackend(
        responses=[GenerationResult(text="hello earth", tokens_in=2, tokens_out=2)]
    )
    scorers = [ScriptedScorer(name="cosine_similarity", queue=["accept"])]
    orchestrator = _orchestrator(backend=backend, scorers=scorers)

    result = await orchestrator.transform(
        text="hello world",
        mode="dejargon",
        intensity=0.5,
        preserve=[],
        request_id="req-9",
    )

    ops = [d.op for d in result.diff]
    assert ops == ["equal", "delete", "insert"]
