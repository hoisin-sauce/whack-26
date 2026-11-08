"""WI-3: the research -> factors -> context -> critic citation graph.

One real mock/offline pipeline run is shared by every test.  The run seeds a
single credible catalog paper so the offline research stage has a record to
rank; the scripted mock LLM then cites that record and maps one factor to the
dataset's ``x`` column.  Everything downstream is the production pipeline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from integration_fixtures import (
    TARGET,
    make_dataset,
    make_settings,
    read_json,
    run_mock_pipeline,
)
from whack26.critic.corpus import Corpus, load_corpus
from whack26.embeddings import build_embedding_client
from whack26.factors import Factor
from whack26.hypotheses.context import build_context_pack, render_context_pack
from whack26.ids import factor_id
from whack26.io import read_dataset
from whack26.llm import MockLLMClient
from whack26.profile import DatasetProfile, build_profile, build_summary
from whack26.research.scrape.models import Document
from whack26.run import RunOutcome
from whack26.store import FieldStore

pytestmark = pytest.mark.integration

EVIDENCE_ID = "mock:evidence-1"
MAPPED_FACTOR_NAME = "mock income factor"
CONTEXT_FACTOR_NAME = "mock context factor"
MAPPED_COLUMN = "x"
ROWS = 80


@dataclass(frozen=True)
class PipelineBundle:
    tmp_path: Path
    settings: Any
    outcome: RunOutcome
    dataset: Path
    profile: DatasetProfile
    summary: str
    corpus: Corpus


class ClosedGraphMockLLM(MockLLMClient):
    """Deterministic fixtures that cite the one seeded research record."""

    def __init__(self, *, mapped_factor_id: str, evidence_id: str) -> None:
        super().__init__()
        self.mapped_factor_id = mapped_factor_id
        self.evidence_id = evidence_id

    def generate_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        purpose: str = "generic",
    ) -> dict[str, Any]:
        if purpose == "factors":
            return {
                "factors": [
                    {
                        "name": MAPPED_FACTOR_NAME,
                        "description": "A deterministic factor mapped to x.",
                        "direction": "positive",
                        "claim_type": "association",
                        "evidence_ids": [self.evidence_id],
                        "candidate_columns": [MAPPED_COLUMN],
                        "rationale": "mock fixture",
                    },
                    {
                        "name": CONTEXT_FACTOR_NAME,
                        "description": (
                            "A deterministic placeholder context factor with no "
                            "matching dataset column."
                        ),
                        "direction": "unclear",
                        "claim_type": "descriptive",
                        "evidence_ids": [self.evidence_id],
                        "candidate_columns": [],
                        "rationale": "mock fixture",
                    },
                ]
            }

        payload = super().generate_json(
            system=system,
            user=user,
            schema=schema,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            purpose=purpose,
        )
        if purpose.startswith("hypothesis_"):
            payload = dict(payload)
            payload.update(
                {
                    "rationale": (
                        "The mapped research factor is associated with the "
                        "outcome in this deterministic mock derivation."
                    ),
                    "required_columns": [MAPPED_COLUMN],
                    "optional_columns": ["group"],
                    "min_rows": 1,
                    "factors_used": [self.mapped_factor_id],
                    "source_ids": [self.evidence_id],
                }
            )
        return payload


def _seed_catalog_paper(
    settings: Any,
    *,
    summary_vector: list[float],
) -> None:
    """Store one credible paper so DB-first research has offline evidence."""
    document = Document(
        id=EVIDENCE_ID,
        kind="paper",
        source="openalex",
        source_id="W1",
        title="Income, housing prices, and spatial dependence",
        summary=(
            "Household income is associated with housing prices. The authors "
            "note that spatial autocorrelation is a limitation of the design."
        ),
        summary_embedding=list(summary_vector),
        embedding_model="mock",
        canonical_url="https://example.test/wi3-evidence-1",
        authors=["Ada Lovelace"],
        published_at="2024-01-01",
        credibility_score=0.9,
        credible=True,
        status="summarized",
    )
    with FieldStore(
        settings.fields_db,
        dims=settings.embed_dims,
        max_depth=settings.max_depth,
    ) as store:
        store.upsert_document(document)


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory: pytest.TempPathFactory) -> PipelineBundle:
    tmp_path = tmp_path_factory.mktemp("wi-3-research-hypotheses")
    dataset = make_dataset(
        tmp_path / "input.csv", kind="continuous", rows=ROWS, seed=0
    )
    settings = make_settings(
        tmp_path,
        field_pitfalls_max_queries_per_field=1,
        field_pitfalls_max_per_field=2,
        field_pitfalls_max_total=4,
    )

    loaded = read_dataset(dataset, target=TARGET)
    profile = build_profile(loaded, target=TARGET)
    summary = build_summary(profile, loaded.frame)

    # The classifier embeds this exact summary with the same mock embedding
    # client, so an identical vector makes the seeded paper a first-class
    # catalog hit (cosine 1.0) during the run.
    embeddings = build_embedding_client(settings)
    summary_vector = embeddings.embed([summary])[0]
    _seed_catalog_paper(settings, summary_vector=summary_vector)

    mapped_factor_id = factor_id(profile.dataset_id, MAPPED_FACTOR_NAME)
    llm = ClosedGraphMockLLM(
        mapped_factor_id=mapped_factor_id, evidence_id=EVIDENCE_ID
    )

    outcome = run_mock_pipeline(
        tmp_path,
        dataset,
        settings=settings,
        llm=llm,
        field_pitfalls=True,
    )

    corpus = load_corpus(settings.corpus_path)
    assert corpus is not None, "the statistical-pitfalls corpus must be loadable"
    return PipelineBundle(
        tmp_path=tmp_path,
        settings=settings,
        outcome=outcome,
        dataset=dataset,
        profile=profile,
        summary=summary,
        corpus=corpus,
    )


def _factor_from_section(section: Any) -> Factor:
    """Rebuild the context-pack input from an artifact ``FactorSection``."""
    return Factor(
        id=section.id,
        name=section.name,
        description=section.description,
        direction=section.direction,
        claim_type=section.claim_type,
        evidence_ids=list(section.evidence_ids),
        rationale=section.rationale,
        in_dataset=section.mapping.in_dataset,
        mapped_column=section.mapping.column,
        mapping_confidence=section.mapping.confidence,
    )


def _screen_target_name(screen: dict[str, Any]) -> str | None:
    for key in ("target", "target_summary"):
        value = screen.get(key)
        if isinstance(value, dict) and value.get("name"):
            return str(value["name"])
    return None


def test_research_records_and_factors_form_a_closed_citation_graph(
    pipeline: PipelineBundle,
) -> None:
    artifact = pipeline.outcome.artifact
    run_dir = pipeline.outcome.run_dir

    record_ids = [record.id for record in artifact.research.records]
    assert record_ids, "the mock/offline run must produce research records"
    assert len(record_ids) == len(set(record_ids)), "research record ids repeat"
    assert EVIDENCE_ID in record_ids

    factor_payload = read_json(run_dir / "factors.json")
    factor_json_ids = [factor["id"] for factor in factor_payload]
    artifact_factor_ids = [factor.id for factor in artifact.factors]
    assert factor_json_ids == artifact_factor_ids
    assert len(artifact_factor_ids) == len(set(artifact_factor_ids))
    assert MAPPED_FACTOR_NAME in {factor.name for factor in artifact.factors}

    scraped_ids: set[str] = set()
    scraping = artifact.research.semantic_scraping
    if scraping is not None:
        scraped_ids = {document.id for document in scraping.documents}
    evidence_universe = set(record_ids) | scraped_ids

    factors_json_by_id = {factor["id"]: factor for factor in factor_payload}
    dataset_columns = {column.name for column in artifact.dataset.columns}
    assert MAPPED_COLUMN in dataset_columns
    assert artifact.dataset.rows == ROWS

    for factor in artifact.factors:
        assert factor.evidence_ids, f"{factor.id} has no evidence ids"
        assert set(factor.evidence_ids) <= evidence_universe, (
            f"{factor.id} cites evidence outside research/scraped ids: "
            f"{sorted(set(factor.evidence_ids) - evidence_universe)}"
        )
        if factor.mapping.in_dataset:
            assert factor.mapping.column in dataset_columns
            confidence = factor.mapping.confidence
            assert confidence is not None
            assert 0.0 <= float(confidence) <= 1.0
            method = factors_json_by_id[factor.id].get("mapping_method")
            assert isinstance(method, str) and method.strip(), (
                f"{factor.id} is mapped without a mapping_method"
            )
        else:
            assert factor.mapping.in_dataset is False
            assert factor.mapping.column is None

    mapped = [factor for factor in artifact.factors if factor.mapping.in_dataset]
    unmapped = [
        factor for factor in artifact.factors if not factor.mapping.in_dataset
    ]
    assert mapped, "at least one factor must be mapped for this slice"
    assert unmapped, "the fixture includes one intentionally unmapped factor"
    assert all(factor.evidence_ids for factor in mapped + unmapped)


def test_hypotheses_only_reference_available_factors_and_columns(
    pipeline: PipelineBundle,
) -> None:
    artifact = pipeline.outcome.artifact
    run_dir = pipeline.outcome.run_dir
    dataset_columns = {column.name for column in artifact.dataset.columns}
    factor_ids = {factor.id for factor in artifact.factors}
    mapped_factor_ids = {
        factor.id for factor in artifact.factors if factor.mapping.in_dataset
    }

    record_ids = {record.id for record in artifact.research.records}
    scraped_ids: set[str] = set()
    scraping = artifact.research.semantic_scraping
    if scraping is not None:
        scraped_ids = {document.id for document in scraping.documents}
    evidence_ids = record_ids | scraped_ids

    assert artifact.hypotheses, "the run must produce hypotheses"
    raw_artifact = read_json(pipeline.outcome.artifact_path)
    raw_by_id = {
        section["id"]: section for section in raw_artifact["hypotheses"]
    }
    object_by_id = {
        section.id: section for section in artifact.hypotheses
    }
    assert set(raw_by_id) == set(object_by_id)

    for hypothesis in artifact.hypotheses:
        factors_used = set(hypothesis.factors_used)
        source_ids = set(hypothesis.source_ids)
        required = set(hypothesis.required_columns)
        optional = set(hypothesis.optional_columns)

        assert factors_used <= factor_ids
        assert factors_used <= mapped_factor_ids, (
            f"{hypothesis.id} uses unmapped factors: "
            f"{sorted(factors_used - mapped_factor_ids)}"
        )
        assert source_ids <= evidence_ids, (
            f"{hypothesis.id} cites unavailable evidence: "
            f"{sorted(source_ids - evidence_ids)}"
        )
        assert required | optional <= dataset_columns
        assert 1 <= hypothesis.min_rows <= artifact.dataset.rows

        code_path = run_dir / hypothesis.code_path
        assert code_path.is_file(), f"missing code file: {hypothesis.code_path}"
        predictions_path = hypothesis.execution.predictions_path
        assert predictions_path, f"{hypothesis.id} has no predictions path"
        assert (run_dir / predictions_path).is_file(), (
            f"missing predictions file: {predictions_path}"
        )

        raw = raw_by_id[hypothesis.id]
        original = object_by_id[hypothesis.id]
        assert raw["factors_used"] == original.factors_used
        assert raw["source_ids"] == original.source_ids
        assert raw["factors_used"] == hypothesis.factors_used
        assert raw["source_ids"] == hypothesis.source_ids

    assert any(hypothesis.factors_used for hypothesis in artifact.hypotheses)
    assert any(hypothesis.source_ids for hypothesis in artifact.hypotheses)


def test_context_pack_consumes_research_factors_and_pitfalls(
    pipeline: PipelineBundle,
) -> None:
    artifact = pipeline.outcome.artifact
    run_dir = pipeline.outcome.run_dir

    # Rebuild the run inputs from the run directory, then use the artifact id
    # for store lookups (the stored copy can have a different byte hash).
    loaded = read_dataset(run_dir / "dataset.csv", target=TARGET)
    profile = build_profile(loaded, target=TARGET)
    summary = build_summary(profile, loaded.frame)
    assert profile.rows == ROWS
    assert profile.target is not None and profile.target.name == TARGET

    factors = [_factor_from_section(section) for section in artifact.factors]
    screen_payload = artifact.screen.model_dump(mode="json")
    pitfalls = artifact.field_pitfalls.pitfalls if artifact.field_pitfalls else []
    assert pitfalls, "the field-pitfall stage must produce at least one pitfall"

    embeddings = build_embedding_client(pipeline.settings)
    with FieldStore(
        pipeline.settings.fields_db,
        dims=pipeline.settings.embed_dims,
        max_depth=pipeline.settings.max_depth,
    ) as store:
        stored = store.get_dataset(artifact.dataset.id)
        assert stored is not None, "the run dataset is missing from the field store"
        pack = build_context_pack(
            profile,
            summary=summary,
            store=store,
            dataset_id=artifact.dataset.id,
            factors=factors,
            research_models=artifact.research_models,
            model_candidates=pipeline.outcome.model_search,
            screen=screen_payload,
            corpus=pipeline.corpus,
            embeddings=embeddings,
            field_pitfalls=pitfalls,
        )

    rendered = render_context_pack(pack)

    mapped = [factor for factor in pack.factors if factor.mapped_column]
    assert mapped, "the context pack dropped every mapped factor"
    assert any(factor.id in rendered and factor.name in rendered for factor in mapped)

    assert _screen_target_name(pack.screen) == TARGET
    assert TARGET in rendered

    artifact_pitfall_ids = {pitfall.id for pitfall in pitfalls}
    pack_pitfall_ids = {
        str(pitfall.get("id")) for pitfall in pack.field_pitfalls
    }
    assert pack_pitfall_ids & artifact_pitfall_ids
    assert any(
        pitfall_id in rendered for pitfall_id in pack_pitfall_ids
    )

    assert pack.critic_chunks, "the context pack must ground itself in the corpus"
    assert any(chunk.startswith("### S") for chunk in pack.critic_chunks)

    assert str(pipeline.tmp_path) not in rendered
    assert str(pipeline.tmp_path) not in pack.summary
    assert str(pipeline.tmp_path) not in json.dumps(
        pack.model_dump(mode="json")
    )


def test_critic_results_are_grounded_in_the_corpus(
    pipeline: PipelineBundle,
) -> None:
    artifact = pipeline.outcome.artifact
    corpus = pipeline.corpus
    corpus_ids = {chunk.id for chunk in corpus.chunks}
    assert corpus_ids

    artifact_corpus_version = artifact.provenance.corpus_version
    assert artifact_corpus_version == corpus.version
    assert artifact_corpus_version

    field_pitfall_ids: set[str] = set()
    if artifact.field_pitfalls is not None:
        field_pitfall_ids = {
            pitfall.id for pitfall in artifact.field_pitfalls.pitfalls
        }
    citation_universe = corpus_ids | field_pitfall_ids

    allowed_verdicts = {"pass", "refine", "reject", "insufficient"}
    # Production uses ``not_applicable`` where the spec's shorthand says
    # "skip"; both describe a check that did not apply to this hypothesis.
    allowed_check_statuses = {"pass", "flag", "fail", "not_applicable"}
    required_checks = {"code.static", "research.traceability", "reproducibility"}
    banned_claims = re.compile(r"\b(?:true|validated)\b", re.IGNORECASE)

    assert len(artifact.hypotheses) == 3
    for hypothesis in artifact.hypotheses:
        critic = hypothesis.critic
        assert critic is not None, f"{hypothesis.id} has no critic result"
        assert critic.verdict in allowed_verdicts
        assert critic.corpus_version == artifact_corpus_version

        assert critic.retrieved_chunks, (
            f"{hypothesis.id} has no retrieved corpus chunks"
        )
        cited = set(critic.retrieved_chunks)
        assert cited <= citation_universe, (
            f"{hypothesis.id} cites unknown chunks: "
            f"{sorted(cited - citation_universe)}"
        )
        assert cited & corpus_ids, (
            f"{hypothesis.id} cites no actual corpus chunk"
        )

        assert critic.checks, f"{hypothesis.id} has no deterministic checks"
        check_ids = {check.id for check in critic.checks}
        assert required_checks <= check_ids
        assert {check.status for check in critic.checks} <= allowed_check_statuses

        assert not banned_claims.search(critic.summary), (
            f"{hypothesis.id} critic summary claims truth/validation: "
            f"{critic.summary!r}"
        )
