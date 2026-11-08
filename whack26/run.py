"""End-to-end run orchestration: classify through artifact and preview."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

from . import __version__
from .artifact import Artifact, RunInfo, build_artifact, validate_artifact, write_artifact
from .classify import Classifier
from .common import redact_secrets, utc_now_iso
from .config import Settings
from .critic import CorpusError, CriticResult, load_corpus, run_critic
from .embeddings import build_embedding_client
from .factors import FactorExtraction, extract_factors
from .hypotheses import HypothesisContext, HypothesisGenerator, HypothesisSet
from .hypotheses.context import build_context_pack
from .hypotheses.variants import VariantReport, build_variants
from .ids import run_id
from .io import read_dataset
from .llm import build_llm_client
from .model_search.pipeline import run_model_search
from .preview import render_scrape_preview, write_preview
from .profile import build_profile, build_summary
from .research.pipeline import ResearchResult, refine_after_research, run_research
from .research.scrape.models import ScrapeResult
from .research.to_model import extract_research_models
from .screen import build_screen
from .store import FieldStore
from .validation.collection import (
    CollectionMetadata,
    collection_to_json,
    parse_collection_json,
    redact_collection,
)
from .validation.generation import GenerationValidator
from .validation.generation_models import GenerationValidationResult
from .validation.models import (
    ValidationConfig,
    ValidationReport,
    error_report,
    summary_counts,
)


class RunError(RuntimeError):
    """A run failure that should be reported without a traceback."""


@dataclass
class RunOutcome:
    run_dir: Path
    artifact: Artifact
    artifact_path: Path
    preview_path: Path
    scrape: ScrapeResult | None = None
    scrape_md_path: Path | None = None
    validations: dict[str, ValidationReport] = field(default_factory=dict)
    generation_validations: dict[str, GenerationValidationResult] = field(
        default_factory=dict
    )
    model_search: Any | None = None
    variants: Any | None = None


class _RunLog:
    def __init__(self, path: Path, sink: Callable[[str], None] | None) -> None:
        self.path = path
        self.sink = sink
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        line = f"{utc_now_iso()} {redact_secrets(message)}"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        if self.sink is not None:
            self.sink(line)


def _model_names(settings: Settings) -> dict[str, str]:
    if settings.llm_provider == "gemini":
        llm = settings.gemini_model
    else:
        llm = settings.llm_provider
    if settings.embed_provider == "gemini":
        embedding = f"{settings.gemini_embed_model}@{settings.embed_dims}"
    else:
        embedding = f"{settings.embed_provider}@{settings.embed_dims}"
    return {"llm": llm, "embedding": embedding}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _research_payload(research: ResearchResult) -> dict[str, Any]:
    return {
        "queries": [query.model_dump(mode="json") for query in research.queries],
        "records": [record.model_dump(mode="json") for record in research.records],
        "keywords": list(research.keywords),
        "sources_used": list(research.sources_used),
        "limited_mode": research.limited_mode,
        "offline": research.offline,
        "cache": dict(research.cache),
        "warnings": list(research.warnings),
    }


def _warning_list(*groups: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for group in groups:
        for raw in group:
            text = str(raw).strip()
            if text and text not in seen:
                seen.add(text)
                result.append(text)
    return result


def _scraped_evidence(scrape: ScrapeResult | None) -> list[Any]:
    """Convert credible scraped documents into factor/hypothesis evidence rows."""
    if scrape is None:
        return []
    evidence: list[Any] = []
    for document in scrape.documents:
        if not document.credible:
            continue
        year = None
        if document.published_at:
            match = re.match(r"(\d{4})", str(document.published_at))
            if match:
                year = int(match.group(1))
        evidence.append(
            SimpleNamespace(
                id=document.id,
                source=document.source,
                year=year,
                title=document.title,
                abstract=document.summary,
                url=document.canonical_url,
                doi=document.metadata.get("doi"),
            )
        )
    return evidence


def _write_dataset_copy(frame: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, lineterminator="\n")


def run_pipeline(
    *,
    input_path: Path,
    description: str | None = None,
    target: str | None = None,
    seed: int = 0,
    mock: bool = False,
    offline: bool = False,
    refresh: bool = False,
    scrape: bool | None = None,
    scrape_fetch: bool | None = None,
    model_search: bool | None = None,
    variants: bool | None = None,
    validate_generation: bool | None = None,
    collection: Any | None = None,
    validate: bool = False,
    validation_config: ValidationConfig | None = None,
    csv_sep: str = ",",
    out_dir: Path | None = None,
    store_path: Path | None = None,
    settings: Settings | None = None,
    llm: Any | None = None,
    embeddings: Any | None = None,
    log: Callable[[str], None] | None = None,
) -> RunOutcome:
    """Run every stage and write a self-contained run directory."""
    started_at = utc_now_iso()
    settings = settings or Settings.from_env(
        mock=mock, offline=True if offline else None
    )
    if scrape is not None:
        settings = settings.with_overrides(scrape_enabled=bool(scrape))
    if model_search is not None:
        settings = settings.with_overrides(model_search_enabled=bool(model_search))
    if variants is not None:
        settings = settings.with_overrides(variants_enabled=bool(variants))
    if validate_generation is not None:
        settings = settings.with_overrides(
            generation_validation_enabled=bool(validate_generation)
        )
    if scrape_fetch is not None:
        settings = settings.with_overrides(scrape_fetch_enabled=bool(scrape_fetch))
    owns_llm = llm is None
    owns_embeddings = embeddings is None
    llm = llm if llm is not None else build_llm_client(settings)
    embeddings = (
        embeddings if embeddings is not None else build_embedding_client(settings)
    )
    scrape_result: ScrapeResult | None = None
    model_search_result: Any | None = None
    variant_set: Any | None = None
    variant_frames: dict[str, Any] = {}
    research_models: list[Any] = []
    context_pack: Any | None = None
    generation_validations: dict[str, GenerationValidationResult] = {}
    collection_metadata: CollectionMetadata | None = None
    try:
        loaded = read_dataset(input_path, target=target, csv_sep=csv_sep)
        profile = build_profile(
            loaded,
            description=description,
            target=target,
            llm=llm,
        )
        if collection is not None:
            try:
                if isinstance(collection, CollectionMetadata):
                    collection_metadata = redact_collection(collection)
                else:
                    collection_metadata = redact_collection(
                        parse_collection_json(collection)
                    )
            except Exception as exc:  # noqa: BLE001 - advisory metadata
                collection_metadata = None
        identifier = run_id(profile.dataset_id)
        if out_dir is None:
            run_dir = settings.runs_dir / identifier
        else:
            run_dir = Path(out_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        logger = _RunLog(run_dir / "logs" / "run.log", log)
        if collection is not None and collection_metadata is None:
            logger("collection_parse_failed; continuing without metadata")
        logger(
            f"run_start dataset={profile.dataset_id} rows={profile.rows} "
            f"columns={len(profile.columns)} mode={'mock' if settings.mock else 'live'}"
        )
        _write_dataset_copy(loaded.frame, run_dir / "dataset.csv")
        summary = build_summary(profile, loaded.frame)

        try:
            with FieldStore(
                store_path or settings.fields_db,
                dims=settings.embed_dims,
                max_depth=settings.max_depth,
            ) as store:
                classifier = Classifier(
                    llm=llm,
                    embeddings=embeddings,
                    store=store,
                    settings=settings,
                )
                path = classifier.classify(profile, loaded.frame)
                logger(
                    "classify path="
                    + " > ".join(path.names)
                    + f" confidence={path.confidence:.3f}"
                )
                if collection_metadata is not None:
                    try:
                        store.set_collection(
                            profile.dataset_id,
                            collection_to_json(collection_metadata),
                        )
                        logger(
                            "collection design=" + collection_metadata.design
                        )
                    except Exception as exc:  # noqa: BLE001 - advisory
                        logger(
                            f"collection_persist_failed "
                            f"{type(exc).__name__}: {exc}"
                        )
                research = run_research(
                    profile=profile,
                    summary=summary,
                    path_names=path.names,
                    llm=llm,
                    embeddings=embeddings,
                    settings=settings,
                    cache_path=settings.research_db,
                    offline=True if offline else None,
                    refresh=refresh,
                )
                logger(
                    f"research queries={len(research.queries)} "
                    f"records={len(research.records)}"
                )
                try:
                    from .research.search_sessions import record_search_session

                    result_counts: dict[str, int] = {}
                    for record in research.records:
                        key = str(record.query or "")
                        result_counts[key] = result_counts.get(key, 0) + 1
                    for query in research.queries:
                        record_search_session(
                            store,
                            query=query.query,
                            source=query.source,
                            session_id=identifier,
                            parameters={"rationale": query.rationale},
                            result_count=result_counts.get(query.query, 0),
                            dataset_ids=[profile.dataset_id],
                        )
                    logger(
                        f"search_documents={len(research.queries)}"
                    )
                except Exception as exc:  # noqa: BLE001 - advisory
                    research.warnings.append(
                        f"search_sessions_failed: {type(exc).__name__}: {exc}"
                    )
                    logger(
                        f"search_sessions_failed {type(exc).__name__}: {exc}"
                    )
                try:
                    path = refine_after_research(
                        classifier=classifier,
                        profile=profile,
                        frame=loaded.frame,
                        path=path,
                        result=research,
                    )
                except Exception as exc:  # noqa: BLE001 - keep the original path
                    path.warnings.append(
                        "refinement_failed: "
                        f"{type(exc).__name__}: {exc}; kept the original path"
                    )
                    logger(f"refine_failed {type(exc).__name__}")
                if path.refined_after_research:
                    logger(
                        "refine path="
                        + " > ".join(path.names)
                        + f" confidence={path.confidence:.3f}"
                    )
                if settings.variants_enabled:
                    try:
                        variant_set, variant_frames = build_variants(
                            profile, loaded.frame, seed=seed
                        )
                        variant_set = VariantReport(
                            variants=variant_set,
                            seed=seed,
                            target_column=(
                                profile.target.name
                                if profile.target is not None
                                else None
                            ),
                        )
                        _write_json(
                            run_dir / "variants.json",
                            variant_set.model_dump(mode="json"),
                        )
                        logger(
                            "variants "
                            + ", ".join(sorted(variant_frames))
                        )
                    except Exception as exc:  # noqa: BLE001 - advisory
                        research.warnings.append(
                            f"variants_failed: {type(exc).__name__}: {exc}"
                        )
                        logger(f"variants_failed {type(exc).__name__}: {exc}")
                if settings.scrape_enabled:
                    try:
                        from .research.scrape.pipeline import run_semantic_scraping

                        scrape_result = run_semantic_scraping(
                            seeds=research.records,
                            target=summary,
                            store=store,
                            llm=llm,
                            embeddings=embeddings,
                            settings=settings,
                        )
                        logger(
                            f"scrape documents={len(scrape_result.documents)} "
                            f"links={len(scrape_result.links)} "
                            f"comparisons={len(scrape_result.comparisons)} "
                            f"critiques={len(scrape_result.critiques)}"
                        )
                        for warning in scrape_result.warnings:
                            logger(f"scrape_warning {warning}")
                    except Exception as exc:  # noqa: BLE001 - keep the run alive
                        research.warnings.append(
                            f"scrape_failed: {type(exc).__name__}: {exc}"
                        )
                        logger(f"scrape_failed {type(exc).__name__}: {exc}")
        except Exception as exc:
            logger(f"run_failed stage=classify_or_research {type(exc).__name__}: {exc}")
            raise

        evidence_records = _scraped_evidence(scrape_result) + list(research.records)
        factors: FactorExtraction = extract_factors(
            profile=profile,
            summary=summary,
            path_names=path.names,
            records=evidence_records,
            llm=llm,
            embeddings=embeddings,
            settings=settings,
        )
        logger(
            f"factors total={len(factors.factors)} "
            f"mapped={sum(1 for factor in factors.factors if factor.in_dataset)}"
        )
        screen = build_screen(profile, loaded.frame)
        logger(f"screen warnings={len(screen.warnings)}")

        corpus = None
        try:
            corpus = load_corpus(settings.corpus_path)
        except CorpusError as exc:
            logger(f"corpus_error {type(exc).__name__}: {exc}")

        if settings.model_search_enabled:
            try:
                model_search_result = run_model_search(
                    profile, loaded.frame, settings=settings
                )
                _write_json(
                    run_dir / "model_search.json",
                    model_search_result.model_dump(mode="json"),
                )
                logger(
                    "model_search candidates="
                    f"{len(model_search_result.candidates)} "
                    f"selected="
                    f"{getattr(getattr(model_search_result.selected, 'spec', None), 'family', None)}"
                )
                for warning in model_search_result.warnings:
                    logger(f"model_search_warning {warning}")
            except Exception as exc:  # noqa: BLE001 - advisory
                research.warnings.append(
                    f"model_search_failed: {type(exc).__name__}: {exc}"
                )
                logger(f"model_search_failed {type(exc).__name__}: {exc}")

        if settings.model_search_enabled or settings.variants_enabled:
            try:
                from .research.scrape.pipeline import _record_to_document

                documents: list[Any] = []
                if scrape_result is not None:
                    documents.extend(
                        document
                        for document in scrape_result.documents
                        if document.kind in ("paper", "web")
                    )
                documents.extend(
                    _record_to_document(record)
                    for record in research.records[:10]
                )
                documents = [
                    document
                    for document in documents
                    if document.kind in ("paper", "web") and document.credible
                ][:5]
                with FieldStore(
                    store_path or settings.fields_db,
                    dims=settings.embed_dims,
                    max_depth=settings.max_depth,
                ) as store:
                    if documents:
                        research_models = extract_research_models(
                            documents,
                            factors.factors,
                            profile,
                            llm,
                            embeddings,
                            settings,
                            store=store,
                        )
                    context_pack = build_context_pack(
                        profile,
                        summary,
                        store=store,
                        dataset_id=profile.dataset_id,
                        field_path=path.fields,
                        factors=factors.factors,
                        research_models=research_models,
                        model_candidates=model_search_result,
                        screen=screen,
                        corpus=corpus,
                    )
                _write_json(
                    run_dir / "research_models.json",
                    [item.model_dump(mode="json") for item in research_models],
                )
                logger(
                    f"research_models={len(research_models)} "
                    f"context_chars={len(context_pack.summary)}"
                )
            except Exception as exc:  # noqa: BLE001 - advisory
                research.warnings.append(
                    f"context_failed: {type(exc).__name__}: {exc}"
                )
                logger(f"context_failed {type(exc).__name__}: {exc}")

        context = HypothesisContext(
            profile=profile,
            summary=summary,
            path_names=path.names,
            factors=factors.factors,
            records=evidence_records,
            screen=screen,
            model_search=model_search_result,
            research_models=research_models,
            variant_report=variant_set,
            context_pack=context_pack,
            collection=collection_metadata,
        )
        generator = HypothesisGenerator(
            context=context,
            llm=llm,
            embeddings=embeddings,
            run_dir=run_dir,
            dataset_path=run_dir / "dataset.csv",
            seed=seed,
            settings=settings,
        )
        if settings.agent_loop_enabled:
            from .hypotheses.agent_loop import ModelAgentLoop

            def _propose(request: Any) -> Any:
                feedback_text = None
                feedback = request.feedback
                if feedback is not None:
                    pieces = [
                        *list(getattr(feedback, "static_errors", []) or []),
                        *list(getattr(feedback, "summary", "") and [feedback.summary] or []),
                    ]
                    if getattr(feedback, "execution_error", None):
                        pieces.append(str(feedback.execution_error))
                    for diagnostic in getattr(feedback, "diagnostics", []) or []:
                        message = getattr(diagnostic, "message", "")
                        if message:
                            pieces.append(str(message))
                    feedback_text = "\n".join(str(piece) for piece in pieces if piece)
                return generator._ask(
                    request.family,
                    feedback=feedback_text,
                    simpler=request.phase == "replace",
                )

            loop = ModelAgentLoop(
                generator=_propose,
                context=context,
                settings=settings,
                run_dir=run_dir,
                dataset_path=run_dir / "dataset.csv",
                seed=seed,
                context_pack=context_pack,
            )
            hypotheses: HypothesisSet = loop.run()
        else:
            hypotheses = generator.generate()
        logger(
            "hypotheses "
            + ", ".join(
                f"{item.method_family}={item.execution.status}"
                for item in hypotheses.hypotheses
            )
        )

        if settings.generation_validation_enabled:
            validator = GenerationValidator(
                llm=llm,
                embeddings=embeddings,
                settings=settings,
                corpus=corpus,
            )
            for hypothesis in hypotheses.hypotheses:
                try:
                    result = validator.validate(
                        hypothesis,
                        profile=profile,
                        collection=collection_metadata,
                        screen=screen,
                        model_search=model_search_result,
                        research_models=research_models,
                        variant_report=variant_set,
                        code_hash=hypothesis.code_sha256,
                        data_hash=profile.sha256,
                        seed=seed,
                    )
                except Exception as exc:  # noqa: BLE001 - advisory
                    logger(
                        f"generation_validation_failed {hypothesis.id} "
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue
                generation_validations[hypothesis.id] = result
                logger(
                    f"generation_validation {hypothesis.id} "
                    f"verdict={result.verdict} issues={len(result.issues)}"
                )
            if generation_validations:
                _write_json(
                    run_dir / "generation_validation.json",
                    {
                        "schema_version": "1.0.0",
                        "run_id": identifier,
                        "generated_at": utc_now_iso(),
                        "results": [
                            result.model_dump(mode="json")
                            for result in generation_validations.values()
                        ],
                    },
                )

        critics: list[CriticResult] = []
        for hypothesis in hypotheses.hypotheses:
            critic = run_critic(
                hypothesis,
                profile=profile,
                summary=summary,
                field_path=path.fields,
                factors=factors.factors,
                screen=screen,
                research=evidence_records,
                llm=llm,
                embeddings=embeddings,
                settings=settings,
                corpus=corpus,
            )
            critics.append(critic)
            logger(f"critic {hypothesis.id} verdict={critic.verdict}")

        finished_at = utc_now_iso()
        warnings = _warning_list(
            profile.warnings,
            path.warnings,
            research.warnings,
            factors.warnings,
            screen.warnings,
            hypotheses.warnings,
            scrape_result.warnings if scrape_result is not None else [],
            [
                f"critic_warning: {critic.hypothesis_id}: {warning}"
                for critic in critics
                for warning in critic.warnings
            ],
        )
        validations: dict[str, ValidationReport] = {}
        if validate:
            from .validation.pipeline import validate_hypothesis

            config = validation_config or ValidationConfig()
            provisional = build_artifact(
                run=RunInfo(
                    run_id=identifier,
                    started_at=started_at,
                    finished_at=finished_at,
                    mode="mock" if settings.mock else "live",
                    seed=seed,
                    tool_version=__version__,
                    models=_model_names(settings),
                ),
                profile=profile,
                classification=path,
                research=research,
                factors=factors.factors,
                screen=screen,
                hypotheses=hypotheses.hypotheses,
                critics=critics,
                warnings=warnings,
                corpus_version=corpus.version if corpus is not None else None,
                base_dir=settings.base_dir,
                scrape=scrape_result,
                model_search=model_search_result,
                variants=variant_set,
                research_models=research_models,
                collection=(
                    collection_metadata.model_dump(mode="json")
                    if collection_metadata is not None
                    else None
                ),
            )
            for hypothesis in hypotheses.hypotheses:
                section = next(
                    (
                        item
                        for item in provisional.hypotheses
                        if item.id == hypothesis.id
                    ),
                    None,
                )
                if section is None:
                    continue
                try:
                    report = validate_hypothesis(
                        section,
                        run_dir=run_dir,
                        artifact=provisional,
                        config=config,
                    )
                except Exception as exc:  # noqa: BLE001 - validation is advisory
                    report = error_report(
                        hypothesis.id,
                        f"validation failed: {type(exc).__name__}: {exc}",
                    )
                validations[hypothesis.id] = report
                failed = ", ".join(report.failed_layers()) or "none"
                logger(
                    f"validation {hypothesis.id} status={report.status} "
                    f"failed={failed}"
                )
            warnings = _warning_list(
                warnings,
                [
                    f"validation_warning: {report_id}: {warning}"
                    for report_id, report in validations.items()
                    for warning in report.warnings
                ],
            )
            try:
                _write_json(
                    run_dir / "validation.json",
                    {
                        "schema_version": "1.0.0",
                        "run_id": identifier,
                        "generated_at": utc_now_iso(),
                        "config": config.as_dict(),
                        "summary": summary_counts(list(validations.values())),
                        "reports": [
                            report.model_dump(mode="json")
                            for report in validations.values()
                        ],
                    },
                )
            except OSError as exc:
                logger(
                    "validation_report_write_failed "
                    f"{type(exc).__name__}: {exc}"
                )
        finished_at = utc_now_iso()
        artifact = build_artifact(
            run=RunInfo(
                run_id=identifier,
                started_at=started_at,
                finished_at=finished_at,
                mode="mock" if settings.mock else "live",
                seed=seed,
                tool_version=__version__,
                models=_model_names(settings),
            ),
            profile=profile,
            classification=path,
            research=research,
            factors=factors.factors,
            screen=screen,
            hypotheses=hypotheses.hypotheses,
            critics=critics,
            warnings=warnings,
            corpus_version=corpus.version if corpus is not None else None,
            base_dir=settings.base_dir,
            scrape=scrape_result,
            validations=validations or None,
            generation_validations=generation_validations or None,
            model_search=model_search_result,
            variants=variant_set,
            research_models=research_models,
            collection=(
                collection_metadata.model_dump(mode="json")
                if collection_metadata is not None
                else None
            ),
        )
        artifact_path = run_dir / "artifact.json"
        write_artifact(artifact_path, artifact)
        preview_path = run_dir / "preview.md"
        write_preview(preview_path, artifact, base_dir=run_dir)
        _write_json(run_dir / "research.json", _research_payload(research))
        _write_json(
            run_dir / "factors.json",
            [factor.model_dump(mode="json") for factor in factors.factors],
        )
        if scrape_result is not None:
            _write_json(
                run_dir / "scrape.json",
                scrape_result.model_dump(mode="json"),
            )
            scrape_md_path = run_dir / "scrape.md"
            scrape_md_path.write_text(
                render_scrape_preview(scrape_result), encoding="utf-8"
            )
        else:
            scrape_md_path = None
        logger(f"artifact {artifact_path.name} hypotheses={len(hypotheses.hypotheses)}")

        loaded_artifact, errors = validate_artifact(artifact_path)
        if errors or loaded_artifact is None:
            for error in errors:
                logger(f"artifact_invalid {error}")
            raise RunError(
                "the artifact failed handoff validation: " + "; ".join(errors[:5])
            )
        logger("run_done")
        return RunOutcome(
            run_dir=run_dir,
            artifact=artifact,
            artifact_path=artifact_path,
            preview_path=preview_path,
            scrape=scrape_result,
            scrape_md_path=scrape_md_path,
            validations=validations,
            generation_validations=generation_validations,
            model_search=model_search_result,
            variants=variant_set,
        )
    finally:
        for client, owned in ((llm, owns_llm), (embeddings, owns_embeddings)):
            if owned:
                close = getattr(client, "close", None)
                if callable(close):
                    close()
