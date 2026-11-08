"""End-to-end run orchestration: classify through artifact and preview."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
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
from .ids import run_id
from .io import read_dataset
from .llm import build_llm_client
from .preview import render_scrape_preview, write_preview
from .profile import build_profile, build_summary
from .research.pipeline import ResearchResult, refine_after_research, run_research
from .research.scrape.models import ScrapeResult
from .screen import build_screen
from .store import FieldStore


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
    if scrape_fetch is not None:
        settings = settings.with_overrides(scrape_fetch_enabled=bool(scrape_fetch))
    owns_llm = llm is None
    owns_embeddings = embeddings is None
    llm = llm if llm is not None else build_llm_client(settings)
    embeddings = (
        embeddings if embeddings is not None else build_embedding_client(settings)
    )
    scrape_result: ScrapeResult | None = None
    try:
        loaded = read_dataset(input_path, target=target, csv_sep=csv_sep)
        profile = build_profile(
            loaded,
            description=description,
            target=target,
            llm=llm,
        )
        identifier = run_id(profile.dataset_id)
        if out_dir is None:
            run_dir = settings.runs_dir / identifier
        else:
            run_dir = Path(out_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        logger = _RunLog(run_dir / "logs" / "run.log", log)
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

        context = HypothesisContext(
            profile=profile,
            summary=summary,
            path_names=path.names,
            factors=factors.factors,
            records=evidence_records,
            screen=screen,
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
        hypotheses: HypothesisSet = generator.generate()
        logger(
            "hypotheses "
            + ", ".join(
                f"{item.method_family}={item.execution.status}"
                for item in hypotheses.hypotheses
            )
        )

        corpus = None
        try:
            corpus = load_corpus(settings.corpus_path)
        except CorpusError as exc:
            logger(f"corpus_error {type(exc).__name__}: {exc}")
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
        )
    finally:
        for client, owned in ((llm, owns_llm), (embeddings, owns_embeddings)):
            if owned:
                close = getattr(client, "close", None)
                if callable(close):
                    close()
