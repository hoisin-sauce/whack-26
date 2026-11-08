"""Command-line entry point for the slice."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx

from . import __version__
from .artifact import artifact_json_schema, validate_artifact
from .classify import ClassificationError, Classifier
from .common import utc_now_iso
from .config import Settings
from .critic import CorpusError
from .doctor import exit_code, format_checks, run_doctor
from .embeddings import EmbeddingError, build_embedding_client
from .hypotheses import HypothesisGenerationError
from .io import IngestError, read_dataset
from .llm import LLMError, build_llm_client
from .preview import preview_from_path
from .profile import build_profile, build_summary
from .research.pipeline import run_research
from .run import RunError, run_pipeline
from .store import FieldStore, FieldStoreError

CALIFORNIA_HOUSING_URL = (
    "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
)
PENGUINS_URL = (
    "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="whack26",
        description="Dynamic research-to-hypothesis pipeline for tabular datasets.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="check Python, packages, keys, corpus, and providers")
    doctor.add_argument("--mock", action="store_true", help="check mock-mode readiness only")
    doctor.add_argument("--offline", action="store_true", help="skip all network probes")
    doctor.add_argument("--skip-network", action="store_true", help="skip all network probes")
    doctor.set_defaults(func=_cmd_doctor)

    fetch = sub.add_parser("fetch-demo", help="download the demo dataset")
    fetch.add_argument(
        "--out",
        default=None,
        help="destination path (default: data/california_housing.csv or data/penguins.csv)",
    )
    fetch.add_argument("--penguins", action="store_true", help="download Palmer Penguins instead")
    fetch.add_argument("--force", action="store_true", help="overwrite an existing file")
    fetch.set_defaults(func=_cmd_fetch_demo)

    classify = sub.add_parser(
        "classify",
        help="ingest, profile, and classify a dataset into the dynamic field tree",
    )
    classify.add_argument("--input", required=True, help="pandas-readable dataset path")
    classify.add_argument("--description", default=None, help="optional dataset description")
    classify.add_argument("--target", default=None, help="target column; inferred when omitted")
    classify.add_argument("--csv-sep", default=",", help="CSV separator (default: comma)")
    classify.add_argument(
        "--db",
        default=None,
        help="field store path (default: cache/fields.db)",
    )
    classify.add_argument("--out", default=None, help="optional JSON result path")
    classify.add_argument("--mock", action="store_true", help="use deterministic mock providers")
    classify.set_defaults(func=_cmd_classify)

    run = sub.add_parser("run", help="classify, research, hypothesise, critique, and emit an artifact")
    run.add_argument("--input", required=True, help="pandas-readable dataset path")
    run.add_argument("--description", default=None, help="optional dataset description")
    run.add_argument("--target", default=None, help="target column; inferred when omitted")
    run.add_argument("--csv-sep", default=",", help="CSV separator (default: comma)")
    run.add_argument("--out", default=None, help="run directory")
    run.add_argument(
        "--db",
        default=None,
        help="field store path (default: cache/fields.db)",
    )
    run.add_argument("--mock", action="store_true", help="use deterministic mock providers")
    run.add_argument("--offline", action="store_true", help="read research from cache only")
    run.add_argument("--refresh", action="store_true", help="ignore cache TTL")
    run.add_argument("--seed", type=int, default=0, help="random seed for generated code")
    run.add_argument(
        "--validate",
        action="store_true",
        help="run the four-layer statistical validator after hypothesis generation",
    )
    run.add_argument(
        "--validation-timeout",
        type=float,
        default=120.0,
        help="seconds per hypothesis validation subprocess (default: 120)",
    )
    run.add_argument(
        "--validation-num-permutations",
        type=int,
        default=500,
        help="Monte Carlo permutations for validation (default: 500)",
    )
    run.add_argument(
        "--validation-train-split",
        type=float,
        default=0.70,
        help="chronological train split for validation (default: 0.70)",
    )
    run.add_argument(
        "--scrape",
        action="store_true",
        help="enable semantic scraping of research documents",
    )
    run.add_argument(
        "--fetch",
        action="store_true",
        help="enable the compliant HTTP fetcher for scraping (robots-aware)",
    )
    run.add_argument(
        "--model-search",
        action="store_true",
        help="run the automated regression model search",
    )
    run.add_argument(
        "--variants",
        action="store_true",
        help="build anonymized and jumbled-label dataset variants",
    )
    run.add_argument(
        "--validate-generation",
        action="store_true",
        help="review each hypothesis against the dataset collection design",
    )
    run.add_argument(
        "--collection-json",
        default=None,
        help="collection-method metadata JSON file",
    )
    run.add_argument(
        "--collection-field",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="collection-method metadata override (repeatable)",
    )
    run.set_defaults(func=_cmd_run)

    scrape = sub.add_parser(
        "scrape",
        help="run the pipeline with semantic scraping enabled",
    )
    scrape.add_argument("--input", required=True, help="pandas-readable dataset path")
    scrape.add_argument("--description", default=None)
    scrape.add_argument("--target", default=None)
    scrape.add_argument("--csv-sep", default=",", help="CSV separator (default: comma)")
    scrape.add_argument("--out", default=None, help="run directory")
    scrape.add_argument(
        "--db",
        default=None,
        help="field store path (default: cache/fields.db)",
    )
    scrape.add_argument("--mock", action="store_true")
    scrape.add_argument("--offline", action="store_true")
    scrape.add_argument("--refresh", action="store_true")
    scrape.add_argument("--seed", type=int, default=0)
    scrape.add_argument(
        "--fetch",
        action="store_true",
        help="enable the compliant HTTP fetcher for scraping (robots-aware)",
    )
    scrape.add_argument("--max-resources", type=int, default=None)
    scrape.add_argument("--max-depth", type=int, default=None)
    scrape.set_defaults(func=_cmd_scrape)

    preload = sub.add_parser(
        "preload-statistics",
        help="preload the statistics/validity corpus into the document store",
    )
    preload.add_argument("--limit", type=int, default=200)
    preload.add_argument(
        "--db",
        default=None,
        help="field store path (default: cache/fields.db)",
    )
    preload.add_argument("--mock", action="store_true")
    preload.add_argument("--offline", action="store_true")
    preload.set_defaults(func=_cmd_preload_statistics)

    prefetch = sub.add_parser("prefetch", help="warm the research cache for a dataset")
    prefetch.add_argument("--input", required=True, help="pandas-readable dataset path")
    prefetch.add_argument("--description", default=None)
    prefetch.add_argument("--target", default=None)
    prefetch.add_argument("--csv-sep", default=",", help="CSV separator (default: comma)")
    prefetch.add_argument(
        "--db",
        default=None,
        help="field store path (default: cache/fields.db)",
    )
    prefetch.add_argument("--out", default=None, help="optional JSON result path")
    prefetch.add_argument("--refresh", action="store_true")
    prefetch.add_argument("--mock", action="store_true")
    prefetch.set_defaults(func=_cmd_prefetch)

    validate = sub.add_parser("validate-artifact", help="validate a run artifact")
    validate.add_argument("path", help="path to artifact.json")
    validate.set_defaults(func=_cmd_validate_artifact)

    validate_models = sub.add_parser(
        "validate-models",
        help="run the four-layer statistical validator over generated hypotheses",
    )
    validate_models.add_argument(
        "path",
        help="run directory or path to artifact.json",
    )
    validate_models.add_argument(
        "--out",
        default=None,
        help="validation JSON output path (default: <run>/validation.json)",
    )
    validate_models.add_argument(
        "--write-artifact",
        action="store_true",
        help="embed the reports in artifact.json and regenerate preview.md",
    )
    validate_models.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="seconds per hypothesis subprocess (default: 120)",
    )
    validate_models.add_argument(
        "--num-permutations",
        type=int,
        default=500,
        help="Monte Carlo permutations (default: 500)",
    )
    validate_models.add_argument(
        "--train-split-ratio",
        type=float,
        default=0.70,
        help="chronological train split (default: 0.70)",
    )
    validate_models.add_argument(
        "--min-dsr",
        type=float,
        default=0.95,
        help="Layer 1 DSR threshold (default: 0.95)",
    )
    validate_models.add_argument(
        "--max-p",
        type=float,
        default=0.05,
        help="Layer 2 p-value threshold (default: 0.05)",
    )
    validate_models.add_argument(
        "--periods-per-year",
        type=float,
        default=None,
        help="annualisation factor; inferred from the time column when omitted",
    )
    validate_models.add_argument(
        "--sensitivity-range",
        type=float,
        default=0.10,
        help="Layer 4 perturbation range (default: 0.10)",
    )
    validate_models.add_argument(
        "--max-performance-drop",
        type=float,
        default=0.20,
        help="Layer 4 maximum score drop (default: 0.20)",
    )
    validate_models.add_argument(
        "--max-decay-tolerance",
        type=float,
        default=None,
        help="Layer 3 maximum OOS decay; engine default when omitted",
    )
    validate_models.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed passed to the generated model's fit helper",
    )
    validate_models.add_argument(
        "--generation",
        action="store_true",
        help="also re-run claim/design generation validation over the artifact",
    )
    validate_models.add_argument(
        "--collection-json",
        default=None,
        help="collection-method metadata JSON file for generation validation",
    )
    validate_models.add_argument(
        "--collection-field",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="collection-method metadata override (repeatable)",
    )
    validate_models.set_defaults(func=_cmd_validate_models)

    preview = sub.add_parser("preview", help="render preview.md from a run artifact")
    preview.add_argument("artifact", help="path to artifact.json")
    preview.add_argument("--out", default=None, help="output path (default: beside artifact)")
    preview.set_defaults(func=_cmd_preview)

    schema = sub.add_parser("schema", help="write the artifact JSON schema")
    schema.add_argument(
        "--out",
        default="schemas/artifact.schema.json",
        help="output path (default: schemas/artifact.schema.json)",
    )
    schema.set_defaults(func=_cmd_schema)

    return parser


def _cmd_doctor(args: argparse.Namespace) -> int:
    settings = Settings.from_env(mock=args.mock, offline=True if args.offline else None)
    network = not args.skip_network and not settings.offline
    checks = run_doctor(settings, network=network)
    print(format_checks(checks))
    return exit_code(checks)


def _download(url: str, destination: Path, *, force: bool) -> int:
    if destination.exists() and not force:
        print(f"Already exists: {destination} ({destination.stat().st_size:,} bytes)")
        print("Use --force to overwrite.")
        return 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            content = response.content
    except httpx.HTTPError as exc:
        print(f"error: download failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if not content:
        print("error: download returned an empty body", file=sys.stderr)
        return 1
    destination.write_bytes(content)
    print(f"Saved {destination} ({len(content):,} bytes)")
    return 0


def _cmd_fetch_demo(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    if args.penguins:
        url = PENGUINS_URL
        destination = Path(args.out) if args.out else settings.data_dir / "penguins.csv"
    else:
        url = CALIFORNIA_HOUSING_URL
        destination = (
            Path(args.out) if args.out else settings.data_dir / "california_housing.csv"
        )
    return _download(url, destination, force=args.force)


def _cmd_classify(args: argparse.Namespace) -> int:
    settings = Settings.from_env(mock=args.mock)
    llm = build_llm_client(settings)
    embeddings = build_embedding_client(settings)
    try:
        loaded = read_dataset(
            Path(args.input),
            target=args.target,
            csv_sep=args.csv_sep,
        )
        profile = build_profile(
            loaded,
            description=args.description,
            target=args.target,
            llm=llm,
        )
        store_path = Path(args.db) if args.db else settings.fields_db
        with FieldStore(
            store_path,
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
        summary = build_summary(profile, loaded.frame)
    except (
        IngestError,
        ClassificationError,
        FieldStoreError,
        LLMError,
        EmbeddingError,
        ValueError,
    ) as exc:
        print(f"error: classification failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        for client in (llm, embeddings):
            close = getattr(client, "close", None)
            if callable(close):
                close()

    print(f"dataset: {profile.dataset_id} ({profile.rows} rows x {len(profile.columns)} columns)")
    print(f"target: {profile.target.name if profile.target else 'none'}")
    print(f"path: {' > '.join(path.names)}")
    print(f"confidence: {path.confidence:.3f}")
    print(f"created fields: {len(path.created_fields)}")
    for warning in path.warnings:
        print(f"warning: {warning}")

    if args.out:
        profile_payload = profile.model_dump(mode="json")
        profile_payload["columns"] = [
            {
                key: column[key]
                for key in ("name", "dtype", "role", "missing_pct", "cardinality")
            }
            for column in profile_payload["columns"]
        ]
        payload = {
            "schema_version": "0.1.0",
            "generated_at": utc_now_iso(),
            "dataset_id": profile.dataset_id,
            "profile": profile_payload,
            "summary": summary,
            "classification": {
                "path": [
                    {
                        "id": item.id,
                        "parent_id": item.parent_id,
                        "level": item.level,
                        "name": item.name,
                        "description": item.description,
                    }
                    for item in path.fields
                ],
                "confidence": path.confidence,
                "created_fields": path.created_fields,
                "refined_after_research": path.refined_after_research,
                "warnings": path.warnings,
            },
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out_path}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from .validation.collection import parse_collection_metadata
    from .validation.models import ValidationConfig

    validation_config = ValidationConfig(
        timeout_seconds=args.validation_timeout,
        num_permutations=args.validation_num_permutations,
        train_split_ratio=args.validation_train_split,
    )
    collection = None
    if args.collection_json or args.collection_field:
        try:
            collection = parse_collection_metadata(
                json_source=args.collection_json,
                fields=args.collection_field,
            )
        except ValueError as exc:
            print(
                f"error: invalid collection metadata: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 1
    try:
        outcome = run_pipeline(
            input_path=Path(args.input),
            description=args.description,
            target=args.target,
            seed=args.seed,
            mock=args.mock,
            offline=args.offline,
            refresh=args.refresh,
            scrape=True if args.scrape else None,
            scrape_fetch=True if args.fetch else None,
            model_search=True if args.model_search else None,
            variants=True if args.variants else None,
            validate_generation=True if args.validate_generation else None,
            collection=collection,
            validate=args.validate,
            validation_config=validation_config,
            csv_sep=args.csv_sep,
            out_dir=Path(args.out) if args.out else None,
            store_path=Path(args.db) if args.db else None,
            log=print,
        )
    except (
        IngestError,
        ClassificationError,
        FieldStoreError,
        LLMError,
        EmbeddingError,
        HypothesisGenerationError,
        CorpusError,
        RunError,
        OSError,
        ValueError,
    ) as exc:
        print(f"error: run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    artifact = outcome.artifact
    print(f"run: {outcome.run_dir}")
    print(f"path: {' > '.join(item.name for item in artifact.classification.path)}")
    print(
        f"research: {len(artifact.research.records)} record(s), "
        f"{len(artifact.factors)} factor(s)"
    )
    for hypothesis in artifact.hypotheses:
        verdict = hypothesis.critic.verdict if hypothesis.critic else "unreviewed"
        print(
            f"hypothesis: {hypothesis.id} {hypothesis.method_family} "
            f"{hypothesis.execution.status} critic={verdict}"
        )
        validation = outcome.validations.get(hypothesis.id)
        if validation is not None:
            failed = ", ".join(validation.failed_layers()) or "none"
            print(
                f"validation: {hypothesis.id} {validation.status} "
                f"failed={failed}"
            )
    if outcome.scrape is not None:
        print(
            f"scrape: {len(outcome.scrape.documents)} document(s), "
            f"{len(outcome.scrape.comparisons)} comparison(s), "
            f"{len(outcome.scrape.critiques)} critique(s)"
        )
        if outcome.scrape_md_path is not None:
            print(f"scrape markdown: {outcome.scrape_md_path}")
    if outcome.model_search is not None:
        selected = getattr(outcome.model_search, "selected", None)
        family = getattr(getattr(selected, "spec", None), "family", None)
        print(
            f"model search: {len(outcome.model_search.candidates)} candidate(s), "
            f"selected={family or 'none'}"
        )
    if outcome.variants is not None:
        verdict = getattr(outcome.variants, "verdict", "unknown")
        print(f"variants: verdict={verdict}")
    for hypothesis_id, result in outcome.generation_validations.items():
        print(
            f"generation validation: {hypothesis_id} {result.verdict} "
            f"issues={len(result.issues)}"
        )
    print(f"artifact: {outcome.artifact_path}")
    print(f"preview: {outcome.preview_path}")
    return 0


def _cmd_scrape(args: argparse.Namespace) -> int:
    settings = Settings.from_env(
        mock=args.mock, offline=True if args.offline else None
    )
    overrides: dict[str, object] = {"scrape_enabled": True}
    if args.max_resources is not None:
        overrides["scrape_max_resources"] = args.max_resources
    if args.max_depth is not None:
        overrides["scrape_max_depth"] = args.max_depth
    settings = settings.with_overrides(**overrides)
    try:
        outcome = run_pipeline(
            input_path=Path(args.input),
            description=args.description,
            target=args.target,
            seed=args.seed,
            offline=args.offline,
            refresh=args.refresh,
            scrape=True,
            scrape_fetch=True if args.fetch else None,
            csv_sep=args.csv_sep,
            out_dir=Path(args.out) if args.out else None,
            store_path=Path(args.db) if args.db else None,
            settings=settings,
            log=print,
        )
    except (
        IngestError,
        ClassificationError,
        FieldStoreError,
        LLMError,
        EmbeddingError,
        HypothesisGenerationError,
        CorpusError,
        RunError,
        OSError,
        ValueError,
    ) as exc:
        print(f"error: scrape failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    scrape = outcome.scrape
    print(f"run: {outcome.run_dir}")
    if scrape is None:
        print("scrape: no result")
        return 0
    print(
        f"scrape: {len(scrape.documents)} document(s), "
        f"{len(scrape.links)} link(s), "
        f"{len(scrape.comparisons)} comparison(s), "
        f"{len(scrape.critiques)} critique(s)"
    )
    for document in scrape.documents[:10]:
        print(
            f"document: {document.id} {document.kind} "
            f"credible={document.credible} score={document.credibility_score:.2f} "
            f"{document.title}"
        )
    for warning in scrape.warnings:
        print(f"warning: {warning}")
    print(f"scrape json: {outcome.run_dir / 'scrape.json'}")
    if outcome.scrape_md_path is not None:
        print(f"scrape markdown: {outcome.scrape_md_path}")
    print(f"artifact: {outcome.artifact_path}")
    return 0


def _cmd_preload_statistics(args: argparse.Namespace) -> int:
    settings = Settings.from_env(
        mock=args.mock, offline=True if args.offline else None
    )
    from .research.scrape.preload import preload_statistics

    llm = build_llm_client(settings)
    embeddings = build_embedding_client(settings)
    try:
        with FieldStore(
            Path(args.db) if args.db else settings.fields_db,
            dims=settings.embed_dims,
            max_depth=settings.max_depth,
        ) as store:
            result = preload_statistics(
                store=store,
                llm=llm,
                embeddings=embeddings,
                settings=settings,
                limit=args.limit,
            )
    except (LLMError, EmbeddingError, FieldStoreError, OSError, ValueError) as exc:
        print(
            f"error: preload failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    finally:
        for client in (llm, embeddings):
            close = getattr(client, "close", None)
            if callable(close):
                close()

    print(
        f"preload: {len(result.documents)} document(s), "
        f"{len(result.warnings)} warning(s)"
    )
    for warning in result.warnings:
        print(f"warning: {warning}")
    for key, value in sorted(result.stats.items()):
        print(f"{key}: {value}")
    return 0


def _cmd_prefetch(args: argparse.Namespace) -> int:
    settings = Settings.from_env(mock=args.mock)
    llm = build_llm_client(settings)
    embeddings = build_embedding_client(settings)
    try:
        loaded = read_dataset(
            Path(args.input),
            target=args.target,
            csv_sep=args.csv_sep,
        )
        profile = build_profile(
            loaded,
            description=args.description,
            target=args.target,
            llm=llm,
        )
        store_path = Path(args.db) if args.db else settings.fields_db
        with FieldStore(
            store_path,
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
        summary = build_summary(profile, loaded.frame)
        result = run_research(
            profile=profile,
            summary=summary,
            path_names=path.names,
            llm=llm,
            embeddings=embeddings,
            settings=settings,
            cache_path=settings.research_db,
            refresh=args.refresh,
        )
    except (
        IngestError,
        ClassificationError,
        FieldStoreError,
        LLMError,
        EmbeddingError,
        ValueError,
        OSError,
    ) as exc:
        print(
            f"error: prefetch failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    finally:
        for client in (llm, embeddings):
            close = getattr(client, "close", None)
            if callable(close):
                close()

    print(f"dataset: {profile.dataset_id} ({profile.rows} rows x {len(profile.columns)} columns)")
    print(f"target: {profile.target.name if profile.target else 'none'}")
    print(f"path: {' > '.join(path.names)}")
    print(f"queries: {len(result.queries)}")
    counts: dict[str, int] = {}
    for record in result.records:
        counts[record.source] = counts.get(record.source, 0) + 1
    shown = ", ".join(f"{source}={count}" for source, count in sorted(counts.items()))
    print(f"records: {len(result.records)}" + (f" ({shown})" if shown else ""))
    print(
        "cache: "
        f"hits={result.cache.get('hits', 0)} "
        f"misses={result.cache.get('misses', 0)} "
        f"writes={result.cache.get('writes', 0)} "
        f"hit_rate={result.cache.get('hit_rate', 0.0):.3f}"
    )
    for warning in result.warnings:
        print(f"warning: {warning}")

    if args.out:
        payload = {
            "schema_version": "0.1.0",
            "generated_at": utc_now_iso(),
            "dataset": {
                "id": profile.dataset_id,
                "source": Path(profile.source_path).name,
                "rows": profile.rows,
                "columns": len(profile.columns),
                "target": profile.target.name if profile.target else None,
                "description": profile.description,
            },
            "classification": {
                "path": [
                    {
                        "id": item.id,
                        "parent_id": item.parent_id,
                        "level": item.level,
                        "name": item.name,
                        "description": item.description,
                    }
                    for item in path.fields
                ],
                "confidence": path.confidence,
                "warnings": path.warnings,
            },
            "research": {
                "queries": [query.model_dump(mode="json") for query in result.queries],
                "records": [
                    record.model_dump(mode="json") for record in result.records
                ],
                "keywords": result.keywords,
                "sources_used": result.sources_used,
                "limited_mode": result.limited_mode,
                "offline": result.offline,
                "cache": result.cache,
                "warnings": result.warnings,
            },
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {out_path}")
    return 0


def _cmd_validate_artifact(args: argparse.Namespace) -> int:
    path = Path(args.path)
    artifact, errors = validate_artifact(path)
    if artifact is None or errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"OK {path}")
    print(f"schema: {artifact.schema_version}")
    print(f"dataset: {artifact.dataset.id} ({artifact.dataset.rows} rows)")
    print(f"hypotheses: {len(artifact.hypotheses)}")
    verdicts = artifact.critic_summary.verdicts
    print(
        "critic: "
        + ", ".join(f"{verdict}={verdicts.get(verdict, 0)}" for verdict in (
            "pass",
            "refine",
            "reject",
            "insufficient",
        ))
    )
    return 0


def _cmd_preview(args: argparse.Namespace) -> int:
    try:
        destination = preview_from_path(
            Path(args.artifact),
            out=Path(args.out) if args.out else None,
        )
    except (OSError, ValueError) as exc:
        print(f"error: preview failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {destination}")
    return 0


def _cmd_validate_models(args: argparse.Namespace) -> int:
    from .validation.models import ValidationConfig
    from .validation.pipeline import validate_run

    config = ValidationConfig(
        min_dsr=args.min_dsr,
        max_p=args.max_p,
        train_split_ratio=args.train_split_ratio,
        num_permutations=args.num_permutations,
        sensitivity_range=args.sensitivity_range,
        max_performance_drop=args.max_performance_drop,
        max_decay_tolerance=args.max_decay_tolerance,
        periods_per_year=args.periods_per_year,
        timeout_seconds=args.timeout,
        seed=args.seed,
    )
    try:
        result = validate_run(
            Path(args.path),
            config=config,
            out=Path(args.out) if args.out else None,
            write_artifact=args.write_artifact,
        )
    except (OSError, ValueError) as exc:
        print(
            f"error: validation failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    print(f"run: {result.run_dir}")
    for report in result.reports:
        failed = ", ".join(report.failed_layers()) or "none"
        print(
            f"validation: {report.hypothesis_id} {report.status} "
            f"failed={failed}"
        )
        for warning in report.warnings:
            print(f"warning: {report.hypothesis_id}: {warning}")
    summary = {}
    for report in result.reports:
        summary[report.status] = summary.get(report.status, 0) + 1
    print(
        "summary: "
        + ", ".join(f"{key}={value}" for key, value in sorted(summary.items()))
    )
    print(f"validation json: {result.report_path}")
    if args.write_artifact:
        print(f"artifact: {result.artifact_path}")
    if args.generation:
        from .artifact import Artifact
        from .validation.collection import parse_collection_metadata
        from .validation.generation import GenerationValidator

        artifact_path = Path(args.path)
        if artifact_path.is_dir():
            artifact_path = artifact_path / "artifact.json"
        try:
            artifact = Artifact.model_validate(
                json.loads(artifact_path.read_text(encoding="utf-8"))
            )
            collection = None
            if args.collection_json or args.collection_field:
                collection = parse_collection_metadata(
                    json_source=args.collection_json,
                    fields=args.collection_field,
                )
            elif artifact.collection:
                collection = artifact.collection
            settings = Settings.from_env()
            validator = GenerationValidator(settings=settings)
            gen_results = []
            run_dir = artifact_path.parent
            for section in artifact.hypotheses:
                payload = section.model_dump(mode="json")
                code_path = run_dir / section.code_path
                try:
                    payload["code"] = code_path.read_text(encoding="utf-8")
                except OSError:
                    payload["code"] = ""
                hypothesis = SimpleNamespace(**payload)
                gen_result = validator.validate(
                    hypothesis,
                    collection=collection,
                    code_hash=section.code_sha256,
                )
                gen_results.append(gen_result)
                print(
                    f"generation validation: {section.id} "
                    f"{gen_result.verdict} issues={len(gen_result.issues)}"
                )
            (run_dir / "generation_validation.json").write_text(
                json.dumps(
                    {
                        "schema_version": "1.0.0",
                        "generated_at": utc_now_iso(),
                        "results": [
                            item.model_dump(mode="json") for item in gen_results
                        ],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        except (OSError, ValueError) as exc:
            print(
                f"error: generation validation failed: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 1
    return 0


def _cmd_schema(args: argparse.Namespace) -> int:
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(artifact_json_schema(), indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {destination}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
