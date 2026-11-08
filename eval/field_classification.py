"""Field-classification evaluation gate.

Runs the classifier over ten general-domain datasets in three passes (cold,
warm, shuffled cold), computes the automated gates from the spec, accepts
human scores, writes a Markdown report and a JSON result, and exits non-zero
when a gate fails.

Usage:
    python eval/field_classification.py --mock --repeat
    python eval/field_classification.py --scores eval/scores.json
    python eval/field_classification.py --write-scores-template eval/scores.json
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from whack26.classify import ClassificationError, Classifier  # noqa: E402
from whack26.config import Settings  # noqa: E402
from whack26.embeddings import (  # noqa: E402
    EmbeddingError,
    build_embedding_client,
    cosine,
)
from whack26.io import IngestError, read_dataset  # noqa: E402
from whack26.llm import LLMError, build_llm_client  # noqa: E402
from whack26.profile import build_profile  # noqa: E402
from whack26.store import FieldStore, FieldStoreError  # noqa: E402

DEFAULT_OUTPUT = REPO_ROOT / "docs" / "eval" / "field-classification-results.md"
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "eval"
DEFAULT_WORK_DIR = REPO_ROOT / "runs" / "eval"

COLD_RUNTIME_LIMIT_S = 10 * 60
WARM_RUNTIME_LIMIT_S = 5 * 60


@dataclass(frozen=True)
class EvalDataset:
    slug: str
    name: str
    domain: str
    url: str
    filename: str
    description: str | None
    target: str | None
    acceptable_roots: tuple[str, ...]
    forbidden_labels: tuple[str, ...]
    good_path: str
    csv_sep: str = ","
    archive_member: str | None = None
    local_path: str | None = None


DATASETS: tuple[EvalDataset, ...] = (
    EvalDataset(
        slug="penguins",
        name="Palmer Penguins",
        domain="biology / ecology",
        url="https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv",
        filename="penguins.csv",
        description=(
            "Morphological measurements (bill, flipper, body mass) and sex for "
            "three penguin species observed on three islands in the Palmer Archipelago."
        ),
        target="species",
        acceptable_roots=(
            "biology",
            "ecology",
            "zoology",
            "life sciences",
            "biological sciences",
            "animal science",
            "ornithology",
        ),
        forbidden_labels=("palmer penguins", "penguins dataset", "penguin data"),
        good_path="Life sciences > Ecology or zoology > Morphology / species",
        local_path="data/penguins.csv",
    ),
    EvalDataset(
        slug="wine",
        name="Wine Quality",
        domain="food / chemistry",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        filename="winequality-red.csv",
        description=(
            "Physicochemical measurements of red wine samples (acidity, sugar, "
            "sulphates, alcohol) paired with sensory quality ratings."
        ),
        target="quality",
        acceptable_roots=(
            "food science",
            "chemistry",
            "agriculture",
            "food and beverage",
            "oenology",
            "food",
        ),
        forbidden_labels=("wine quality dataset", "winequality", "uci"),
        good_path="Food science > Oenology > Wine quality",
        csv_sep=";",
    ),
    EvalDataset(
        slug="california",
        name="California Housing",
        domain="housing / economics",
        url="https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv",
        filename="california_housing.csv",
        description=(
            "Block-group housing statistics: median house value, median income, "
            "housing age, rooms, population, and ocean proximity."
        ),
        target="median_house_value",
        acceptable_roots=(
            "housing",
            "economics",
            "real estate",
            "urban planning",
            "geography",
            "social science",
        ),
        forbidden_labels=("california housing", "housing.csv", "california_housing"),
        good_path="Housing or economics > Real estate > Housing prices",
        local_path="data/california_housing.csv",
    ),
    EvalDataset(
        slug="titanic",
        name="Titanic",
        domain="demographics / history",
        url="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
        filename="titanic.csv",
        description=None,
        target="Survived",
        acceptable_roots=(
            "history",
            "demography",
            "sociology",
            "maritime history",
            "disaster studies",
            "social science",
        ),
        forbidden_labels=("titanic", "titanic dataset"),
        good_path="History > Maritime disasters or demography > Survival",
    ),
    EvalDataset(
        slug="air_quality",
        name="Air Quality (UCI)",
        domain="environment",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip",
        filename="AirQualityUCI.zip",
        description=(
            "Hourly air-quality sensor readings from an Italian city, including "
            "carbon monoxide, nitrogen oxides, benzene, temperature, and humidity."
        ),
        target="CO(GT)",
        acceptable_roots=(
            "environment",
            "environmental science",
            "air quality",
            "atmospheric science",
            "public health",
            "chemistry",
        ),
        forbidden_labels=("airqualityuci", "uci air quality", "air quality dataset"),
        good_path="Environment > Air quality > Pollution monitoring",
        archive_member="AirQualityUCI.xlsx",
    ),
    EvalDataset(
        slug="covid",
        name="COVID Country Series",
        domain="public health",
        url="https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv",
        filename="owid-covid-data.csv",
        description=(
            "Country-level daily COVID-19 cases, deaths, vaccinations, testing, "
            "and policy indicators over the course of the pandemic."
        ),
        target="new_cases",
        acceptable_roots=(
            "public health",
            "epidemiology",
            "health",
            "demography",
            "infectious disease",
        ),
        forbidden_labels=("owid", "owid-covid", "covid-19 dataset", "coronavirus dataset"),
        good_path="Public health > Epidemiology > Infectious disease surveillance",
    ),
    EvalDataset(
        slug="spotify",
        name="Spotify Tracks",
        domain="music",
        url="https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv",
        filename="spotify_songs.csv",
        description=None,
        target="track_popularity",
        acceptable_roots=(
            "music",
            "media",
            "culture",
            "entertainment",
            "arts",
            "audio",
        ),
        forbidden_labels=("spotify", "tidytuesday", "spotify_songs"),
        good_path="Music > Musicology or audio analysis > Track popularity",
    ),
    EvalDataset(
        slug="student",
        name="Student Performance",
        domain="education",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip",
        filename="student.zip",
        description=(
            "Secondary-school student grades, demographics, family background, "
            "study habits, and school support."
        ),
        target="G3",
        acceptable_roots=(
            "education",
            "social science",
            "sociology",
            "demography",
            "learning",
        ),
        forbidden_labels=("student performance dataset", "student-mat", "student-por", "uci"),
        good_path="Education > Student outcomes > Academic performance",
        csv_sep=";",
        archive_member="student-mat.csv",
    ),
    EvalDataset(
        slug="retail",
        name="Online Retail",
        domain="retail / commerce",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx",
        filename="Online Retail.xlsx",
        description=None,
        target=None,
        acceptable_roots=(
            "retail",
            "commerce",
            "business",
            "economics",
            "marketing",
        ),
        forbidden_labels=("online retail", "online_retail", "uci"),
        good_path="Retail / commerce > E-commerce > Transactions",
    ),
    EvalDataset(
        slug="soccer",
        name="Soccer Match Results",
        domain="sports",
        url="https://www.football-data.co.uk/mmz4281/2324/E0.csv",
        filename="E0.csv",
        description=None,
        target="FTR",
        acceptable_roots=(
            "sports",
            "sports science",
            "recreation",
            "statistics",
        ),
        forbidden_labels=("football-data", "e0.csv"),
        good_path="Sports > Association football > Match results",
    ),
)

DATASETS_BY_SLUG = {item.slug: item for item in DATASETS}


@dataclass
class LoadedInput:
    entry: EvalDataset
    profile: Any
    frame: Any


@dataclass
class DatasetRun:
    slug: str
    path_names: list[str] = field(default_factory=list)
    path_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    created_fields: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    error: str | None = None

    @property
    def depth(self) -> int:
        return len(self.path_names)

    @property
    def root_name(self) -> str:
        return self.path_names[0] if self.path_names else ""


@dataclass
class RoundResult:
    name: str
    elapsed_s: float
    records: list[DatasetRun]
    stats: dict[str, int]
    roots: list[tuple[str, list[float]]] = field(default_factory=list)


@dataclass
class HumanScore:
    root: int
    path: int
    naming: int | None
    sibling_distinctness: bool
    notes: str = ""


@dataclass
class GateResult:
    name: str
    status: Literal["pass", "fail", "pending", "skip"]
    detail: str
    blocking: bool = True


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize(text: str) -> str:
    return " ".join(
        "".join(character if character.isalnum() else " " for character in text.lower()).split()
    )


def _contains_token_sequence(node: str, label: str) -> bool:
    """Return True when ``label`` appears as whole tokens inside ``node``.

    Forbidden labels are dataset proper nouns.  A plain substring test would
    flag unrelated words such as ``uci`` inside ``mucilage``; matching whole
    token sequences keeps the gate tied to the actual label.
    """
    node_tokens = _normalize(node).split()
    label_tokens = _normalize(label).split()
    if not label_tokens or len(label_tokens) > len(node_tokens):
        return False
    width = len(label_tokens)
    return any(
        node_tokens[index : index + width] == label_tokens
        for index in range(len(node_tokens) - width + 1)
    )


# Human review and the automated human-score gates share these criteria.
HUMAN_PASS_FRACTION = 0.8
HUMAN_MIN_ACCEPTABLE_SCORE = 3
HUMAN_MEAN_THRESHOLD = 3.5
HUMAN_NAMING_MEAN_THRESHOLD = 4.0
# When the automated naming gate passes and the scorer omits ``naming``, assume
# the automated result is correct and give the dataset this score.  This keeps
# the human review focused on root/path plausibility and sibling distinctness.
HUMAN_NAMING_ASSUMED_SCORE = 4
HUMAN_PASS_CRITERIA = (
    f"at least {HUMAN_PASS_FRACTION:.0%} of datasets score "
    f">= {HUMAN_MIN_ACCEPTABLE_SCORE} on the root and, separately, at least "
    f"{HUMAN_PASS_FRACTION:.0%} score >= {HUMAN_MIN_ACCEPTABLE_SCORE} on the "
    f"full path, the mean root and path scores are at least "
    f"{HUMAN_MEAN_THRESHOLD:.1f}, the mean naming score is at least "
    f"{HUMAN_NAMING_MEAN_THRESHOLD:.1f}, and at least "
    f"{HUMAN_PASS_FRACTION:.0%} of descriptions distinguish the field"
)


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _download(url: str, destination: Path, *, attempts: int = 3) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with httpx.stream(
                "GET",
                url,
                follow_redirects=True,
                timeout=httpx.Timeout(120.0, connect=30.0),
                headers={"User-Agent": "whack26-eval/1.0"},
            ) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for chunk in response.iter_bytes(1024 * 1024):
                        handle.write(chunk)
            temporary.replace(destination)
            return
        except (httpx.HTTPError, OSError) as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            if attempt < attempts:
                time.sleep(2 * attempt)
    raise RuntimeError(f"could not download {url}: {last_error}")


def _resolve_dataset(entry: EvalDataset, data_dir: Path) -> Path:
    if entry.local_path:
        local = REPO_ROOT / entry.local_path
        if local.is_file():
            return local
    target_dir = data_dir / entry.slug
    if entry.archive_member:
        member_path = target_dir / entry.archive_member
        if member_path.is_file():
            return member_path
        archive_path = target_dir / entry.filename
        if not archive_path.is_file():
            _download(entry.url, archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            match = next(
                (
                    name
                    for name in names
                    if Path(name).name == entry.archive_member
                ),
                None,
            )
            if match is None:
                raise RuntimeError(
                    f"{entry.filename} does not contain {entry.archive_member}; "
                    f"found {names}"
                )
            member_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(match) as source, member_path.open("wb") as destination:
                shutil.copyfileobj(source, destination)
        return member_path
    destination = target_dir / entry.filename
    if not destination.is_file():
        _download(entry.url, destination)
    return destination


def load_inputs(entries: Sequence[EvalDataset], data_dir: Path) -> list[LoadedInput]:
    loaded_inputs: list[LoadedInput] = []
    for entry in entries:
        path = _resolve_dataset(entry, data_dir)
        loaded = read_dataset(path, target=entry.target, csv_sep=entry.csv_sep)
        profile = build_profile(
            loaded,
            description=entry.description,
            target=entry.target,
            llm=None,
        )
        loaded_inputs.append(LoadedInput(entry=entry, profile=profile, frame=loaded.frame))
        print(
            f"loaded {entry.slug}: {profile.rows} rows, "
            f"{len(profile.columns)} columns, target={profile.target.name if profile.target else 'none'}"
        )
    return loaded_inputs


def _reset_store(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm"):
        Path(str(path) + suffix).unlink(missing_ok=True)


def run_round(
    *,
    name: str,
    inputs: Sequence[LoadedInput],
    settings: Settings,
    llm: Any,
    embeddings: Any,
    store_path: Path,
    fresh: bool,
) -> RoundResult:
    if fresh:
        _reset_store(store_path)
    started = time.perf_counter()
    records: list[DatasetRun] = []
    stats: dict[str, int] = {}
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
        for item in inputs:
            item_started = time.perf_counter()
            record = DatasetRun(slug=item.entry.slug)
            try:
                path = classifier.classify(item.profile, item.frame)
                record.path_names = list(path.names)
                record.path_ids = list(path.ids)
                record.confidence = float(path.confidence)
                record.created_fields = list(path.created_fields)
                record.warnings = list(path.warnings)
            except Exception as exc:  # noqa: BLE001 - the gate records failures
                record.error = f"{type(exc).__name__}: {exc}"
            record.elapsed_s = time.perf_counter() - item_started
            records.append(record)
            shown = " > ".join(record.path_names) if record.path_names else "(no path)"
            print(
                f"[{name}] {item.entry.slug}: {shown} "
                f"depth={record.depth} confidence={record.confidence:.3f} "
                f"{record.elapsed_s:.1f}s"
                + (f" ERROR {record.error}" if record.error else "")
            )
        stats = store.stats()
        roots = [(item.name, list(item.description_embedding)) for item in store.roots()]
    return RoundResult(
        name=name,
        elapsed_s=time.perf_counter() - started,
        records=records,
        stats=stats,
        roots=roots,
    )


def _record_map(result: RoundResult) -> dict[str, DatasetRun]:
    return {record.slug: record for record in result.records}


def _same_root(left: DatasetRun, right: DatasetRun) -> bool:
    return bool(left.root_name) and _normalize(left.root_name) == _normalize(right.root_name)


def _same_path(left: DatasetRun, right: DatasetRun) -> bool:
    return (
        bool(left.path_names)
        and [_normalize(name) for name in left.path_names]
        == [_normalize(name) for name in right.path_names]
    )


def _duplicate_roots(
    roots: Sequence[tuple[str, list[float]]],
) -> tuple[int, int]:
    """Count duplicate root fields, not datasets that share a root."""
    exact_groups = Counter(_normalize(name) for name, _ in roots if name)
    exact_duplicates = sum(count - 1 for count in exact_groups.values() if count > 1)
    vectors = [embedding for _, embedding in roots if embedding]
    near_pairs = 0
    for index, left in enumerate(vectors):
        for right in vectors[index + 1 :]:
            if cosine(left, right) >= 0.90:
                near_pairs += 1
    return exact_duplicates, near_pairs


def _ratio_gate(
    name: str,
    matched: int,
    total: int,
    fraction: float,
    description: str,
    *,
    blocking: bool = True,
) -> GateResult:
    required = max(1, math.ceil(fraction * total)) if total else 1
    status: Literal["pass", "fail"] = "pass" if matched >= required else "fail"
    return GateResult(
        name=name,
        status=status,
        detail=f"{description}: {matched}/{total} (required {required})",
        blocking=blocking,
    )


def evaluate_automated(
    *,
    cold: RoundResult,
    warm: RoundResult,
    shuffled: RoundResult,
    repeat: RoundResult | None,
    expected: int,
) -> list[GateResult]:
    gates: list[GateResult] = []
    rounds = [cold, warm, shuffled] + ([repeat] if repeat else [])
    errors = sum(1 for result in rounds for record in result.records if record.error)
    paths = sum(1 for record in cold.records if record.path_names)
    gates.append(
        GateResult(
            "Robustness (crashes)",
            "pass" if errors == 0 else "fail",
            f"{errors} crash(es) across {sum(len(result.records) for result in rounds)} classifications",
        )
    )
    gates.append(
        GateResult(
            "Paths produced",
            "pass" if paths == expected else "fail",
            f"{paths}/{expected} cold classifications produced a path",
        )
    )
    max_depth = max((record.depth for record in cold.records), default=0)
    gates.append(
        GateResult(
            "Max depth",
            "pass" if all(record.depth <= 4 for record in cold.records if not record.error) else "fail",
            f"maximum observed depth {max_depth}; limit 4",
        )
    )
    deep = sum(1 for record in cold.records if record.depth >= 2)
    gates.append(
        _ratio_gate(
            "Hierarchy use",
            deep,
            expected,
            0.6,
            "paths with depth >= 2",
        )
    )
    exact_duplicates, near_pairs = _duplicate_roots(cold.roots)
    gates.append(
        GateResult(
            "Duplicate roots (exact)",
            "pass" if exact_duplicates == 0 else "fail",
            f"{exact_duplicates} exact duplicate root(s)",
        )
    )
    gates.append(
        GateResult(
            "Duplicate roots (cosine)",
            "pass" if near_pairs == 0 else "fail",
            f"{near_pairs} root pair(s) with cosine >= 0.90; allowed 0",
        )
    )
    orphans = {
        result.name: result.stats.get("orphan_fields", -1)
        for result in [cold, warm, shuffled]
    }
    gates.append(
        GateResult(
            "Orphans",
            "pass" if all(value == 0 for value in orphans.values()) else "fail",
            ", ".join(f"{name}={value}" for name, value in orphans.items()),
        )
    )
    gates.append(
        GateResult(
            "Runtime (cold)",
            "pass" if cold.elapsed_s <= COLD_RUNTIME_LIMIT_S else "fail",
            f"{cold.elapsed_s:.1f}s; limit {COLD_RUNTIME_LIMIT_S}s",
        )
    )
    gates.append(
        GateResult(
            "Runtime (warm)",
            "pass" if warm.elapsed_s <= WARM_RUNTIME_LIMIT_S else "fail",
            f"{warm.elapsed_s:.1f}s; limit {WARM_RUNTIME_LIMIT_S}s",
        )
    )
    warm_map = _record_map(warm)
    root_matches = sum(
        1
        for record in cold.records
        if record.slug in warm_map and _same_root(record, warm_map[record.slug])
    )
    path_matches = sum(
        1
        for record in cold.records
        if record.slug in warm_map and _same_path(record, warm_map[record.slug])
    )
    gates.append(_ratio_gate("Warm stability (root)", root_matches, expected, 0.9, "same root"))
    gates.append(_ratio_gate("Warm stability (path)", path_matches, expected, 0.7, "identical path"))
    shuffled_map = _record_map(shuffled)
    order_matches = sum(
        1
        for record in cold.records
        if record.slug in shuffled_map and _same_root(record, shuffled_map[record.slug])
    )
    gates.append(
        _ratio_gate(
            "Order independence",
            order_matches,
            expected,
            0.8,
            "same root after reversal",
            blocking=False,
        )
    )
    if repeat is None:
        gates.append(
            GateResult(
                "Reproducibility",
                "skip",
                "not run; pass --repeat to classify a second cold pass",
                blocking=False,
            )
        )
    else:
        repeat_map = _record_map(repeat)
        identical = sum(
            1
            for record in cold.records
            if record.slug in repeat_map and _same_path(record, repeat_map[record.slug])
        )
        gates.append(
            _ratio_gate("Reproducibility", identical, expected, 1.0, "identical repeated path")
        )
    return gates


def evaluate_naming(
    *,
    cold: RoundResult,
    entries: Sequence[EvalDataset],
) -> list[GateResult]:
    entry_map = {entry.slug: entry for entry in entries}
    violations: list[str] = []
    banned_roots = {
        "miscellaneous",
        "misc",
        "other",
        "others",
        "uncategorized",
        "uncategorised",
        "general",
    }
    for record in cold.records:
        entry = entry_map[record.slug]
        if _normalize(record.root_name) in banned_roots:
            violations.append(f"{record.slug}: banned root {record.root_name!r}")
        for node in record.path_names:
            for label in entry.forbidden_labels:
                if _contains_token_sequence(node, label):
                    violations.append(f"{record.slug}: {node!r} contains {label!r}")
    return [
        GateResult(
            "Naming (automated)",
            "pass" if not violations else "fail",
            "no dataset proper nouns or banned roots"
            if not violations
            else "; ".join(violations[:8]),
        )
    ]


def evaluate_human(
    *,
    cold: RoundResult,
    scores: dict[str, HumanScore],
    expected: int,
    assume_automated_naming: bool = False,
) -> list[GateResult]:
    scored = [scores[record.slug] for record in cold.records if record.slug in scores]
    if len(scored) < expected:
        missing = expected - len(scored)
        pending = [
            GateResult(name, "pending", f"{len(scored)}/{expected} datasets scored; {missing} missing")
            for name in (
                "Top-level plausibility",
                "Full-path plausibility",
                "Naming (human)",
                "Sibling distinctness",
            )
        ]
        return pending
    root_scores = [score.root for score in scored]
    path_scores = [score.path for score in scored]
    naming_values = [score.naming for score in scored]
    assumed_naming = 0
    if assume_automated_naming:
        naming_scores = [
            value if value is not None else HUMAN_NAMING_ASSUMED_SCORE
            for value in naming_values
        ]
        assumed_naming = sum(1 for value in naming_values if value is None)
    else:
        naming_scores = [value for value in naming_values if value is not None]
    distinct = sum(1 for score in scored if score.sibling_distinctness)
    gates: list[GateResult] = []
    good_roots = sum(1 for value in root_scores if value >= HUMAN_MIN_ACCEPTABLE_SCORE)
    required_roots = max(1, math.ceil(HUMAN_PASS_FRACTION * expected))
    gates.append(
        GateResult(
            "Top-level plausibility",
            "pass"
            if good_roots >= required_roots
            and sum(root_scores) / len(root_scores) >= HUMAN_MEAN_THRESHOLD
            else "fail",
            f"{good_roots}/{expected} roots >= {HUMAN_MIN_ACCEPTABLE_SCORE} "
            f"(required {required_roots}); "
            f"mean {sum(root_scores) / len(root_scores):.2f} "
            f"(required {HUMAN_MEAN_THRESHOLD:.2f})",
        )
    )
    good_paths = sum(1 for value in path_scores if value >= HUMAN_MIN_ACCEPTABLE_SCORE)
    required_paths = max(1, math.ceil(HUMAN_PASS_FRACTION * expected))
    gates.append(
        GateResult(
            "Full-path plausibility",
            "pass"
            if good_paths >= required_paths
            and sum(path_scores) / len(path_scores) >= HUMAN_MEAN_THRESHOLD
            else "fail",
            f"{good_paths}/{expected} paths >= {HUMAN_MIN_ACCEPTABLE_SCORE} "
            f"(required {required_paths}); "
            f"mean {sum(path_scores) / len(path_scores):.2f} "
            f"(required {HUMAN_MEAN_THRESHOLD:.2f})",
        )
    )
    if not naming_scores or len(naming_scores) < expected:
        missing = expected - len(naming_scores)
        gates.append(
            GateResult(
                "Naming (human)",
                "pending",
                f"{len(naming_scores)}/{expected} naming scores; {missing} missing "
                "(provide naming scores or allow the automated naming result)",
            )
        )
    else:
        naming_mean = sum(naming_scores) / len(naming_scores)
        detail = f"mean {naming_mean:.2f} (required {HUMAN_NAMING_MEAN_THRESHOLD:.2f})"
        if assumed_naming:
            detail += f"; {assumed_naming} assumed from automated naming"
        gates.append(
            GateResult(
                "Naming (human)",
                "pass" if naming_mean >= HUMAN_NAMING_MEAN_THRESHOLD else "fail",
                detail,
            )
        )
    required_distinct = max(1, math.ceil(HUMAN_PASS_FRACTION * expected))
    gates.append(
        GateResult(
            "Sibling distinctness",
            "pass" if distinct >= required_distinct else "fail",
            f"{distinct}/{expected} descriptions distinguish the field (required {required_distinct})",
        )
    )
    return gates


def overall_status(gates: Sequence[GateResult]) -> Literal["pass", "fail", "pending"]:
    blocking = [gate for gate in gates if gate.blocking]
    if any(gate.status == "fail" for gate in blocking):
        return "fail"
    if any(gate.status == "pending" for gate in blocking):
        return "pending"
    return "pass"


def load_scores(path: Path, entries: Sequence[EvalDataset]) -> dict[str, HumanScore]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    scores: dict[str, HumanScore] = {}
    for entry in entries:
        raw = payload.get(entry.slug)
        if not isinstance(raw, dict):
            continue
        values = [raw.get(key) for key in ("root", "path", "naming", "sibling_distinctness")]
        if all(value is None for value in values):
            continue
        required = ("root", "path", "sibling_distinctness")
        missing = [key for key in required if raw.get(key) is None]
        if missing:
            raise ValueError(
                f"{entry.slug}: provide root, path, and sibling_distinctness "
                "(naming is optional)"
            )
        for key in ("root", "path"):
            value = raw[key]
            if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 5:
                raise ValueError(f"{entry.slug}.{key} must be an integer from 1 to 5")
        naming = raw.get("naming")
        if naming is not None and (
            not isinstance(naming, int)
            or isinstance(naming, bool)
            or not 1 <= naming <= 5
        ):
            raise ValueError(f"{entry.slug}.naming must be an integer from 1 to 5 or null")
        sibling = raw["sibling_distinctness"]
        if not isinstance(sibling, bool):
            raise ValueError(f"{entry.slug}.sibling_distinctness must be true or false")
        scores[entry.slug] = HumanScore(
            root=raw["root"],
            path=raw["path"],
            naming=naming,
            sibling_distinctness=sibling,
            notes=str(raw.get("notes") or ""),
        )
    return scores


def write_scores_template(path: Path, entries: Sequence[EvalDataset]) -> None:
    payload: dict[str, Any] = {
        "_instructions": (
            "Score each dataset using the 1-5 rubric. Fill root and path with "
            "integers from 1 to 5 and sibling_distinctness with true/false. "
            "Naming is optional: leave it null to use the automated naming result "
            f"when that gate passes (assumed score {HUMAN_NAMING_ASSUMED_SCORE}); "
            f"provide 1-5 to override it. A submission passes when {HUMAN_PASS_CRITERIA}. "
            "Leave root, path, and sibling_distinctness null to mark a dataset as "
            "not yet scored."
        ),
        "_rubric": {
            "5": "Named as a practitioner would; parent-child relation obvious; description accurate",
            "4": "Minor naming quibble; relation clear",
            "3": "Acceptable, but a different path is also reasonable",
            "2": "A node is misleading, too broad, or too narrow",
            "1": "Nonsensical or dataset-specific label",
        },
    }
    for entry in entries:
        payload[entry.slug] = {
            "root": None,
            "path": None,
            "naming": None,
            "sibling_distinctness": None,
            "notes": "",
            "reference": {
                "acceptable_roots": list(entry.acceptable_roots),
                "good_path": entry.good_path,
            },
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _path_text(record: DatasetRun) -> str:
    return " > ".join(record.path_names) if record.path_names else "(none)"


def render_markdown(
    *,
    generated_at: str,
    settings: Settings,
    commit: str,
    mode: str,
    entries: Sequence[EvalDataset],
    cold: RoundResult,
    warm: RoundResult,
    shuffled: RoundResult,
    repeat: RoundResult | None,
    gates: Sequence[GateResult],
    scores: dict[str, HumanScore],
    json_path: Path,
    assume_automated_naming: bool = False,
) -> str:
    status = overall_status(gates)
    lines = [
        "# Field classification evaluation",
        "",
        f"Generated: {generated_at}",
        f"Commit: `{commit}`",
        f"Mode: {mode}",
        f"LLM: `{settings.llm_provider}` / `{settings.llm_model_name}`",
        f"Embeddings: `{settings.embed_provider}` / `{settings.embed_model_name}` "
        f"at {settings.embed_dims} dimensions",
        f"Datasets: {len(entries)}",
        f"Overall status: **{status.upper()}**",
        "",
        "## Gates",
        "",
        "| Gate | Status | Detail |",
        "|---|---|---|",
    ]
    for gate in gates:
        suffix = "" if gate.blocking else " (diagnostic)"
        lines.append(
            f"| {_escape(gate.name)}{suffix} | {gate.status} | {_escape(gate.detail)} |"
        )
    lines.extend(
        [
            "",
            "## Runs",
            "",
        ]
    )
    for result in [cold, warm, shuffled] + ([repeat] if repeat else []):
        lines.extend(
            [
                f"### {result.name} ({result.elapsed_s:.1f}s)",
                "",
                "| Dataset | Root | Full path | Depth | Confidence | Created | Time |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        for record in result.records:
            lines.append(
                f"| {record.slug} | {_escape(record.root_name or '(none)')} | "
                f"{_escape(_path_text(record))} | {record.depth} | "
                f"{record.confidence:.3f} | {len(record.created_fields)} | "
                f"{record.elapsed_s:.1f}s |"
            )
        lines.append("")
    lines.extend(
        [
            "## Human scores",
            "",
        ]
    )
    if scores:
        lines.extend(
            [
                "| Dataset | Root | Path | Naming | Sibling distinctness | Notes |",
                "|---|---|---|---|---|---|",
            ]
        )
        for entry in entries:
            score = scores.get(entry.slug)
            if score is None:
                lines.append(f"| {entry.slug} | - | - | - | - | not scored |")
            else:
                if score.naming is None:
                    naming_text = "auto" if assume_automated_naming else "missing"
                else:
                    naming_text = str(score.naming)
                lines.append(
                    f"| {entry.slug} | {score.root} | {score.path} | {naming_text} | "
                    f"{'yes' if score.sibling_distinctness else 'no'} | "
                    f"{_escape(score.notes)} |"
                )
        if any(score.naming is None for score in scores.values()):
            if assume_automated_naming:
                lines.append(
                    "`auto` means the automated naming gate was trusted for that dataset."
                )
            else:
                lines.append(
                    "`missing` means no human naming score was supplied and the "
                    "automated naming result was not trusted."
                )
        lines.append("")
    else:
        lines.extend(
            [
                "No human scores supplied. Run `--write-scores-template` to create a "
                "template, fill it in, and pass it with `--scores`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Warnings",
            "",
        ]
    )
    warning_lines: list[str] = []
    for result in [cold, warm, shuffled] + ([repeat] if repeat else []):
        for record in result.records:
            if record.error:
                warning_lines.append(f"- {result.name} / {record.slug}: ERROR {record.error}")
            for warning in record.warnings:
                warning_lines.append(f"- {result.name} / {record.slug}: {warning}")
    if warning_lines:
        lines.extend(warning_lines[:100])
        if len(warning_lines) > 100:
            lines.append(f"- ... {len(warning_lines) - 100} more")
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Method",
            "",
            "- Cold: empty field store, fixed dataset order, no research cache.",
            "- Warm: the same field store and repeated dataset inputs. A dataset whose "
            "profile is unchanged reuses its committed path, so warm runs do not call the "
            "provider.",
            "- Shuffled cold: fresh field store with the dataset order reversed; diagnostic only.",
            "- Repeat cold: optional fresh field store with the cold order, used for the "
            "reproducibility gate.",
            "- Human scores are supplied separately; the harness never invents them.",
            f"- Machine-readable results: `{_display_path(json_path)}`.",
            "",
            "## Dataset registry",
            "",
            "| Dataset | Domain | Target | Description | Source |",
            "|---|---|---|---|---|",
        ]
    )
    for entry in entries:
        lines.append(
            f"| {entry.name} | {entry.domain} | {entry.target or '-'} | "
            f"{'yes' if entry.description else 'no'} | `{entry.filename}` |"
        )
    lines.append("")
    return "\n".join(lines)


def build_json(
    *,
    generated_at: str,
    settings: Settings,
    commit: str,
    mode: str,
    entries: Sequence[EvalDataset],
    cold: RoundResult,
    warm: RoundResult,
    shuffled: RoundResult,
    repeat: RoundResult | None,
    gates: Sequence[GateResult],
    scores: dict[str, HumanScore],
) -> dict[str, Any]:
    def round_payload(result: RoundResult) -> dict[str, Any]:
        return {
            "name": result.name,
            "elapsed_s": round(result.elapsed_s, 3),
            "stats": result.stats,
            "roots": [name for name, _ in result.roots],
            "records": [
                {
                    "slug": record.slug,
                    "path": record.path_names,
                    "field_ids": record.path_ids,
                    "depth": record.depth,
                    "confidence": round(record.confidence, 6),
                    "created_fields": len(record.created_fields),
                    "warnings": record.warnings,
                    "error": record.error,
                    "elapsed_s": round(record.elapsed_s, 3),
                }
                for record in result.records
            ],
        }

    return {
        "schema_version": "1.0.0",
        "generated_at": generated_at,
        "commit": commit,
        "mode": mode,
        "provider": {
            "llm": settings.llm_provider,
            "llm_model": settings.llm_model_name,
            "embeddings": settings.embed_provider,
            "embedding_model": settings.embed_model_name,
            "embed_dims": settings.embed_dims,
        },
        "datasets": [
            {
                "slug": entry.slug,
                "name": entry.name,
                "domain": entry.domain,
                "target": entry.target,
                "description_supplied": bool(entry.description),
                "acceptable_roots": list(entry.acceptable_roots),
                "good_path": entry.good_path,
            }
            for entry in entries
        ],
        "status": overall_status(gates),
        "gates": [
            {
                "name": gate.name,
                "status": gate.status,
                "detail": gate.detail,
                "blocking": gate.blocking,
            }
            for gate in gates
        ],
        "runs": {
            "cold": round_payload(cold),
            "warm": round_payload(warm),
            "shuffled": round_payload(shuffled),
            **({"repeat": round_payload(repeat)} if repeat else {}),
        },
        "human_scores": {
            slug: {
                "root": score.root,
                "path": score.path,
                "naming": score.naming,
                "sibling_distinctness": score.sibling_distinctness,
                "notes": score.notes,
            }
            for slug, score in scores.items()
        },
    }


def _parse_datasets(value: str | None) -> list[EvalDataset]:
    if not value:
        return list(DATASETS)
    slugs = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [slug for slug in slugs if slug not in DATASETS_BY_SLUG]
    if unknown:
        raise ValueError(
            f"unknown dataset slug(s) {unknown}; choose from {sorted(DATASETS_BY_SLUG)}"
        )
    return [DATASETS_BY_SLUG[slug] for slug in slugs]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the field-classification evaluation gate."
    )
    parser.add_argument("--mock", action="store_true", help="use deterministic mock providers")
    parser.add_argument(
        "--datasets",
        help="comma-separated slugs (default: all ten)",
    )
    parser.add_argument("--scores", type=Path, help="human score JSON file")
    parser.add_argument(
        "--write-scores-template",
        type=Path,
        help="write a score template and exit",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--repeat",
        action="store_true",
        help="run a second cold pass for the reproducibility gate",
    )
    parser.add_argument(
        "--allow-pending",
        action="store_true",
        help="exit 0 when only human-score gates are pending",
    )
    parser.add_argument(
        "--no-assume-automated-naming",
        action="store_true",
        help="require human naming scores even when the automated naming gate passes",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        entries = _parse_datasets(args.datasets)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.write_scores_template:
        write_scores_template(args.write_scores_template, entries)
        print(f"wrote score template to {args.write_scores_template}")
        return 0

    settings = Settings.from_env(mock=args.mock)
    if not settings.mock:
        required_keys: list[tuple[str, str | None]] = []
        if settings.llm_provider == "deepseek":
            required_keys.append(("DEEPSEEK_API_KEY", settings.deepseek_api_key))
        if settings.llm_provider == "gemini" or settings.embed_provider == "gemini":
            required_keys.append(("GEMINI_API_KEY", settings.gemini_api_key))
        missing = [name for name, value in required_keys if not value]
        if missing:
            print(
                f"error: {', '.join(missing)} required for a live run; "
                "use --mock for a dry run",
                file=sys.stderr,
            )
            return 2

    scores: dict[str, HumanScore] = {}
    if args.scores:
        if not args.scores.is_file():
            print(f"error: score file not found: {args.scores}", file=sys.stderr)
            return 2
        try:
            scores = load_scores(args.scores, entries)
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"error: invalid score file: {exc}", file=sys.stderr)
            return 2

    data_dir = args.data_dir.resolve()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    generated_at = _now()
    mode = "mock" if settings.mock else "live"
    print(f"mode={mode} datasets={len(entries)} work_dir={work_dir}")

    try:
        inputs = load_inputs(entries, data_dir)
    except (IngestError, RuntimeError, OSError, httpx.HTTPError) as exc:
        print(f"error: could not load datasets: {exc}", file=sys.stderr)
        return 1

    llm = build_llm_client(settings)
    embeddings = build_embedding_client(settings)
    try:
        cold = run_round(
            name="Cold",
            inputs=inputs,
            settings=settings,
            llm=llm,
            embeddings=embeddings,
            store_path=work_dir / "cold" / "fields.db",
            fresh=True,
        )
        warm = run_round(
            name="Warm",
            inputs=inputs,
            settings=settings,
            llm=llm,
            embeddings=embeddings,
            store_path=work_dir / "cold" / "fields.db",
            fresh=False,
        )
        shuffled = run_round(
            name="Shuffled cold",
            inputs=list(reversed(inputs)),
            settings=settings,
            llm=llm,
            embeddings=embeddings,
            store_path=work_dir / "shuffled" / "fields.db",
            fresh=True,
        )
        repeat = None
        if args.repeat:
            repeat = run_round(
                name="Repeat cold",
                inputs=inputs,
                settings=settings,
                llm=llm,
                embeddings=embeddings,
                store_path=work_dir / "repeat" / "fields.db",
                fresh=True,
            )
    except (ClassificationError, FieldStoreError, LLMError, EmbeddingError) as exc:
        print(f"error: evaluation aborted: {exc}", file=sys.stderr)
        return 1
    finally:
        for client in (llm, embeddings):
            close = getattr(client, "close", None)
            if callable(close):
                close()

    gates = evaluate_automated(
        cold=cold,
        warm=warm,
        shuffled=shuffled,
        repeat=repeat,
        expected=len(entries),
    )
    naming_gates = evaluate_naming(cold=cold, entries=entries)
    gates.extend(naming_gates)
    automated_naming_passed = all(gate.status == "pass" for gate in naming_gates)
    assume_naming = not args.no_assume_automated_naming and automated_naming_passed
    gates.extend(
        evaluate_human(
            cold=cold,
            scores=scores,
            expected=len(entries),
            assume_automated_naming=assume_naming,
        )
    )
    status = overall_status(gates)

    json_path = args.json_output or args.output.with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(
            build_json(
                generated_at=generated_at,
                settings=settings,
                commit=_git_commit(),
                mode=mode,
                entries=entries,
                cold=cold,
                warm=warm,
                shuffled=shuffled,
                repeat=repeat,
                gates=gates,
                scores=scores,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_markdown(
            generated_at=generated_at,
            settings=settings,
            commit=_git_commit(),
            mode=mode,
            entries=entries,
            cold=cold,
            warm=warm,
            shuffled=shuffled,
            repeat=repeat,
            gates=gates,
            scores=scores,
            json_path=json_path,
            assume_automated_naming=assume_naming,
        ),
        encoding="utf-8",
    )
    print(f"status={status} results={args.output}")
    if status == "fail":
        return 1
    if status == "pending" and not args.allow_pending:
        print(
            "human scores are pending; rerun with --scores or --allow-pending",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
