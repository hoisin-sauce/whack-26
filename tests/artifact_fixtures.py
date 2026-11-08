"""Shared artifact fixtures for the preview and artifact tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from whack26.artifact import RunInfo, build_artifact
from whack26.classify import FieldPath
from whack26.critic.models import Check, CriticResult, Issue
from whack26.factors import Factor
from whack26.hypotheses.execute import ExecutionResult
from whack26.hypotheses.generate import HypothesisDraft, GeneratedHypothesis
from whack26.io import read_dataset
from whack26.profile import build_profile
from whack26.research.base import ResearchRecord
from whack26.research.pipeline import ResearchQuery, ResearchResult
from whack26.screen import build_screen
from whack26.store import Field

TIMESTAMP = "2026-11-08T10:00:00Z"


def make_field(identifier: str, *, level: int, parent: str | None, name: str) -> Field:
    return Field(
        id=identifier,
        parent_id=parent,
        level=level,
        name=name,
        description=f"{name} description.",
        description_embedding=[1.0, 0.0],
        centroid=None,
        member_count=1,
        created_at=TIMESTAMP,
        updated_at=TIMESTAMP,
    )


def make_artifact(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "region": ["north", "south", "north", "south", "north"],
            "y": [2.0, 4.0, 6.0, 8.0, 10.0],
        }
    )
    dataset_path = run_dir / "dataset.csv"
    frame.to_csv(dataset_path, index=False)
    loaded = read_dataset(dataset_path, target="y")
    profile = build_profile(
        loaded, target="y", description="An observational cross-sectional sample."
    )
    screen = build_screen(profile, loaded.frame)

    classification = FieldPath(
        fields=[
            make_field("fld_root", level=1, parent=None, name="Mock Domain"),
            make_field("fld_leaf", level=2, parent="fld_root", name="Regression"),
        ],
        confidence=0.81,
        created_fields=["fld_root", "fld_leaf"],
    )

    record = ResearchRecord(
        id="openalex:W1",
        source="openalex",
        source_id="W1",
        title="Income and prices",
        authors=["A. Author"],
        year=2024,
        doi="10.1000/example",
        url="https://example.test/W1",
        retrieved_at=TIMESTAMP,
        query="income prices",
        relevance=0.9,
    )
    research = ResearchResult(
        records=[record],
        queries=[
            ResearchQuery(
                source="openalex",
                query="income prices",
                rationale="A domain query.",
            )
        ],
        warnings=["research_cache_miss"],
        keywords=["income"],
        sources_used=["openalex"],
        cache={"hits": 1, "misses": 0},
    )

    factor = Factor(
        id="fac_aaaaaaaaaaaa",
        name="household income",
        description="Median household income.",
        direction="positive",
        claim_type="association",
        evidence_ids=["openalex:W1"],
        in_dataset=True,
        mapped_column="x",
        mapping_method="exact",
        mapping_confidence=0.91,
        rationale="The x column is income.",
    )

    hypotheses: list[GeneratedHypothesis] = []
    critics: list[CriticResult] = []
    verdicts = ["pass", "refine", "insufficient"]
    for index, family in enumerate(
        ("linear_model", "group_comparison", "interaction_or_derived")
    ):
        code = (
            "import pandas as pd\n\n"
            "def applies(df: pd.DataFrame) -> bool:\n"
            "    return len(df) > 0\n\n"
            "def run(df: pd.DataFrame, seed: int = 0) -> dict:\n"
            "    return {'predictions': [0.0] * len(df), 'target': 'y'}\n"
        )
        draft = HypothesisDraft(
            title=f"{family} hypothesis",
            rationale=(
                "No factor applies; this baseline is justified from domain "
                "knowledge."
            ),
            assumptions=["The conditional mean is approximately linear."],
            required_columns=["x"],
            optional_columns=[],
            min_rows=5,
            claim_type="prediction",
            factors_used=[],
            source_ids=[],
            code=code,
            target="y",
        )
        identifier = f"hyp_00000000000{index}"
        code_path = f"hypotheses/{identifier}.py"
        predictions_path = f"predictions/{identifier}.csv"
        (run_dir / "hypotheses").mkdir(exist_ok=True)
        (run_dir / "predictions").mkdir(exist_ok=True)
        (run_dir / code_path).write_text(code, encoding="utf-8")
        pd.DataFrame(
            {
                "row_index": list(range(len(frame))),
                "prediction": [0.0] * len(frame),
            }
        ).to_csv(run_dir / predictions_path, index=False)
        hypotheses.append(
            GeneratedHypothesis(
                id=identifier,
                method_family=family,  # type: ignore[arg-type]
                draft=draft,
                code_sha256=hashlib.sha256(code.encode("utf-8")).hexdigest(),
                code_path=code_path,
                execution=ExecutionResult(
                    status="ok",
                    applies=True,
                    runtime_ms=25 + index,
                    n_predictions=len(frame),
                    target="y",
                    predictions_path=predictions_path,
                ),
                generation_history=[{"attempt": 1, "status": "ok", "error": None}],
                validation_warnings=[],
            )
        )
        critics.append(
            CriticResult(
                hypothesis_id=identifier,
                claim_type="prediction",
                verdict=verdicts[index],  # type: ignore[arg-type]
                issues=(
                    [
                        Issue(
                            chunk="S2-leakage",
                            severity="major",
                            finding="A split protocol is not described.",
                            evidence="split.preprocessing",
                            why="Preprocessing before a split leaks information.",
                            fix="Split first.",
                        )
                    ]
                    if index == 1
                    else []
                ),
                checks=[
                    Check(id="code.static", status="pass", severity="none")
                ],
                summary="A method review.",
                retrieved_chunks=["S0-how-to-use", "S2-leakage"],
                model="mock",
                corpus_version="1.0.0+aaaaaaaa",
            )
        )

    artifact = build_artifact(
        run=RunInfo(
            run_id="run_20261108T100000Z_aaaaaaaaaaaa",
            started_at=TIMESTAMP,
            finished_at=TIMESTAMP,
            mode="mock",
            seed=0,
            tool_version="0.1.0",
            models={"llm": "mock", "embedding": "mock@16"},
        ),
        profile=profile,
        classification=classification,
        research=research,
        factors=[factor],
        screen=screen,
        hypotheses=hypotheses,
        critics=critics,
        warnings=["demo_warning", "demo_warning", " "],
        corpus_version="1.0.0+aaaaaaaa",
        base_dir=tmp_path,
    )
    return artifact, run_dir, hypotheses
