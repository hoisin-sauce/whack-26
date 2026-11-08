"""Content-derived identifiers shared across the pipeline."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Iterable


def normalized_name(value: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", value.lower())
    return " ".join(tokens)


def factor_id(dataset_id: str, name: str) -> str:
    material = f"{dataset_id}:{normalized_name(name)}"
    return "fac_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]


def hypothesis_id(code: str, factors_used: Iterable[str]) -> str:
    material = code + "\x00" + "\x00".join(sorted(factors_used))
    return "hyp_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]


def run_id(dataset_id: str, *, now: datetime | None = None) -> str:
    moment = now or datetime.now(timezone.utc)
    stamp = moment.strftime("%Y%m%dT%H%M%SZ")
    suffix = dataset_id[3:11] if dataset_id.startswith("ds_") else dataset_id[:8]
    return f"run_{stamp}_{suffix}"
