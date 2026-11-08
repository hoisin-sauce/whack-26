"""Store behaviour under the WAL journal and multiple connections."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from whack26.store import FieldStore

DIMS = 2


def create_root(store: FieldStore, name: str) -> None:
    store.create_field(
        parent_id=None,
        name=name,
        description=f"{name} description.",
        description_embedding=[1.0, 0.0],
    )


def test_store_uses_wal_and_a_busy_timeout(tmp_path: Path) -> None:
    with FieldStore(tmp_path / "fields.db", dims=DIMS) as store:
        connection = store._connection()
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 5000


def test_two_connections_see_each_others_committed_fields(tmp_path: Path) -> None:
    path = tmp_path / "fields.db"
    with FieldStore(path, dims=DIMS) as first, FieldStore(path, dims=DIMS) as second:
        create_root(first, "Housing")
        create_root(second, "Health")

        assert {field.name for field in first.roots()} == {"Housing", "Health"}
        assert {field.name for field in second.roots()} == {"Housing", "Health"}

    with FieldStore(path, dims=DIMS) as reopened:
        assert {field.name for field in reopened.roots()} == {"Housing", "Health"}


def test_concurrent_writers_serialize_without_losing_fields(tmp_path: Path) -> None:
    path = tmp_path / "fields.db"
    with FieldStore(path, dims=DIMS):
        pass

    errors: list[BaseException] = []

    def writer(prefix: str) -> None:
        try:
            with FieldStore(path, dims=DIMS) as store:
                for index in range(5):
                    create_root(store, f"{prefix} {index}")
        except BaseException as exc:  # noqa: BLE001 - surfaced in the assertion
            errors.append(exc)

    threads = [
        threading.Thread(target=writer, args=("alpha",)),
        threading.Thread(target=writer, args=("beta",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert errors == []
    with FieldStore(path, dims=DIMS) as store:
        assert len(store.roots()) == 10
        assert len(store.all_fields()) == 10


def test_duplicate_root_names_are_rejected_across_connections(
    tmp_path: Path,
) -> None:
    from whack26.store import DuplicateFieldError

    path = tmp_path / "fields.db"
    with FieldStore(path, dims=DIMS) as first, FieldStore(path, dims=DIMS) as second:
        create_root(first, "Housing")
        with pytest.raises(DuplicateFieldError):
            create_root(second, "Housing")
