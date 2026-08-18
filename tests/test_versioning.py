"""Dataset versioning (§M3 gate). Without this, reproducibility is a claim."""

from __future__ import annotations

from pathlib import Path

import pytest

from data.store.versioning import DatasetVersion, compute_version


@pytest.fixture
def lake(tmp_path) -> Path:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "one.parquet").write_bytes(b"first")
    (tmp_path / "a" / "two.parquet").write_bytes(b"second")
    (tmp_path / "notes.txt").write_bytes(b"ignored")
    return tmp_path


class TestStability:
    def test_same_bytes_same_hash(self, lake):
        assert compute_version(lake).content_hash == compute_version(lake).content_hash

    def test_hash_is_content_not_mtime(self, lake):
        before = compute_version(lake)
        # Rewrite identical bytes; mtime changes, content does not.
        (lake / "a" / "one.parquet").write_bytes(b"first")
        assert compute_version(lake).content_hash == before.content_hash

    def test_pattern_filters_non_data_files(self, lake):
        assert compute_version(lake).file_count == 2

    def test_totals_reported(self, lake):
        version = compute_version(lake)
        assert version.total_bytes == len(b"first") + len(b"second")

    def test_short_form(self, lake):
        assert len(compute_version(lake).short) == 12


class TestChangeDetection:
    def test_modified_file_changes_hash(self, lake):
        before = compute_version(lake)
        (lake / "a" / "one.parquet").write_bytes(b"CORRECTED")
        assert compute_version(lake).content_hash != before.content_hash

    def test_added_file_changes_hash(self, lake):
        before = compute_version(lake)
        (lake / "a" / "three.parquet").write_bytes(b"third")
        assert compute_version(lake).content_hash != before.content_hash

    def test_removed_file_changes_hash(self, lake):
        before = compute_version(lake)
        (lake / "a" / "two.parquet").unlink()
        assert compute_version(lake).content_hash != before.content_hash

    def test_moved_file_changes_hash(self, lake):
        # Partition layout is part of what a reader depends on.
        before = compute_version(lake)
        (lake / "b").mkdir()
        (lake / "a" / "one.parquet").rename(lake / "b" / "one.parquet")
        assert compute_version(lake).content_hash != before.content_hash

    def test_differs_from_names_the_changed_file(self, lake):
        before = compute_version(lake)
        (lake / "a" / "one.parquet").write_bytes(b"CORRECTED")
        assert compute_version(lake).differs_from(before) == ["a/one.parquet"]

    def test_differs_from_names_additions(self, lake):
        before = compute_version(lake)
        (lake / "a" / "new.parquet").write_bytes(b"x")
        assert compute_version(lake).differs_from(before) == ["a/new.parquet"]

    def test_identical_versions_have_no_differences(self, lake):
        assert compute_version(lake).differs_from(compute_version(lake)) == []


class TestFailureModes:
    def test_missing_root_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            compute_version(tmp_path / "nope")

    def test_empty_dataset_raises(self, tmp_path):
        # An empty dataset must not produce a valid-looking version (§14.1.5).
        with pytest.raises(FileNotFoundError, match="no files matching"):
            compute_version(tmp_path)


class TestSerialisation:
    def test_roundtrip(self, lake):
        original = compute_version(lake)
        restored = DatasetVersion.from_json(original.to_json())
        assert restored.content_hash == original.content_hash
        assert restored.files == original.files
        assert restored.file_count == original.file_count

    def test_json_is_stable(self, lake):
        version = compute_version(lake)
        assert version.to_json() == version.to_json()
