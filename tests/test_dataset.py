from __future__ import annotations

import json
from pathlib import Path

import pytest

from trackrefinery.dataset import DatasetContractError, InferenceDataset
from trackrefinery.synthetic import CASES, generate_dataset
from trackrefinery.targets import TargetDataset


def test_generated_inputs_and_targets_are_separate(tmp_path: Path) -> None:
    generated = generate_dataset(tmp_path / "synthetic")
    inference = InferenceDataset.open(generated.inference_root)
    targets = TargetDataset.open(generated.target_root)

    assert inference.dataset_id == targets.dataset_id == "synthetic-v1"
    assert len(inference.sequences) == 6
    assert len(inference.cases) == 7
    assert len(targets.entries) == 7
    assert sum(len(case.frames) for case in inference.validate()) == 31


def test_inference_validation_never_opens_targets(tmp_path: Path) -> None:
    generated = generate_dataset(tmp_path / "synthetic")
    (generated.target_root / "targetset.json").write_text(
        "not valid JSON", encoding="utf-8"
    )

    cases = InferenceDataset.open(generated.inference_root).validate()

    assert len(cases) == 7


def test_shared_sequence_frames_are_reused_without_copy(tmp_path: Path) -> None:
    generated = generate_dataset(tmp_path / "synthetic")
    dataset = InferenceDataset.open(generated.inference_root)
    first = dataset.load_case("neighboring_tracks")
    second = dataset.load_case("neighboring_tracks_neighbor")

    assert first.frames[0] is second.frames[0]
    assert first.frames[0].points_xyz is second.frames[0].points_xyz


def test_reader_rejects_path_escape(tmp_path: Path) -> None:
    generated = generate_dataset(tmp_path / "synthetic")
    dataset_path = generated.inference_root / "dataset.json"
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    payload["sequences"][0]["manifest_path"] = "../outside.json"
    dataset_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DatasetContractError, match="remain inside"):
        InferenceDataset.open(generated.inference_root)


def test_committed_fixture_is_valid() -> None:
    root = Path(__file__).parent / "fixtures" / "synthetic_v1"

    assert len(InferenceDataset.open(root / "inference").validate()) == 7
    assert len(TargetDataset.open(root / "targets").entries) == 7
    assert len(CASES) == 6
