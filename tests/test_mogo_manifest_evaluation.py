from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import warnings

import pandas as pd

from scripts.evaluation.run_mogo_manifest_evaluation import (
    default_output_paths,
    merge_csvs,
    write_evaluation_summary,
)


def _write_legacy_result(path: Path, tokens: list[str], scores: list[float]) -> None:
    frame = pd.DataFrame(
        {
            "token": [*tokens, "average"],
            "valid": [True] * (len(tokens) + 1),
            "score": [*scores, sum(scores) / len(scores)],
            "rank": [*range(1, len(tokens) + 1), 0],
            "trajectory_path": [""] * (len(tokens) + 1),
        }
    )
    frame.to_csv(path)


def test_merge_preserves_all_scenarios_and_writes_summary(tmp_path: Path) -> None:
    first_csv = tmp_path / "first.csv"
    second_csv = tmp_path / "second.csv"
    _write_legacy_result(first_csv, ["a", "b"], [0.2, 0.4])
    _write_legacy_result(second_csv, ["c", "d"], [0.6, 0.8])

    datasets = [
        (SimpleNamespace(name="first", city="beijing"), first_csv),
        (SimpleNamespace(name="second", city="shanghai"), second_csv),
    ]
    merged_path = tmp_path / "experiment" / "merged_results.csv"
    summary_path = merged_path.parent / "evaluation_summary.md"

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        merged = merge_csvs(datasets, merged_path)
    write_evaluation_summary(
        merged=merged,
        summary_path=summary_path,
        merged_csv_path=merged_path,
        experiment_name="experiment",
        checkpoint="checkpoint.ckpt",
        manifest=Path("manifest.yaml"),
        generated_at=datetime(2026, 7, 17, 5, 11, 21),
    )

    saved = pd.read_csv(merged_path)
    assert set(saved["token"]) == {"a", "b", "c", "d", "average"}
    assert "Unnamed: 0" not in saved.columns
    assert saved.loc[saved["token"] == "average", "score"].item() == 0.5

    summary = summary_path.read_text(encoding="utf-8")
    assert "Total scenarios: 4" in summary
    assert "Overall average score: `0.500000000000`" in summary
    assert "| first | beijing | 2 | 2 | 0 | 0.300000000000 |" in summary
    assert "| second | shanghai | 2 | 2 | 0 | 0.700000000000 |" in summary


def test_default_output_paths_use_a_per_run_directory(tmp_path: Path) -> None:
    merged_path, summary_path = default_output_paths(tmp_path, "experiment_xx", "2026.07.17.05.11.21")

    run_dir = tmp_path / "experiment_xx_2026.07.17.05.11.21"
    assert merged_path == run_dir / "merged_results.csv"
    assert summary_path == run_dir / "evaluation_summary.md"
