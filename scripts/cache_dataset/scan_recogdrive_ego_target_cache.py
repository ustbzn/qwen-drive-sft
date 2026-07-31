#!/usr/bin/env python3
"""Read-only numeric scan of ReCogDrive ego sidecars and trajectory targets."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import multiprocessing as mp
import os
import pickle
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import torch  # Preload tensor pickling support before worker processes fork.


EGO_FILE = "recogdrive_ego_feature_v1.gz"
TARGET_FILE = "trajectory_target.gz"
EXPECTED_SHAPES = {
    "history_trajectory": (4, 3),
    "high_command_one_hot": (4,),
    "status_feature": (13,),
    "trajectory": (8, 3),
}
VALID_COMMANDS = {
    (1, 0, 0, 0),  # left
    (0, 1, 0, 0),  # straight
    (0, 0, 1, 0),  # right
    (1, 1, 0, 0),  # u-turn
    (0, 0, 0, 1),  # unknown
}
FP16_MAX = float(np.finfo(np.float16).max)
FP16_SQUARE_LIMIT = math.sqrt(FP16_MAX)


def _load(path: Path) -> Mapping[str, Any]:
    with gzip.open(path, "rb") as file_handle:
        value = pickle.load(file_handle)
    if not isinstance(value, Mapping):
        raise TypeError(f"expected mapping, got {type(value).__name__}")
    return value


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"expected numeric array, got dtype={array.dtype}")
    return array


def _empty_extrema() -> Dict[str, Any]:
    return {
        "min": math.inf,
        "max": -math.inf,
        "abs_max": -math.inf,
        "abs_max_token": None,
    }


def _record_issue(
    result: Dict[str, Any], reason: str, token: str, details: Dict[str, Any] | None = None
) -> None:
    result["reason_sample_counts"][reason] += 1
    examples = result["reason_examples"][reason]
    if len(examples) < result["example_limit"]:
        record: Dict[str, Any] = {"token": token}
        if details:
            record.update(details)
        examples.append(record)


def _update_extrema(result: Dict[str, Any], field: str, array: np.ndarray, token: str) -> None:
    finite = np.asarray(array, dtype=np.float64)[np.isfinite(array)]
    if finite.size == 0:
        return
    extrema = result["field_extrema"][field]
    minimum = float(finite.min())
    maximum = float(finite.max())
    abs_max = float(np.abs(finite).max())
    extrema["min"] = min(extrema["min"], minimum)
    extrema["max"] = max(extrema["max"], maximum)
    if abs_max > extrema["abs_max"]:
        extrema["abs_max"] = abs_max
        extrema["abs_max_token"] = token


def _check_array(
    result: Dict[str, Any], field: str, value: Any, token: str
) -> np.ndarray | None:
    try:
        array = _as_numpy(value)
    except Exception as exc:
        _record_issue(
            result,
            f"{field}_invalid_type",
            token,
            {"error": f"{type(exc).__name__}: {exc}"},
        )
        return None

    if tuple(array.shape) != EXPECTED_SHAPES[field]:
        _record_issue(
            result,
            f"{field}_shape_mismatch",
            token,
            {"shape": list(array.shape), "expected": list(EXPECTED_SHAPES[field])},
        )
        return None

    result["field_dtypes"][field][str(array.dtype)] += 1
    _update_extrema(result, field, array, token)
    if not bool(np.isfinite(array).all()):
        _record_issue(result, f"{field}_nonfinite", token)
    if bool((np.abs(array) > FP16_MAX).any()):
        finite_abs = np.abs(np.asarray(array, dtype=np.float64)[np.isfinite(array)])
        _record_issue(
            result,
            f"{field}_fp16_overflow",
            token,
            {"finite_abs_max": float(finite_abs.max()) if finite_abs.size else None},
        )
    return array


def _scan_log(log_path_text: str, example_limit: int) -> Dict[str, Any]:
    log_path = Path(log_path_text)
    result: Dict[str, Any] = {
        "log_name": log_path.name,
        "token_directories": 0,
        "ego_files_scanned": 0,
        "target_files_scanned": 0,
        "reason_sample_counts": Counter(),
        "reason_examples": defaultdict(list),
        "field_dtypes": defaultdict(Counter),
        "field_extrema": defaultdict(_empty_extrema),
        "example_limit": example_limit,
    }

    try:
        token_paths = [path for path in log_path.iterdir() if path.is_dir()]
    except Exception as exc:
        _record_issue(
            result,
            "log_directory_read_error",
            "",
            {"error": f"{type(exc).__name__}: {exc}"},
        )
        return result

    for token_path in token_paths:
        token = token_path.name
        result["token_directories"] += 1
        ego_path = token_path / EGO_FILE
        target_path = token_path / TARGET_FILE

        ego: Mapping[str, Any] | None = None
        target: Mapping[str, Any] | None = None
        if not ego_path.is_file():
            _record_issue(result, "missing_ego_file", token)
        else:
            try:
                ego = _load(ego_path)
                result["ego_files_scanned"] += 1
            except Exception as exc:
                _record_issue(
                    result,
                    "ego_read_error",
                    token,
                    {"error": f"{type(exc).__name__}: {exc}"},
                )

        if not target_path.is_file():
            _record_issue(result, "missing_target_file", token)
        else:
            try:
                target = _load(target_path)
                result["target_files_scanned"] += 1
            except Exception as exc:
                _record_issue(
                    result,
                    "target_read_error",
                    token,
                    {"error": f"{type(exc).__name__}: {exc}"},
                )

        history = command = status = trajectory = None
        if ego is not None:
            for field in ("history_trajectory", "high_command_one_hot", "status_feature"):
                if field not in ego:
                    _record_issue(result, f"missing_{field}", token)
            if "history_trajectory" in ego:
                history = _check_array(result, "history_trajectory", ego["history_trajectory"], token)
            if "high_command_one_hot" in ego:
                command = _check_array(
                    result, "high_command_one_hot", ego["high_command_one_hot"], token
                )
            if "status_feature" in ego:
                status = _check_array(result, "status_feature", ego["status_feature"], token)

        if target is not None:
            if "trajectory" not in target:
                _record_issue(result, "missing_trajectory", token)
            else:
                trajectory = _check_array(result, "trajectory", target["trajectory"], token)

        if history is not None and bool(np.isfinite(history).all()):
            max_distance = float(np.linalg.norm(history[:, :2], axis=-1).max())
            if max_distance > 100.0:
                _record_issue(
                    result,
                    "history_xy_distance_gt_100m",
                    token,
                    {"max_xy_distance_m": max_distance},
                )
            if bool((np.abs(history[:, 2]) > math.pi + 1e-4).any()):
                _record_issue(
                    result,
                    "history_heading_outside_pi",
                    token,
                    {"heading_abs_max": float(np.abs(history[:, 2]).max())},
                )

        if trajectory is not None and bool(np.isfinite(trajectory).all()):
            max_distance = float(np.linalg.norm(trajectory[:, :2], axis=-1).max())
            if max_distance > 100.0:
                _record_issue(
                    result,
                    "trajectory_xy_distance_gt_100m",
                    token,
                    {"max_xy_distance_m": max_distance},
                )
            if bool((np.abs(trajectory[:, 2]) > math.pi + 1e-4).any()):
                _record_issue(
                    result,
                    "trajectory_heading_outside_pi",
                    token,
                    {"heading_abs_max": float(np.abs(trajectory[:, 2]).max())},
                )
            normalized = np.stack(
                (
                    2.0 * (trajectory[:, 0] + 1.57) / 66.74 - 1.0,
                    2.0 * (trajectory[:, 1] + 19.68) / 42.0 - 1.0,
                    2.0 * (trajectory[:, 2] + 1.67) / 3.53 - 1.0,
                ),
                axis=-1,
            )
            normalized_abs_max = float(np.abs(normalized).max())
            _update_extrema(result, "normalized_trajectory", normalized, token)
            if normalized_abs_max > FP16_SQUARE_LIMIT:
                _record_issue(
                    result,
                    "normalized_trajectory_fp16_square_overflow_risk",
                    token,
                    {"normalized_abs_max": normalized_abs_max},
                )

        if status is not None and bool(np.isfinite(status).all()):
            status_abs_max = float(np.abs(status).max())
            if status_abs_max > 100.0:
                _record_issue(
                    result,
                    "status_abs_gt_100",
                    token,
                    {"status_abs_max": status_abs_max},
                )

        if command is not None and bool(np.isfinite(command).all()):
            is_binary = bool(np.all(np.isclose(command, 0.0) | np.isclose(command, 1.0)))
            command_code = tuple(int(round(float(value))) for value in command)
            if not is_binary or command_code not in VALID_COMMANDS:
                _record_issue(
                    result,
                    "invalid_command_code",
                    token,
                    {"command": command.astype(float).tolist()},
                )
            if status is not None and bool(np.isfinite(status[:4]).all()) and not bool(
                np.allclose(command, status[:4])
            ):
                _record_issue(
                    result,
                    "command_status_mismatch",
                    token,
                    {
                        "command": command.astype(float).tolist(),
                        "status_prefix": status[:4].astype(float).tolist(),
                    },
                )

    return result


def _iter_log_paths(cache_root: Path, max_logs: int | None) -> Iterable[Path]:
    paths = sorted(
        path
        for path in cache_root.iterdir()
        if path.is_dir() and path.name != "metadata" and not path.name.startswith(".")
    )
    return paths if max_logs is None else paths[:max_logs]


def _finite_extrema(extrema: Dict[str, Any]) -> Dict[str, Any]:
    output = dict(extrema)
    for key in ("min", "max", "abs_max"):
        if not math.isfinite(output[key]):
            output[key] = None
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_root", type=Path)
    parser.add_argument("--workers", type=int, default=min(96, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-logs", type=int)
    parser.add_argument("--example-limit-per-log", type=int, default=20)
    parser.add_argument("--example-limit-per-reason", type=int, default=2000)
    args = parser.parse_args()

    if not args.cache_root.is_dir():
        parser.error(f"cache root does not exist: {args.cache_root}")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.max_logs is not None and args.max_logs < 1:
        parser.error("--max-logs must be positive")

    log_paths = list(_iter_log_paths(args.cache_root, args.max_logs))
    started = time.time()
    aggregate_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    reason_logs: Dict[str, set[str]] = defaultdict(set)
    reason_examples: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    field_dtypes: Dict[str, Counter[str]] = defaultdict(Counter)
    field_extrema: Dict[str, Dict[str, Any]] = defaultdict(_empty_extrema)

    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=mp.get_context("fork"),
    ) as executor:
        futures = {
            executor.submit(_scan_log, str(path), args.example_limit_per_log): path.name
            for path in log_paths
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            log_name = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                reason_counts["worker_error"] += 1
                reason_logs["worker_error"].add(log_name)
                if len(reason_examples["worker_error"]) < args.example_limit_per_reason:
                    reason_examples["worker_error"].append(
                        {"log_name": log_name, "error": f"{type(exc).__name__}: {exc}"}
                    )
                continue

            aggregate_counts["logs_scanned"] += 1
            for key in ("token_directories", "ego_files_scanned", "target_files_scanned"):
                aggregate_counts[key] += int(result[key])
            for reason, count in result["reason_sample_counts"].items():
                reason_counts[reason] += int(count)
                reason_logs[reason].add(log_name)
                remaining = args.example_limit_per_reason - len(reason_examples[reason])
                if remaining > 0:
                    reason_examples[reason].extend(
                        {"log_name": log_name, **example}
                        for example in result["reason_examples"][reason][:remaining]
                    )
            for field, dtypes in result["field_dtypes"].items():
                field_dtypes[field].update(dtypes)
            for field, source in result["field_extrema"].items():
                target = field_extrema[field]
                if source["min"] < target["min"]:
                    target["min"] = source["min"]
                if source["max"] > target["max"]:
                    target["max"] = source["max"]
                if source["abs_max"] > target["abs_max"]:
                    target["abs_max"] = source["abs_max"]
                    target["abs_max_log"] = log_name
                    target["abs_max_token"] = source["abs_max_token"]

            if completed % 100 == 0 or completed == len(futures):
                elapsed = time.time() - started
                print(
                    f"logs={completed}/{len(futures)} tokens={aggregate_counts['token_directories']} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

    elapsed = time.time() - started
    report = {
        "schema_version": 1,
        "dataset": "zhuhai_predata",
        "read_only_scan": True,
        "scan_execution_complete": aggregate_counts["logs_scanned"] == len(log_paths)
        and reason_counts["worker_error"] == 0,
        "cache_root": str(args.cache_root),
        "criteria": {
            "expected_shapes": {key: list(value) for key, value in EXPECTED_SHAPES.items()},
            "nonfinite_is_anomalous": True,
            "fp16_abs_limit": FP16_MAX,
            "history_or_trajectory_xy_distance_gt_m": 100.0,
            "status_abs_gt": 100.0,
            "trajectory_heading_abs_gt_rad": math.pi,
            "normalized_trajectory_fp16_square_limit": FP16_SQUARE_LIMIT,
            "valid_command_codes": [list(value) for value in sorted(VALID_COMMANDS)],
        },
        "scan_scope": {
            "logs_discovered": len(log_paths),
            "logs_scanned": int(aggregate_counts["logs_scanned"]),
            "token_directories": int(aggregate_counts["token_directories"]),
            "ego_files_scanned": int(aggregate_counts["ego_files_scanned"]),
            "target_files_scanned": int(aggregate_counts["target_files_scanned"]),
            "workers": args.workers,
            "elapsed_seconds": elapsed,
            "max_logs": args.max_logs,
        },
        "reason_sample_counts": dict(sorted(reason_counts.items())),
        "reason_affected_log_counts": {
            reason: len(log_names) for reason, log_names in sorted(reason_logs.items())
        },
        "reason_affected_logs": {
            reason: sorted(log_names) for reason, log_names in sorted(reason_logs.items())
        },
        "reason_examples": dict(sorted(reason_examples.items())),
        "reason_examples_truncated": {
            reason: reason_counts[reason] > len(examples)
            for reason, examples in sorted(reason_examples.items())
        },
        "field_dtypes": {
            field: dict(sorted(dtypes.items())) for field, dtypes in sorted(field_dtypes.items())
        },
        "field_extrema": {
            field: _finite_extrema(extrema) for field, extrema in sorted(field_extrema.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file_handle:
        json.dump(report, file_handle, ensure_ascii=False, indent=2, allow_nan=False)
        file_handle.write("\n")
    print(f"report={args.output}", flush=True)
    return 0 if report["scan_execution_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
