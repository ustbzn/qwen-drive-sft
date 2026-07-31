#!/usr/bin/env python3
"""Remove JSON-listed log names from a NAVSIM log-split YAML file."""
from __future__ import annotations

import argparse
import json
import os
import re
import stat
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT_YAML = (
    REPO_ROOT / "navsim/planning/script/config/training/log_split/mogo_multi.yaml"
)
DEFAULT_REMOVE_JSON = REPO_ROOT / "need_rm_pkl.json"
DEFAULT_JSON_KEY = "recommended_conservative_source_cleanup"
SPLIT_KEYS = ("train_logs", "val_logs", "test_logs")

_SECTION_PATTERN = re.compile(
    rf"^(?P<key>{'|'.join(SPLIT_KEYS)}):[ \t]*(?:#.*)?(?:\r?\n)?$"
)
_LIST_ITEM_PATTERN = re.compile(
    r"^[ \t]+-[ \t]+(?P<value>[^ \t#\r\n]+)[ \t]*(?:#.*)?(?:\r?\n)?$"
)
_TOP_LEVEL_PATTERN = re.compile(r"^[^\s#]")


def normalize_log_name(value: str) -> str:
    """Return the basename used by NAVSIM log-split files."""
    name = Path(value).name
    return name.removesuffix(".pkl")


def load_removal_names(path: Path, json_key: str) -> set[str]:
    with path.open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)

    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    values = payload.get(json_key)
    if not isinstance(values, list):
        raise ValueError(f"{path} field {json_key!r} must be a list")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{path} field {json_key!r} must contain non-empty strings")

    names = {normalize_log_name(value.strip()) for value in values}
    if not names:
        raise ValueError(f"{path} field {json_key!r} is empty")
    return names


def remove_logs_from_yaml(
    content: str, removal_names: set[str]
) -> tuple[str, dict[str, list[str]]]:
    """Remove matching list items while preserving all other YAML text."""
    current_section: str | None = None
    seen_sections: set[str] = set()
    output_lines: list[str] = []
    removed = {key: [] for key in SPLIT_KEYS}

    for line in content.splitlines(keepends=True):
        section_match = _SECTION_PATTERN.fullmatch(line)
        if section_match:
            current_section = section_match.group("key")
            seen_sections.add(current_section)
            output_lines.append(line)
            continue

        if _TOP_LEVEL_PATTERN.match(line):
            current_section = None

        if current_section is not None:
            item_match = _LIST_ITEM_PATTERN.fullmatch(line)
            if item_match:
                log_name = normalize_log_name(item_match.group("value"))
                if log_name in removal_names:
                    removed[current_section].append(log_name)
                    continue

        output_lines.append(line)

    missing_sections = set(SPLIT_KEYS) - seen_sections
    if missing_sections:
        raise ValueError(
            "YAML is missing expected section(s): " + ", ".join(sorted(missing_sections))
        )
    return "".join(output_lines), removed


def atomic_write(path: Path, content: str) -> None:
    """Replace an existing file atomically while retaining its permission bits."""
    mode = stat.S_IMODE(path.stat().st_mode)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        os.chmod(temporary_name, mode)
        with os.fdopen(
            file_descriptor, "w", encoding="utf-8", newline=""
        ) as file_handle:
            file_handle.write(content)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def print_summary(
    removal_names: set[str], removed: dict[str, list[str]], write: bool
) -> None:
    removed_names = {name for names in removed.values() for name in names}
    print(f"Removal list: {len(removal_names)} unique log(s)")
    for split_key in SPLIT_KEYS:
        print(f"{split_key}: removed {len(removed[split_key])}")

    not_found = sorted(removal_names - removed_names)
    if not_found:
        print(f"Not found in YAML: {len(not_found)}")
        for name in not_found:
            print(f"  {name}")
    if not write:
        print("Dry run only; pass --write to update the YAML file.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-yaml", type=Path, default=DEFAULT_SPLIT_YAML)
    parser.add_argument("--remove-json", type=Path, default=DEFAULT_REMOVE_JSON)
    parser.add_argument("--json-key", default=DEFAULT_JSON_KEY)
    parser.add_argument(
        "--write",
        action="store_true",
        help="atomically overwrite --split-yaml; the default is a dry run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    removal_names = load_removal_names(args.remove_json, args.json_key)
    with args.split_yaml.open("r", encoding="utf-8", newline="") as file_handle:
        original_content = file_handle.read()
    updated_content, removed = remove_logs_from_yaml(original_content, removal_names)

    print_summary(removal_names, removed, args.write)
    if args.write:
        if updated_content == original_content:
            print(f"No changes needed: {args.split_yaml}")
        else:
            atomic_write(args.split_yaml, updated_content)
            print(f"Updated: {args.split_yaml}")


if __name__ == "__main__":
    main()
