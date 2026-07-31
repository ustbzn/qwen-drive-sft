import json

from scripts.cache_dataset.remove_logs_from_split import (
    load_removal_names,
    remove_logs_from_yaml,
)


def test_load_removal_names_normalizes_paths_and_pkl_suffix(tmp_path):
    remove_json = tmp_path / "remove.json"
    remove_json.write_text(
        json.dumps({"logs": ["/data/log-a.pkl", "log-b", "log-a.pkl"]}),
        encoding="utf-8",
    )

    assert load_removal_names(remove_json, "logs") == {"log-a", "log-b"}


def test_remove_logs_from_all_splits_preserves_other_yaml_text():
    content = (
        "# generated file\n"
        "train_logs:\n"
        "  - keep-train\n"
        "  - remove-train\n"
        "\n"
        "val_logs:\n"
        "  - remove-val\n"
        "  - keep-val\n"
        "\n"
        "test_logs:\n"
        "  - keep-test\n"
    )

    updated, removed = remove_logs_from_yaml(
        content, {"remove-train", "remove-val", "not-present"}
    )

    assert updated == (
        "# generated file\n"
        "train_logs:\n"
        "  - keep-train\n"
        "\n"
        "val_logs:\n"
        "  - keep-val\n"
        "\n"
        "test_logs:\n"
        "  - keep-test\n"
    )
    assert removed == {
        "train_logs": ["remove-train"],
        "val_logs": ["remove-val"],
        "test_logs": [],
    }
