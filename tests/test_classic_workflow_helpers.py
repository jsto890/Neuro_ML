from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / "Scripts" / "Classic_Learning" / "complete_workflow.py"


def load_workflow_module():
    spec = importlib.util.spec_from_file_location("complete_workflow", WORKFLOW_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load complete_workflow.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validate_input_data_accepts_existing_csv(tmp_path: Path) -> None:
    module = load_workflow_module()
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("subject_id,label\ns1,0\n", encoding="utf-8")
    assert module.validate_input_data(str(csv_file)) is True


def test_validate_input_data_raises_for_missing_file(tmp_path: Path) -> None:
    module = load_workflow_module()
    missing = tmp_path / "missing.csv"
    try:
        module.validate_input_data(str(missing))
    except FileNotFoundError as exc:
        assert str(missing) in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for missing input path")


def test_load_config_falls_back_to_defaults() -> None:
    module = load_workflow_module()
    config = module.load_config(None)
    assert "data" in config
    assert "feature_engineering" in config
    assert config["data"]["binary_only"] is True

