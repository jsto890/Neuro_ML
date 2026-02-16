from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UTILS_PATH = REPO_ROOT / "Scripts" / "Preprocessing" / "DSPECT" / "pipeline_utils.py"


def load_utils_module():
    spec = importlib.util.spec_from_file_location("pipeline_utils", UTILS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load pipeline_utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_command_success() -> None:
    module = load_utils_module()
    ok = module.run_command(["python3", "-c", "print('ok')"], "echo command")
    assert ok is True


def test_run_command_failure() -> None:
    module = load_utils_module()
    ok = module.run_command(["python3", "-c", "import sys; sys.exit(3)"], "failing command")
    assert ok is False


def test_ensure_script_exists_raises_for_missing_file(tmp_path: Path) -> None:
    module = load_utils_module()
    missing = "no_such_script.py"
    try:
        module.ensure_script_exists(tmp_path, missing)
    except FileNotFoundError as exc:
        assert missing in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for missing script")

