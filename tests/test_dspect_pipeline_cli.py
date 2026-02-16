from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DSPECT_DIR = REPO_ROOT / "Scripts" / "Preprocessing" / "DSPECT"


def load_module(module_name: str, file_path: Path):
    """Load a Python module directly from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_parser_accepts_is_hazel_alias() -> None:
    sys.path.insert(0, str(DSPECT_DIR))
    module = load_module("dspect_run_pipeline", DSPECT_DIR / "run_pipeline.py")
    parser = module.build_parser()
    args = parser.parse_args(["--diagnosis", "CN", "--isHasel"])
    assert args.is_hazel is True


def test_build_command_adds_shape_and_mask_args() -> None:
    sys.path.insert(0, str(DSPECT_DIR))
    module = load_module("dspect_run_pipeline_cmd", DSPECT_DIR / "run_pipeline.py")
    parser = module.build_parser()
    args = parser.parse_args(["--diagnosis", "PD", "--mask_type", "occipital", "--shape", "91", "109", "91"])
    step = {"script": "4_masking.py", "description": "Step 4"}
    command = module.build_command(step, args, DSPECT_DIR)
    assert command[:4] == ["python3", "4_masking.py", "--diagnosis", "PD"]
    assert "--mask_type" in command
    assert "occipital" in command

