from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DSPECT_DIR = REPO_ROOT / "Scripts" / "Preprocessing" / "DSPECT"


def load_pipeline_module():
    sys.path.insert(0, str(DSPECT_DIR))
    spec = importlib.util.spec_from_file_location("run_pipeline", DSPECT_DIR / "run_pipeline.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load run_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_step_config_reads_default_file() -> None:
    module = load_pipeline_module()
    config_path = DSPECT_DIR / "pipeline_config.yaml"
    steps = module.load_step_config(config_path)
    assert isinstance(steps, list)
    assert steps
    assert "script" in steps[0]

