from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_expected_top_level_directories_exist() -> None:
    expected_directories = [
        "Scripts",
        "Templates",
        "backend",
        "frontend",
    ]
    for directory in expected_directories:
        assert (REPO_ROOT / directory).is_dir(), f"Missing top-level directory: {directory}"


def test_expected_root_files_exist() -> None:
    expected_files = [
        "README.md",
        "requirements.txt",
        "environment.yml",
        "config.yaml",
    ]
    for filename in expected_files:
        assert (REPO_ROOT / filename).is_file(), f"Missing root file: {filename}"
