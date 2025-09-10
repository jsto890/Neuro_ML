#!/usr/bin/env python3
import re
import subprocess
from pathlib import Path

# ─── CONFIG ────────────────────────────────────────────────────────────────────
DATASETS = {
    "ADNI": Path("~/reseng202500013-ndd-ml/data/raw/PET/ADNI/AD/ADNI_PET_AD"),
}
DEST_ROOT = Path("~/reseng202500013-ndd-ml/data/raw")
DICOM_EXT  = ".dcm"
# Regex to catch “site_modality_diagnosis” or with “_1” suffix
SMD_RE     = re.compile(r"^([^_]+)_([^_]+)_([^_]+)(?:_\d+)?$")
# ────────────────────────────────────────────────────────────────────────────────

# Expand user (~) in configured paths
DATASETS = {k: Path(str(v)).expanduser() for k, v in DATASETS.items()}
DEST_ROOT = Path(str(DEST_ROOT)).expanduser()

def find_subject_dirs(root: Path):
    """Yield any folder that contains DICOM files directly under it."""
    for d in root.rglob("*"):
        if d.is_dir() and any(p.suffix.lower() == DICOM_EXT for p in d.iterdir()):
            yield d

def extract_smd(folder: Path):
    """
    Walk up from `folder` until we find a parent whose name matches SMD_RE.
    Returns (site, modality, diagnosis).
    """
    for anc in folder.parents:
        m = SMD_RE.match(anc.name)
        if m:
            return m.group(1), m.group(2), m.group(3)
    raise RuntimeError(f"Could not find site_modality_diagnosis for {folder}")

def convert_dicom(dicom_dir: Path, out_dir: Path, prefix: str):
    """Run dcm2niix to convert and name the outputs with our prefix."""
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "dcm2niix",
        "-b", "y",                     # write JSON sidecar
        "-z", "y",                     # gzip
        "-f", prefix,                  # filename prefix
        "-o", str(out_dir),
        str(dicom_dir)
    ], check=True)

def main():
    for ds_name, ds_root in DATASETS.items():
        if not ds_root.exists():
            print(f"[!] Missing: {ds_root}")
            continue

        for subj_dir in find_subject_dirs(ds_root):
            site, modality, diagnosis = extract_smd(subj_dir)
            subject_id = subj_dir.name

            out_prefix = f"sub-{subject_id}_{site}_{modality}_{diagnosis}"
            dest_folder = (
                DEST_ROOT
                / modality
                / site
                / diagnosis
                / out_prefix
            )

            print(f"Converting {ds_name}/{subj_dir.relative_to(ds_root)} → {dest_folder}")
            convert_dicom(subj_dir, dest_folder, out_prefix)

    print("All done.")

if __name__ == "__main__":
    main()
