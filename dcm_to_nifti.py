#!/usr/bin/env python3
import re
import subprocess
import logging
import shutil
from pathlib import Path

# ─── CONFIG ────────────────────────────────────────────────────────────────────
DATASETS = {
    "PPMI": Path("PPMI"),
    "ADNI": Path("ADNI"),
}
DEST_ROOT = Path("data/raw")
DICOM_EXT  = ".dcm"
# Regex to catch “site_modality_diagnosis” or with optional “_1” suffix
SMD_RE     = re.compile(r"^(?P<site>[^_]+)_(?P<modality>[^_]+)_(?P<diagnosis>[^_]+)(?:_\d+)?$")
# ────────────────────────────────────────────────────────────────────────────────


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def find_subject_dirs(root: Path):
    """Yield any folder that directly contains DICOM files."""
    for d in root.rglob("*"):
        if d.is_dir() and any(p.suffix.lower() == DICOM_EXT for p in d.iterdir()):
            yield d


def convert_dicom(dicom_dir: Path, out_dir: Path, prefix: str, skip_deletion: bool):
    """
    Run dcm2niix to convert DICOMs in dicom_dir to out_dir with prefix.
    Remove only subject-specific folders unless skip_deletion is True.
    """
    if out_dir.exists() and not skip_deletion:
        if out_dir.name.startswith('sub-'):
            logging.info(f"Removing existing subject output: {out_dir}")
            shutil.rmtree(out_dir)

    logging.info(f"Converting folder: {dicom_dir}")
    files = sorted(p.name for p in dicom_dir.iterdir() if p.suffix.lower() == DICOM_EXT)
    logging.info(f"  {len(files)} DICOM files: {files}")

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run([
            "dcm2niix", "-b", "y", "-z", "y", "-f", prefix,
            "-o", str(out_dir), str(dicom_dir)
        ], check=True)
        logging.info(f"Converted: {dicom_dir} → {out_dir}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error converting {dicom_dir}: {e}")
        return False


def main():
    setup_logging()
    logging.info("Starting DICOM → NIfTI batch conversion")

    success = fail = 0
    for ds_name, ds_root in DATASETS.items():
        if not ds_root.exists():
            logging.warning(f"Missing dataset root: {ds_root}")
            continue
        logging.info(f"Processing {ds_name} at {ds_root}")

        for subj_dir in find_subject_dirs(ds_root):
            rel = subj_dir.relative_to(ds_root).parts
            if len(rel) < 3:
                logging.warning(f"Path too shallow to extract SMD: {subj_dir}")
                fail += 1
                continue
            smd = rel[2]
            m = SMD_RE.match(smd)
            if not m:
                logging.warning(f"Skipping unmatched folder: {smd} in {subj_dir}")
                fail += 1
                continue

            site = m.group('site')
            modality = m.group('modality')
            diagnosis = m.group('diagnosis')
            subject_id = subj_dir.name
            prefix = f"sub-{subject_id}_{site}_{modality}_{diagnosis}"
            dest = DEST_ROOT / modality / site / diagnosis / prefix

            # Determine if this is a BL site (skip deletions but allow re-check)
            skip_del = (site.upper() == 'BL')
            if skip_del:
                logging.info(f"BL site detected, will preserve existing outputs: {dest}")

            # If a converted NIfTI already exists, skip conversion entirely
            if (dest / f"{prefix}.nii.gz").exists():
                logging.info(f"Skipping {subject_id}: output already exists at {dest}")
                continue

            logging.info(f"Preparing {subject_id} → {modality}/{site}/{diagnosis}")
            converted = convert_dicom(subj_dir, dest, prefix, skip_del)
            if converted:
                success += 1
            else:
                fail += 1

    logging.info(f"Done. {success} succeeded, {fail} failed.")

if __name__ == "__main__":
    main()

