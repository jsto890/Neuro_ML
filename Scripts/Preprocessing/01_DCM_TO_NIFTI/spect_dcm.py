#!/usr/bin/env python3
import re
import json
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import pydicom

# ─── CONFIG ────────────────────────────────────────────────────────────────────
DATASETS = {
    "PPMI": Path("/Volumes/reseng202500013-ndd-ml/PPMI"),
    "ADNI": Path("/Volumes/reseng202500013-ndd-ml/ADNI"),
}
DEST_ROOT = Path("/Volumes/reseng202500013-ndd-ml/data/raw")
DICOM_EXT  = ".dcm"
SMD_RE     = re.compile(r"^([^_]+)_([^_]+)_([^_]+)(?:_\d+)?$")

GOOD_SERIES_HINTS   = ("SPECT", "RECON", "REC", "TRANSVERSE", "TOMO", "BRAIN")
BAD_SERIES_HINTS    = ("PROJECTION", "PLANAR", "LOCALIZER", "SCOUT", "SINOGRAM", "MIP", "TOP", "WHOLEBODY")
GOOD_IMAGETYPE_HINT = ("TOMO", "RECON")
BAD_IMAGETYPE_HINT  = ("PROJECTION", "PROJ", "PLANAR")

# Prefer attenuation-corrected if available
AC_HINTS  = ("AC", "ATTN")
NAC_HINTS = ("NAC",)

# ───────────────────────────────────────────────────────────────────────────────

def find_subject_dirs(root: Path):
    """Yield any folder that contains DICOM files directly under it."""
    for d in root.rglob("*"):
        if d.is_dir():
            try:
                if any(p.suffix.lower() == DICOM_EXT for p in d.iterdir()):
                    yield d
            except PermissionError:
                continue

def extract_smd(folder: Path) -> Tuple[str, str, str]:
    """Walk up from `folder` until we find a parent whose name matches SMD_RE."""
    for anc in folder.parents:
        m = SMD_RE.match(anc.name)
        if m:
            return m.group(1), m.group(2), m.group(3)
    # Fallback if not found
    return ("PPMI", "SPECT", "UNKNOWN")

def _read_one(ds_path: Path) -> Optional[pydicom.dataset.FileDataset]:
    try:
        return pydicom.dcmread(str(ds_path), stop_before_pixels=True, force=True)
    except Exception:
        return None

def sniff_series(dicom_dir: Path) -> Optional[Dict[str, Any]]:
    """Read a few files and summarize key DICOM tags to decide if this is a good brain SPECT RECON."""
    files = [p for p in dicom_dir.iterdir() if p.suffix.lower() == DICOM_EXT]
    if not files:
        return None
    # sample up to 8 slices/frames
    samples = [f for i, f in enumerate(files) if i < 8]
    metas: List[Dict[str, Any]] = []
    for f in samples:
        ds = _read_one(f)
        if ds is None:
            continue
        def g(tag, default=""):
            try:
                v = ds.get(tag, default)
                if v is None: return default
                return str(v)
            except Exception:
                return default
        # Tags of interest
        info = {
            "Modality": g("Modality"),
            "SeriesDescription": g("SeriesDescription").upper(),
            "ProtocolName": g("ProtocolName").upper(),
            "ImageType": "|".join([s.upper() for s in (ds.get("ImageType", []) or [])]),
            "CorrectedImage": g("CorrectedImage").upper(),          # (0028,0051)
            "AcquisitionType": g("AcquisitionType").upper(),        # (0018,9302) sometimes absent
            "NumberOfFrames": g("NumberOfFrames"),
            "Rows": g("Rows"),
            "Columns": g("Columns"),
            "SeriesInstanceUID": g("SeriesInstanceUID"),
        }
        metas.append(info)

    if not metas:
        return None

    # Consolidate/score
    m0 = metas[0]
    modality_ok = (m0["Modality"].upper() == "NM")

    series_text = (m0["SeriesDescription"] + " " + m0["ProtocolName"]).upper()
    imgtype = m0["ImageType"]

    def has_any(text: str, keys: Tuple[str, ...]) -> bool:
        t = text.upper()
        return any(k in t for k in keys)

    good_series = has_any(series_text, GOOD_SERIES_HINTS)
    bad_series  = has_any(series_text, BAD_SERIES_HINTS)
    good_it     = has_any(imgtype, GOOD_IMAGETYPE_HINT)
    bad_it      = has_any(imgtype, BAD_IMAGETYPE_HINT)

    corrected   = m0["CorrectedImage"]
    is_ac_hint  = has_any(corrected, AC_HINTS) or has_any(series_text, AC_HINTS)
    is_nac_hint = has_any(corrected, NAC_HINTS) or has_any(series_text, NAC_HINTS)

    # Heuristics for a *reconstructed tomographic* brain SPECT:
    looks_tomo   = good_series or good_it
    not_planar   = (not bad_series) and (not bad_it)
    rows = int(m0["Rows"] or 0)
    cols = int(m0["Columns"] or 0)
    dims_ok = 64 <= rows <= 512 and 64 <= cols <= 512

    score = 0
    score += 3 if modality_ok else -5
    score += 4 if looks_tomo else -4
    score += 2 if not_planar else -4
    score += 1 if dims_ok else -2
    score += 2 if is_ac_hint else 0
    score -= 1 if is_nac_hint else 0

    return {
        "ok": score >= 3,
        "score": score,
        "meta": m0,
        "is_ac": bool(is_ac_hint),
        "is_nac": bool(is_nac_hint)
    }

def convert_dicom(dicom_dir: Path, out_dir: Path, prefix: str) -> Optional[Path]:
    """Run dcm2niix to convert into a single .nii.gz + .json under out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "dcm2niix",
        "-b", "y",     # write JSON
        "-z", "y",     # gzip
        "-m", "y",     # merge 2D slices/frames into 3D
        "-x", "n",     # do NOT crop
        "-r", "y",     # reorient to closest orthogonal if needed
        "-f", prefix,  # filename
        "-o", str(out_dir),
        str(dicom_dir)
    ]
    subprocess.run(cmd, check=True)
    # Return path of the NIfTI we just made (prefix*.nii.gz)
    niftis = sorted(out_dir.glob(prefix + "*.nii.gz"))
    return niftis[-1] if niftis else None

def main():
    for ds_name, ds_root in DATASETS.items():
        if not ds_root.exists():
            print(f"[!] Missing: {ds_root}")
            continue

        for subj_dir in find_subject_dirs(ds_root):
            site, modality, diagnosis = extract_smd(subj_dir)
            subject_id = subj_dir.name

            sniff = sniff_series(subj_dir)
            if sniff is None or not sniff["ok"]:
                # Skip non-NM/projection/localizer/etc.
                continue

            # prefer AC: only convert NAC if AC wasn’t detected
            if sniff["is_nac"]:
                # If it’s NAC, keep going only if no AC sibling exists.
                # Check sibling dirs with similar naming for AC tags
                parent = subj_dir.parent
                siblings = [d for d in parent.iterdir() if d.is_dir() and d != subj_dir]
                ac_found = False
                for sib in siblings:
                    s2 = sniff_series(sib)
                    if s2 and s2["ok"] and s2["is_ac"]:
                        ac_found = True
                        subj_dir = sib
                        sniff = s2
                        break
                if not ac_found:
                    # keep NAC, but mark it
                    pass

            out_prefix = f"sub-{subject_id}_{site}_{modality}_{diagnosis}"
            dest_folder = DEST_ROOT / modality / site / diagnosis / out_prefix

            print(f"[{ds_name}] Converting: {subj_dir.relative_to(ds_root)}  "
                  f"(score={sniff['score']} AC={sniff['is_ac']} NAC={sniff['is_nac']}) -> {dest_folder}")

            nii_path = convert_dicom(subj_dir, dest_folder, out_prefix)
            if nii_path:
                # write a small QA json
                qa = {
                    "chosen_series": sniff["meta"],
                    "score": sniff["score"],
                    "is_ac": sniff["is_ac"],
                    "is_nac": sniff["is_nac"],
                    "source_dir": str(subj_dir),
                    "nifti": str(nii_path)
                }
                with open(nii_path.with_suffix(".qa.json"), "w") as f:
                    json.dump(qa, f, indent=2)

    print("All done.")

if __name__ == "__main__":
    main()
