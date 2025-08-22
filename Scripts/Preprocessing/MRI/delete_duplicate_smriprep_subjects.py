#!/usr/bin/env python3
import argparse
import os
import csv
import shutil
from pathlib import Path


def list_subject_dirs(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {
        p.name
        for p in root.iterdir()
        if p.is_dir() and p.name.startswith("sub-")
    }


def delete_subject_dir(dest_root: Path, subject: str, dry_run: bool) -> bool:
    target = dest_root / subject
    if not target.exists():
        return False

    # Safety checks
    if not subject.startswith("sub-"):
        raise RuntimeError(f"Refusing to delete non-subject directory name: {subject}")
    # Ensure we are deleting inside the intended destination
    expected_segment = os.path.join("data", "preprocessed", "MRI", "smriprep")
    if expected_segment not in str(dest_root):
        raise RuntimeError(f"Refusing to delete outside expected dest: {dest_root}")

    if dry_run:
        print(f"DRY-RUN: would delete {target}")
        return True

    shutil.rmtree(target)
    print(f"Deleted {target}")
    return True


def load_disease_map(csv_path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not csv_path.exists():
        return mapping
    try:
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = (row.get("SubjectID") or "").strip()
                dis = (row.get("Disease") or "Unknown").strip()
                if sid and sid not in mapping:
                    mapping[sid] = dis
    except Exception:
        # Fail silently; will classify as Unknown
        pass
    return mapping


def subject_id_from_folder(name: str) -> str:
    return name[4:] if name.startswith("sub-") else name


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Delete only duplicate subject folders from DEST that are present in SOURCE.\n"
            "Use this to remove duplicates under MRI/smriprep so you can move from smriprep/smriprep."
        )
    )
    parser.add_argument(
        "--source",
        default=str(Path.home() / "reseng202500013-ndd-ml/data/preprocessed/smriprep/smriprep"),
        help="Path to source smriprep root (contains subject folders to keep)",
    )
    parser.add_argument(
        "--dest",
        default=str(Path.home() / "reseng202500013-ndd-ml/data/preprocessed/MRI/smriprep"),
        help="Path to destination MRI/smriprep root (duplicates here will be deleted)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually delete. If omitted, runs in dry-run mode",
    )

    args = parser.parse_args()
    source_root = Path(os.path.expanduser(args.source)).resolve()
    dest_root = Path(os.path.expanduser(args.dest)).resolve()
    dry_run = not args.yes

    print(f"Source: {source_root}")
    print(f"Destination: {dest_root}")
    print(f"Mode: {'DRY-RUN' if dry_run else 'DELETE'}")

    src_subjects = list_subject_dirs(source_root)
    dest_subjects = list_subject_dirs(dest_root)

    if not src_subjects:
        print("Warning: no subject folders found in source.")
    if not dest_subjects:
        print("Warning: no subject folders found in destination.")

    # Only delete those in dest that also exist in source
    to_delete = sorted(dest_subjects.intersection(src_subjects))
    to_add = sorted(src_subjects.difference(dest_subjects))

    # Disease breakdowns
    disease_csv = Path.home() / "reseng202500013-ndd-ml/data/imaging_records.csv"
    dis_map = load_disease_map(disease_csv)

    def count_by_disease(subject_list: list[str]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for folder in subject_list:
            sid = subject_id_from_folder(folder)
            dis = dis_map.get(sid, "Unknown")
            counts[dis] = counts.get(dis, 0) + 1
        return counts

    del_counts = count_by_disease(to_delete)
    add_counts = count_by_disease(to_add)

    print(f"Found {len(src_subjects)} subject(s) in source, {len(dest_subjects)} in dest")
    print(f"Duplicates to delete from dest: {len(to_delete)}")
    if del_counts:
        print("Delete by disease:")
        for dis, cnt in sorted(del_counts.items(), key=lambda x: (-x[1], x[0])):
            print(f"  {dis}: {cnt}")
    print(f"To be added (present in source, missing in dest): {len(to_add)}")
    if add_counts:
        print("Add by disease:")
        for dis, cnt in sorted(add_counts.items(), key=lambda x: (-x[1], x[0])):
            print(f"  {dis}: {cnt}")

    for subj in to_delete:
        delete_subject_dir(dest_root, subj, dry_run)

    print("Done.")


if __name__ == "__main__":
    main()


