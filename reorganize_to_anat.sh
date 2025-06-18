#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/home/jsto890/reseng202500013-ndd-ml/data/raw/MRI"

# find all sub-*/ directories
find "$BASE_DIR" -type d -name 'sub-*' -print0 | while IFS= read -r -d '' subj; do
  echo "Processing $subj …"

  # make anat folder
  anat_dir="$subj/anat"
  mkdir -p "$anat_dir"

  # move everything except the anat folder itself into anat/
  find "$subj" -mindepth 1 -maxdepth 1 ! -name 'anat' -print0 \
    | xargs -0 -I {} mv {} "$anat_dir"/

done

echo "Done."
