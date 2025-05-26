import os
from nipype.interfaces.freesurfer import ReconAll

def fs_recon_all_nipype(input_nii, subj_id, subjects_dir, num_threads=1):
    # 1) Define FS_HOME & PATH
    os.environ["FREESURFER_HOME"] = "/home/jsto890/freesurfer-8.0.0/8.0.0"
    os.environ["PATH"] = os.environ["FREESURFER_HOME"] + "/bin:" + os.environ.get("PATH", "")
    # 2) **Export SUBJECTS_DIR**
    os.environ["SUBJECTS_DIR"] = subjects_dir

    recon = ReconAll()
    recon.inputs.subjects_dir = subjects_dir
    recon.inputs.subject_id   = subj_id
    recon.inputs.T1_files      = [input_nii]
    recon.inputs.directive     = "all"
    if num_threads > 1:
        recon.inputs.args = f"-parallel -openmp {num_threads}"

    print("NiPype recon-all cmd:", recon.cmdline)
    os.environ["SUBJECTS_DIR"] = subjects_dir
    result = recon.run()
    if result.runtime.returncode != 0:
        print("Full stderr:\n", result.runtime.stderr)
        raise RuntimeError(f"NiPype recon-all failed (code {result.runtime.returncode})")
    return result
