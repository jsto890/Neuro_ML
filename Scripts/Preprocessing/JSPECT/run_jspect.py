#!/usr/bin/env python3
"""
DaT-SPECT preprocessing pipeline (JSPECT)

Steps per subject:
 1) Brain masking via Otsu threshold (native space)
 2) Rigid + affine registration to symmetric FP-CIT DaT-SPECT template (MNI)
 3) SUVR normalization using occipital mask (in template space)
 4) Optional intensity clipping
 5) Save registered (pre-SUVR) and final SUVR images in output directory

Paths are read from config.yaml. By default, output is written under the
preprocessed SPECT root from config, with an additional subdirectory name you
can control via --output-subdir (defaults to 'jfinal').

Example:
  python Scripts/Preprocessing/JSPECT/run_jspect.py \
    --output-subdir jfinal \
    --clip-upper 10.0 \
    --skip-existing

Note: This script intentionally writes outputs outside of the repository (into
the configured preprocessed data folder), avoiding the project tree.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
import yaml
import matplotlib.pyplot as plt


def expand_user_and_resolve(path_str: str) -> Path:
    return Path(os.path.expanduser(path_str)).resolve()


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def find_spect_subject_niis(raw_spect_root: Path) -> List[Path]:
    """Discover subject NIfTIs under the expected PPMI layout.

    Expected structure:
      raw_spect_root/PPMI/{CN,PD}/sub-*_PPMI_SPECT_*/<file>.nii[.gz]
    """
    candidates: List[Path] = []
    ppmi_dir = raw_spect_root / "PPMI"
    if not ppmi_dir.exists():
        # Fall back to scanning the whole SPECT root
        search_roots = [raw_spect_root]
    else:
        search_roots = [ppmi_dir]

    for root in search_roots:
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if name.endswith(".nii") or name.endswith(".nii.gz"):
                    # Ignore sidecars like .json
                    if name.endswith(".json"):
                        continue
                    candidates.append(Path(dirpath) / name)

    # Prefer the most specific subject-level file if multiple exist
    # No filtering beyond extension here; upstream structure keeps it clean
    return sorted(candidates)


def derive_subject_id(nii_path: Path) -> str:
    """Return a concise subject identifier from the filename or its parent folder."""
    base = nii_path.stem
    if base.endswith(".nii"):
        base = base[:-4]
    # Prefer parent directory if it encodes the full subject id
    parent_name = nii_path.parent.name
    if parent_name.startswith("sub-"):
        return parent_name
    return base


def _keep_largest_component(binary_mask: sitk.Image) -> sitk.Image:
    cc = sitk.ConnectedComponent(binary_mask)
    relabeled = sitk.RelabelComponent(cc, sortByObjectSize=True)
    largest = sitk.BinaryThreshold(relabeled, lowerThreshold=1, upperThreshold=1, insideValue=1, outsideValue=0)
    return sitk.Cast(largest, sitk.sitkUInt8)


def _compute_mask_fraction(mask: sitk.Image) -> float:
    arr = sitk.GetArrayFromImage(mask).astype(bool)
    return float(arr.mean())


def create_brain_mask_otsu(image: sitk.Image) -> sitk.Image:
    # Ensure non-negative values for stable thresholding
    img_arr = sitk.GetArrayFromImage(image)
    min_intensity = float(np.nanmin(img_arr))
    if min_intensity < 0:
        image = sitk.ShiftScale(image, shift=-min_intensity, scale=1.0)

    # Replace NaNs/Infs with 0 for stability
    img_arr = sitk.GetArrayFromImage(image)
    img_arr = np.nan_to_num(img_arr, nan=0.0, posinf=0.0, neginf=0.0)
    image = sitk.GetImageFromArray(img_arr)
    image.CopyInformation(sitk.Cast(image, sitk.sitkFloat32))

    mask = sitk.OtsuThreshold(image, 0, 1, numberOfHistogramBins=256)
    mask = sitk.BinaryFillhole(mask)
    mask = sitk.BinaryMorphologicalClosing(mask, [2, 2, 2])
    mask = sitk.BinaryDilate(mask, [1, 1, 1])
    mask = _keep_largest_component(mask)

    # Sanity check on fraction; adapt morphology if extreme
    frac = _compute_mask_fraction(mask)
    if frac < 0.03:
        # Too small: dilate more and refit largest component
        mask = sitk.BinaryDilate(mask, [3, 3, 3])
        mask = _keep_largest_component(mask)
    elif frac > 0.80:
        # Too large: erode and refit largest component
        mask = sitk.BinaryErode(mask, [2, 2, 2])
        mask = _keep_largest_component(mask)

    return sitk.Cast(mask, sitk.sitkUInt8)


def apply_mask(image: sitk.Image, mask: sitk.Image) -> sitk.Image:
    return sitk.Mask(image, mask)


def register_to_template(
    moving_image: sitk.Image,
    fixed_template: sitk.Image,
    sampling_percentage: float = 0.2,
    num_pyramid_levels: int = 3,

    enable_nonlinear: bool = True,
) -> sitk.Transform:
    """Multi-stage registration: rigid -> affine -> optional BSpline non-linear.

    Returns the final composite transform mapping moving -> fixed.
    """

    def _run_stage(initial_tx: sitk.Transform, transform: sitk.Transform) -> sitk.Transform:
        registration = sitk.ImageRegistrationMethod()
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(sampling_percentage)
        registration.SetInterpolator(sitk.sitkLinear)

        registration.SetOptimizerAsRegularStepGradientDescent(
            learningRate=2.0,
            minStep=1e-4,
            numberOfIterations=200,
            relaxationFactor=0.5,
        )
        registration.SetOptimizerScalesFromPhysicalShift()

        registration.SetShrinkFactorsPerLevel([2] * num_pyramid_levels)
        registration.SetSmoothingSigmasPerLevel([1] * num_pyramid_levels)
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        registration.SetInitialTransform(initial_tx, inPlace=False)
        final_tx = registration.Execute(fixed_template, moving_image)
        return final_tx

    # Rigid initialization (centered)
    initial_rigid = sitk.CenteredTransformInitializer(
        fixed_template,
        moving_image,
        sitk.VersorRigid3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    rigid_tx = _run_stage(initial_rigid, sitk.VersorRigid3DTransform())

    # Affine refinement initialized from rigid
    affine_init = sitk.AffineTransform(3)
    affine_init.SetMatrix(sitk.VersorRigid3DTransform(rigid_tx).GetMatrix())
    affine_init.SetTranslation(sitk.VersorRigid3DTransform(rigid_tx).GetTranslation())
    affine_tx = _run_stage(affine_init, sitk.AffineTransform(3))

    if not enable_nonlinear:
        return sitk.Transform(affine_tx)

    # BSpline non-linear refinement initialized on the fixed (template) grid
    grid_physical_spacing = [50.0, 50.0, 50.0]
    image_physical_size = [sz * sp for sz, sp in zip(fixed_template.GetSize(), fixed_template.GetSpacing())]
    mesh_size = [int(sz / gs + 0.5) for sz, gs in zip(image_physical_size, grid_physical_spacing)]
    mesh_size = [max(1, m) for m in mesh_size]

    bspline_initial = sitk.BSplineTransformInitializer(image1=fixed_template, transformDomainMeshSize=mesh_size, order=3)

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(sampling_percentage)
    registration.SetInterpolator(sitk.sitkLinear)

    registration.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=100, maximumNumberOfCorrections=5, maximumNumberOfFunctionEvaluations=500)
    registration.SetShrinkFactorsPerLevel([2] * num_pyramid_levels)
    registration.SetSmoothingSigmasPerLevel([1] * num_pyramid_levels)
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Compose affine then bspline
    composite_init = sitk.Transform(affine_tx)
    composite = sitk.Transform(composite_init)
    bspline_tx = sitk.BSplineTransform(bspline_initial)
    registration.SetMovingInitialTransform(composite)
    registration.SetInitialTransform(bspline_tx, inPlace=False)
    final_bspline = registration.Execute(fixed_template, moving_image)

    # Build final composite: affine followed by bspline
    final_composite = sitk.Transform(3, sitk.sitkComposite)
    final_composite.AddTransform(affine_tx)
    final_composite.AddTransform(final_bspline)
    return final_composite


def resample_to_reference(moving: sitk.Image, reference: sitk.Image, transform: sitk.Transform) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(transform)
    resampler.SetDefaultPixelValue(0.0)
    return resampler.Execute(moving)


def resample_to_iso_like_template(image: sitk.Image, template: sitk.Image, iso_mm: Optional[float]) -> sitk.Image:
    if iso_mm is None or iso_mm <= 0:
        return image
    out_spacing = (float(iso_mm), float(iso_mm), float(iso_mm))
    in_size = np.array(template.GetSize(), dtype=float)
    in_spacing = np.array(template.GetSpacing(), dtype=float)
    out_size = np.maximum(np.round(in_size * (in_spacing / np.array(out_spacing))), 1).astype(int)

    ref = sitk.Image(int(out_size[0]), int(out_size[1]), int(out_size[2]), sitk.sitkFloat32)
    ref.SetOrigin(template.GetOrigin())
    ref.SetDirection(template.GetDirection())
    ref.SetSpacing(out_spacing)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetDefaultPixelValue(0.0)
    return resampler.Execute(image)


def compute_occipital_stat(
    template_space_image: sitk.Image,
    occipital_mask_img: sitk.Image,
    method: str = "median",
    nonzero_only: bool = True,
) -> float:
    img_np = sitk.GetArrayFromImage(template_space_image).astype(np.float64)
    mask_np = sitk.GetArrayFromImage(occipital_mask_img).astype(bool)
    if img_np.shape != mask_np.shape:
        raise ValueError(
            f"Shape mismatch: image {img_np.shape} vs mask {mask_np.shape}. Ensure the occipital mask matches the template grid."
        )
    masked_vals = img_np[mask_np]
    masked_vals = masked_vals[np.isfinite(masked_vals)]
    nz = masked_vals > 0 if nonzero_only else np.ones_like(masked_vals, dtype=bool)
    if not np.any(nz):
        nz = np.ones_like(masked_vals, dtype=bool)
    vals = masked_vals[nz]
    if vals.size == 0:
        raise ValueError("Occipital mask contains no valid voxels in the resampled image")
    if method.lower() == "median":
        stat = float(np.median(vals))
    elif method.lower() == "mean":
        stat = float(np.mean(vals))
    else:
        raise ValueError(f"Unknown method '{method}', expected 'median' or 'mean'")
    if not np.isfinite(stat) or stat <= 1e-6:
        raise ValueError(f"Invalid occipital statistic encountered: {stat}")
    return stat


def suvr_normalize(
    template_space_image: sitk.Image,
    occipital_mask_img: sitk.Image,
    method: str = "median",
    nonzero_only: bool = True,
) -> sitk.Image:
    occ_val = compute_occipital_stat(template_space_image, occipital_mask_img, method=method, nonzero_only=nonzero_only)
    scale = 1.0 / max(occ_val, 1e-6)
    suvr_img = sitk.ShiftScale(template_space_image, shift=0.0, scale=scale)
    return suvr_img


def clip_intensity(image: sitk.Image, lower: Optional[float], upper: Optional[float]) -> sitk.Image:
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    if lower is not None or upper is not None:
        lower_v = lower if lower is not None else -np.inf
        upper_v = upper if upper is not None else np.inf
        arr = np.clip(arr, lower_v, upper_v)
    # Clean NaN/Inf
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(image)
    return out


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_image(img: sitk.Image, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    # Enforce float32 output for consistency
    img32 = sitk.Cast(img, sitk.sitkFloat32)
    sitk.WriteImage(img32, str(out_path))


def create_template_brain_mask(template_img: sitk.Image) -> sitk.Image:
    # Threshold template using Otsu to get brain-like region, then clean up
    mask = sitk.OtsuThreshold(template_img, 0, 1, numberOfHistogramBins=128)
    mask = sitk.BinaryFillhole(mask)
    mask = sitk.BinaryMorphologicalClosing(mask, [1, 1, 1])
    mask = _keep_largest_component(mask)
    return sitk.Cast(mask, sitk.sitkUInt8)


def compute_ncc(a: sitk.Image, b: sitk.Image) -> float:
    a_np = sitk.GetArrayFromImage(a).astype(np.float64)
    b_np = sitk.GetArrayFromImage(b).astype(np.float64)
    a_np = np.nan_to_num(a_np, nan=0.0, posinf=0.0, neginf=0.0)
    b_np = np.nan_to_num(b_np, nan=0.0, posinf=0.0, neginf=0.0)
    a0 = a_np - a_np.mean()
    b0 = b_np - b_np.mean()
    denom = (np.linalg.norm(a0) * np.linalg.norm(b0))
    if denom == 0:
        return 0.0
    return float(np.tensordot(a0.ravel(), b0.ravel(), axes=1) / denom)


def save_qc_png(registered: sitk.Image, suvr: sitk.Image, template: sitk.Image, out_path: Path) -> None:
    reg_np = sitk.GetArrayFromImage(registered)
    suv_np = sitk.GetArrayFromImage(suvr)
    tpl_np = sitk.GetArrayFromImage(template)

    z = reg_np.shape[0] // 2
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(reg_np[z, :, :], cmap="gray"); axes[0].set_title("Registered")
    axes[1].imshow(suv_np[z, :, :], cmap="gray"); axes[1].set_title("SUVR")
    axes[2].imshow(tpl_np[z, :, :], cmap="gray"); axes[2].set_title("Template")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(out_path))
    plt.close(fig)


def process_subject(
    subject_nii: Path,
    template_img: sitk.Image,
    occipital_mask_img: sitk.Image,
    output_dir: Path,
    clip_lower: Optional[float],
    clip_upper: Optional[float],
    skip_existing: bool,
    enable_nonlinear: bool,
    final_iso_mm: Optional[float],
    save_transforms: bool,
    save_qc: bool,
    occipital_method: str,
    occipital_nonzero_only: bool,
) -> Tuple[str, Optional[str]]:
    subject_id = derive_subject_id(subject_nii)
    out_pre = output_dir / f"{subject_id}_space-MNI.nii.gz"
    out_suvr = output_dir / f"{subject_id}_space-MNI_SUVR.nii.gz"
    out_qc = output_dir / f"{subject_id}_qc.png"
    out_json = output_dir / f"{subject_id}_space-MNI_SUVR.json"
    out_affine = output_dir / f"{subject_id}_affine.tfm"
    out_transform = output_dir / f"{subject_id}_composite.tfm"

    if skip_existing and out_suvr.exists():
        return subject_id, "skipped"

    moving_img = sitk.ReadImage(str(subject_nii))

    # Optional light pre-smoothing to stabilize MI on noisy scans
    moving_img = sitk.DiscreteGaussian(moving_img, variance=1.5)

    # 1) Masking in native space
    brain_mask = create_brain_mask_otsu(moving_img)
    masked_img = apply_mask(moving_img, brain_mask)

    # 2) Register to template (rigid + affine) and resample to template grid
    tx = register_to_template(masked_img, template_img, enable_nonlinear=enable_nonlinear)
    registered_img = resample_to_reference(masked_img, template_img, tx)

    # Save the pre-SUVR registered image
    # Post-registration template brain mask to clean background
    tpl_mask = create_template_brain_mask(template_img)
    registered_img = sitk.Mask(registered_img, tpl_mask)

    # Optional final resampling to isotropic grid similar to template FOV
    registered_iso = resample_to_iso_like_template(registered_img, template_img, final_iso_mm)
    save_image(registered_iso, out_pre)

    # 3) SUVR using occipital mask (robust by default)
    suvr_img = suvr_normalize(
        registered_iso,
        occipital_mask_img,
        method=occipital_method,
        nonzero_only=occipital_nonzero_only,
    )

    # 4) Optional clipping
    # Percentile-based additional clipping safeguard
    arr = sitk.GetArrayFromImage(suvr_img)
    p99 = float(np.percentile(arr[np.isfinite(arr)], 99)) if np.any(np.isfinite(arr)) else None
    suvr_img = clip_intensity(suvr_img, clip_lower, clip_upper if clip_upper is not None else (p99 if p99 else None))

    # 5) Save final
    save_image(suvr_img, out_suvr)

    # Save transforms
    if save_transforms:
        try:
            if isinstance(tx, sitk.Transform):
                # Try to extract the first transform as affine if composite
                if tx.GetName() == 'CompositeTransform' and tx.GetNumberOfTransforms() > 0:
                    sitk.WriteTransform(tx.GetNthTransform(0), str(out_affine))
                elif tx.GetName() == 'AffineTransform':
                    sitk.WriteTransform(tx, str(out_affine))
                # Save composite
                sitk.WriteTransform(tx, str(out_transform))
        except Exception:
            pass

    # Compute QC metrics and save sidecar
    ncc = compute_ncc(registered_iso, template_img)
    brain_fraction = _compute_mask_fraction(tpl_mask)
    try:
        occ_stat_val = compute_occipital_stat(
            registered_iso, occipital_mask_img, method=occipital_method, nonzero_only=occipital_nonzero_only
        )
    except Exception:
        occ_stat_val = None
    provenance = {
        "subject": subject_id,
        "enable_nonlinear": enable_nonlinear,
        "final_iso_mm": final_iso_mm,
        "clip_lower": clip_lower,
        "clip_upper": clip_upper,
        "occipital_reference_method": occipital_method,
        "occipital_reference_nonzero_only": occipital_nonzero_only,
        "occipital_reference_value": occ_stat_val,
        "ncc_to_template": ncc,
        "template_brain_fraction": brain_fraction,
        "registered_path": str(out_pre),
        "suvr_path": str(out_suvr),
    }
    try:
        with open(out_json, 'w') as f:
            json.dump(provenance, f, indent=2)
    except Exception:
        pass

    if save_qc:
        try:
            save_qc_png(registered_iso, suvr_img, template_img, out_qc)
        except Exception:
            pass

    return subject_id, None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="JSPECT preprocessing pipeline")
    p.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parents[3] / "config.yaml"),
        help="Path to config.yaml",
    )
    p.add_argument(
        "--input-root",
        type=str,
        default=None,
        help="Override raw SPECT root (defaults to config.raw_data.spect)",
    )
    p.add_argument(
        "--output-subdir",
        type=str,
        default="jfinal",
        help="Subdirectory under config.preprocessed_data.spect_p to write outputs",
    )
    p.add_argument(
        "--template",
        type=str,
        default=None,
        help="Override SPECT template path (defaults to config.templates.SPECT_template)",
    )
    p.add_argument(
        "--occipital-mask",
        type=str,
        default=None,
        help="Override occipital mask path (defaults to config.templates.SPECT_occipital)",
    )
    p.add_argument("--clip-lower", type=float, default=0.0, help="Lower clip bound (or None)")
    p.add_argument("--clip-upper", type=float, default=10.0, help="Upper clip bound (or None)")
    p.add_argument("--skip-existing", action="store_true", help="Skip subjects whose final output exists")
    p.add_argument("--limit", type=int, default=None, help="Limit number of subjects for a dry run")
    p.add_argument("--disable-nonlinear", action="store_true", help="Disable non-linear BSpline refinement")
    p.add_argument("--final-iso-mm", type=float, default=1.0, help="Final isotropic voxel size in mm (<=0 to skip)")
    p.add_argument("--no-save-transforms", action="store_true", help="Do not save transform .tfm files")
    p.add_argument("--no-save-qc", action="store_true", help="Do not save QC PNG slices")
    p.add_argument(
        "--occipital-method",
        type=str,
        default="median",
        choices=["median", "mean"],
        help="Statistic over occipital mask for SUVR (default: median)",
    )
    p.add_argument(
        "--occipital-include-zeros",
        action="store_true",
        help="Include zero voxels inside occipital mask when computing the statistic",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(expand_user_and_resolve(args.config))
    raw_spect_root = expand_user_and_resolve(
        args.input_root if args.input_root is not None else cfg["raw_data"]["spect"]
    )
    out_root = expand_user_and_resolve(cfg["preprocessed_data"]["spect_p"]) / args.output_subdir

    template_path = expand_user_and_resolve(
        args.template if args.template is not None else cfg["templates"]["SPECT_template"]
    )
    occipital_mask_path = expand_user_and_resolve(
        args.occipital_mask if args.occipital_mask is not None else cfg["templates"]["SPECT_occipital"]
    )

    print("Config:")
    print(json.dumps(
        {
            "raw_spect_root": str(raw_spect_root),
            "output_root": str(out_root),
            "template": str(template_path),
            "occipital_mask": str(occipital_mask_path),
            "clip_lower": args.clip_lower,
            "clip_upper": args.clip_upper,
            "skip_existing": args.skip_existing,
            "nonlinear": not args.disable_nonlinear,
            "final_iso_mm": args.final_iso_mm,
            "occipital_method": args.occipital_method,
            "occipital_include_zeros": args.occipital_include_zeros,
        },
        indent=2,
    ))

    subject_niis = find_spect_subject_niis(raw_spect_root)
    if args.limit is not None:
        subject_niis = subject_niis[: args.limit]
    if not subject_niis:
        print(f"No SPECT NIfTI files found under {raw_spect_root}")
        return

    # Load template and occipital mask
    template_img = sitk.ReadImage(str(template_path))
    occipital_mask_img = sitk.ReadImage(str(occipital_mask_path))

    # Ensure occipital mask aligns to template (size, spacing, direction)
    if (
        template_img.GetSize() != occipital_mask_img.GetSize()
        or template_img.GetSpacing() != occipital_mask_img.GetSpacing()
        or template_img.GetDirection() != occipital_mask_img.GetDirection()
        or template_img.GetOrigin() != occipital_mask_img.GetOrigin()
    ):
        # Resample mask to template grid using nearest neighbor
        print("Resampling occipital mask to template grid (nearest neighbor)...")
        occ_nn_resampler = sitk.ResampleImageFilter()
        occ_nn_resampler.SetReferenceImage(template_img)
        occ_nn_resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        occ_nn_resampler.SetDefaultPixelValue(0)
        occ_nn_resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
        occipital_mask_img = occ_nn_resampler.Execute(occipital_mask_img)

    ensure_dir(out_root)

    results = []
    for i, nii_path in enumerate(subject_niis, start=1):
        try:
            subject_id, skipped_reason = process_subject(
                subject_nii=nii_path,
                template_img=template_img,
                occipital_mask_img=occipital_mask_img,
                output_dir=out_root,
                clip_lower=args.clip_lower,
                clip_upper=args.clip_upper,
                skip_existing=args.skip_existing,
                enable_nonlinear=(not args.disable_nonlinear),
                final_iso_mm=args.final_iso_mm,
                save_transforms=(not args.no_save_transforms),
                save_qc=(not args.no_save_qc),
                occipital_method=args.occipital_method,
                occipital_nonzero_only=(not args.occipital_include_zeros),
            )
            status = "skipped" if skipped_reason else "ok"
            print(f"[{i}/{len(subject_niis)}] {subject_id}: {status}")
            results.append({"subject": subject_id, "status": status})
        except Exception as e:
            print(f"[{i}/{len(subject_niis)}] ERROR processing {nii_path}: {e}")
            results.append({"subject": derive_subject_id(nii_path), "status": "error", "error": str(e)})

    # Summarize
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_skipped = sum(1 for r in results if r["status"] == "skipped")
    n_err = sum(1 for r in results if r["status"] == "error")
    print(f"Done. ok={n_ok}, skipped={n_skipped}, errors={n_err}")


if __name__ == "__main__":
    main()


