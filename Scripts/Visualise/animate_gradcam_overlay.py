#!/usr/bin/env python3
"""
Animated Grad-CAM Overlay Viewer

Description:
- Loads a base anatomical NIfTI and a Grad-CAM NIfTI overlay.
- Displays axial, sagittal, and coronal views side-by-side with overlay.
- Automatically cycles slices in a ping-pong fashion for presentations.
- Optional MP4/GIF export.

Usage examples:
  python animate_gradcam_overlay.py \
    --base /abs/path/to/base.nii.gz \
    --overlay /abs/path/to/gradcam.nii.gz \
    --fps 8 --alpha 0.45 --cycles 2 --mode sequential

  python animate_gradcam_overlay.py \
    --base /abs/path/to/base.nii.gz \
    --overlay /abs/path/to/gradcam.nii.gz \
    --fps 12 --alpha 0.5 --mode simul \
    --save /abs/path/out.mp4

Keys:
- Space: pause/resume
- r: reset to center slices
- q or Esc: quit
"""

import os
import sys
import argparse
from typing import Tuple, List
from math import gcd

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter


def _robust_normalize(arr: np.ndarray, lo_p: float = 2.0, hi_p: float = 98.0) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo = np.percentile(arr, lo_p)
    hi = np.percentile(arr, hi_p)
    if hi - lo < 1e-6:
        return arr
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def _normalize_overlay_within_mask(overlay: np.ndarray, base: np.ndarray, lo_p: float = 90.0, hi_p: float = 99.5) -> np.ndarray:
    mask = (base != 0)
    vals = overlay[mask] if np.any(mask) else overlay
    lo = np.percentile(vals, lo_p)
    hi = np.percentile(vals, hi_p)
    if hi - lo < 1e-6:
        return _robust_normalize(overlay, 2.0, 98.0)
    out = (overlay - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def _ping_pong_indices(n: int) -> List[int]:
    if n <= 1:
        return [0]
    forward = list(range(0, n))
    backward = list(range(n - 2, 0, -1))
    return forward + backward


def _is_under_path(path: str, parent: str) -> bool:
    try:
        path_abs = os.path.abspath(path)
        parent_abs = os.path.abspath(parent)
        common = os.path.commonpath([path_abs, parent_abs])
        return common == parent_abs
    except Exception:
        return False


def _load_nifti(path: str) -> Tuple[nib.Nifti1Image, np.ndarray]:
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    if data.ndim == 4:
        data = data.mean(axis=-1)
    if data.ndim != 3:
        raise ValueError("Expected 3D or 4D NIfTI")
    if np.any(np.isnan(data)):
        data = np.nan_to_num(data, nan=0.0)
    if np.any(np.isinf(data)):
        data = np.nan_to_num(data, posinf=0.0, neginf=0.0)
    return img, data


def _build_sequential_frames(nx: int, ny: int, nz: int, cycles: int) -> List[Tuple[str, int]]:
    frames: List[Tuple[str, int]] = []
    ax_seq = _ping_pong_indices(nz)
    sag_seq = _ping_pong_indices(nx)
    cor_seq = _ping_pong_indices(ny)
    for _ in range(max(1, cycles)):
        for z in ax_seq:
            frames.append(("axial", z))
        for x in sag_seq:
            frames.append(("sagittal", x))
        for y in cor_seq:
            frames.append(("coronal", y))
    return frames


def _lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b) if a and b else 0


def _lcm3(a: int, b: int, c: int) -> int:
    return _lcm(_lcm(a, b), c)


def _build_simul_frame_count(nx: int, ny: int, nz: int, cycles: int) -> int:
    # Use LCM of ping-pong lengths so all three views start and end together
    lx = len(_ping_pong_indices(nx))
    ly = len(_ping_pong_indices(ny))
    lz = len(_ping_pong_indices(nz))
    period = _lcm3(lx, ly, lz)
    return max(1, period) * max(1, cycles)


def main():
    parser = argparse.ArgumentParser(description="Animated Grad-CAM Overlay Viewer")
    parser.add_argument("--base", required=True, type=str, help="Absolute path to base anatomical NIfTI")
    parser.add_argument("--overlay", required=True, type=str, help="Absolute path to Grad-CAM NIfTI")
    parser.add_argument("--alpha", type=float, default=0.4, help="Overlay alpha [0-1]")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for animation")
    parser.add_argument("--cycles", type=int, default=1, help="Number of ping-pong cycles per view")
    parser.add_argument("--mode", choices=["sequential", "simul"], default="sequential", help="Animate views sequentially or simultaneously")
    parser.add_argument("--save", type=str, default=None, help="Optional absolute path to save animation (.mp4 or .gif)")
    args = parser.parse_args()

    if not os.path.isabs(args.base) or not os.path.isabs(args.overlay):
        print("Please provide absolute paths for --base and --overlay")
        sys.exit(2)

    if args.save is not None and not os.path.isabs(args.save):
        print("--save must be an absolute path if provided")
        sys.exit(2)

    if args.save is not None:
        workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if _is_under_path(args.save, workspace_root):
            print(f"Warning: Save path is under workspace {workspace_root}. Prefer saving outside this folder.")

    try:
        b_img, b_data = _load_nifti(args.base)
        o_img, o_data = _load_nifti(args.overlay)
    except Exception as e:
        print(f"Failed to load NIfTI: {e}")
        sys.exit(1)

    if b_data.shape != o_data.shape:
        print(f"Warning: Shape mismatch base {b_data.shape} vs overlay {o_data.shape}. Proceeding without resample.")

    base = _robust_normalize(b_data)
    heat = _normalize_overlay_within_mask(o_data, b_data)

    nx, ny, nz = base.shape
    x_idx, y_idx, z_idx = nx // 2, ny // 2, nz // 2

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Grad-CAM Overlay", fontsize=16)

    im1 = ax1.imshow(base[:, :, z_idx].T, cmap='gray', origin='lower')
    hm1 = ax1.imshow(heat[:, :, z_idx].T, cmap='hot', origin='lower', alpha=float(args.alpha))
    ax1.set_title(f'Axial (Z={z_idx})')
    ax1.axis('off')

    im2 = ax2.imshow(base[x_idx, :, :].T, cmap='gray', origin='lower')
    hm2 = ax2.imshow(heat[x_idx, :, :].T, cmap='hot', origin='lower', alpha=float(args.alpha))
    ax2.set_title(f'Sagittal (X={x_idx})')
    ax2.axis('off')

    im3 = ax3.imshow(base[:, y_idx, :].T, cmap='gray', origin='lower')
    hm3 = ax3.imshow(heat[:, y_idx, :].T, cmap='hot', origin='lower', alpha=float(args.alpha))
    ax3.set_title(f'Coronal (Y={y_idx})')
    ax3.axis('off')

    vline1 = ax1.axvline(x_idx, color='cyan', linewidth=1, alpha=0.8)
    hline1 = ax1.axhline(y_idx, color='cyan', linewidth=1, alpha=0.8)
    vline2 = ax2.axvline(y_idx, color='cyan', linewidth=1, alpha=0.8)
    hline2 = ax2.axhline(z_idx, color='cyan', linewidth=1, alpha=0.8)
    vline3 = ax3.axvline(x_idx, color='cyan', linewidth=1, alpha=0.8)
    hline3 = ax3.axhline(z_idx, color='cyan', linewidth=1, alpha=0.8)

    paused = {"value": False}

    def _update_views():
        im1.set_array(base[:, :, z_idx].T)
        hm1.set_array(heat[:, :, z_idx].T)
        ax1.set_title(f'Axial (Z={z_idx})')

        im2.set_array(base[x_idx, :, :].T)
        hm2.set_array(heat[x_idx, :, :].T)
        ax2.set_title(f'Sagittal (X={x_idx})')

        im3.set_array(base[:, y_idx, :].T)
        hm3.set_array(heat[:, y_idx, :].T)
        ax3.set_title(f'Coronal (Y={y_idx})')

        vline1.set_xdata([x_idx, x_idx])
        hline1.set_ydata([y_idx, y_idx])
        vline2.set_xdata([y_idx, y_idx])
        hline2.set_ydata([z_idx, z_idx])
        vline3.set_xdata([x_idx, x_idx])
        hline3.set_ydata([z_idx, z_idx])

    def on_key(event):
        nonlocal x_idx, y_idx, z_idx
        if event.key in (' ', 'space'):
            paused["value"] = not paused["value"]
        elif event.key in ('r', 'R'):
            x_idx, y_idx, z_idx = nx // 2, ny // 2, nz // 2
            _update_views()
            fig.canvas.draw_idle()
        elif event.key in ('q', 'Q', 'escape'):
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)

    if args.mode == "sequential":
        frames_seq = _build_sequential_frames(nx, ny, nz, args.cycles)

        def update(i):
            nonlocal x_idx, y_idx, z_idx
            if paused["value"]:
                return
            view, index = frames_seq[i]
            if view == "axial":
                z_idx = int(index)
            elif view == "sagittal":
                x_idx = int(index)
            else:
                y_idx = int(index)
            _update_views()

        anim = FuncAnimation(fig, update, frames=len(frames_seq), interval=int(1000 / max(1, args.fps)), repeat=True)

    else:
        px = _ping_pong_indices(nx)
        py = _ping_pong_indices(ny)
        pz = _ping_pong_indices(nz)
        length = _build_simul_frame_count(nx, ny, nz, args.cycles)

        def update(i):
            nonlocal x_idx, y_idx, z_idx
            if paused["value"]:
                return
            ix = px[i % len(px)]
            iy = py[i % len(py)]
            iz = pz[i % len(pz)]
            x_idx, y_idx, z_idx = int(ix), int(iy), int(iz)
            _update_views()

        anim = FuncAnimation(fig, update, frames=length, interval=int(1000 / max(1, args.fps)), repeat=True)

    if args.save:
        out_path = args.save
        ext = os.path.splitext(out_path)[1].lower()
        try:
            if ext == ".mp4":
                writer = FFMpegWriter(fps=max(1, args.fps), bitrate=4000)
            elif ext in (".gif", ".apng"):
                writer = PillowWriter(fps=max(1, args.fps))
            else:
                print("Unknown extension for --save. Use .mp4 or .gif")
                sys.exit(2)
            print(f"Saving animation to {out_path} ...")
            anim.save(out_path, writer=writer)
            print("Saved.")
        except Exception as e:
            print(f"Failed to save animation: {e}")

    plt.show()


if __name__ == "__main__":
    main()


