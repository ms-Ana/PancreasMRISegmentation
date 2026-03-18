import argparse
import ast
import os
import pickle
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd

try:
    import SimpleITK as sitk
except ImportError:  # Optional dependency; used for .mha/.mhd support.
    sitk = None


def get_label_transition_matrix(mask1: np.ndarray, mask2: np.ndarray):
    """Build a transition matrix for changed labels only."""
    f1 = mask1.flatten()
    f2 = mask2.flatten()
    diff_indices = np.where(f1 != f2)

    if len(diff_indices[0]) == 0:
        return None, 0

    changed_f1 = f1[diff_indices]
    changed_f2 = f2[diff_indices]
    transition_matrix = pd.crosstab(
        pd.Series(changed_f1, name="File1_Label"),
        pd.Series(changed_f2, name="File2_Label"),
    )
    return transition_matrix, len(diff_indices[0])


def recursive_dict_compare(d1: dict[str, Any], d2: dict[str, Any], path: str = ""):
    """Recursively compare dict values while safely handling arrays."""
    diffs = []
    keys = set(d1.keys()) | set(d2.keys())

    for key in keys:
        new_path = f"{path}.{key}" if path else str(key)

        if key not in d1:
            diffs.append(f"Key '{new_path}' missing in File 1")
            continue
        if key not in d2:
            diffs.append(f"Key '{new_path}' missing in File 2")
            continue

        val1, val2 = d1[key], d2[key]

        if isinstance(val1, dict) and isinstance(val2, dict):
            diffs.extend(recursive_dict_compare(val1, val2, new_path))
        elif isinstance(val1, np.ndarray) or isinstance(val2, np.ndarray):
            if not np.array_equal(val1, val2):
                diffs.append(
                    f"Array '{new_path}' differs. Shapes: {np.shape(val1)} vs {np.shape(val2)}"
                )
        elif val1 != val2:
            diffs.append(f"Value '{new_path}' differs: {val1} != {val2}")

    return diffs


def remap_labels(data: np.ndarray, mapping: dict[int, int] | None):
    """Remap integer labels in an array using a lookup table."""
    if not mapping:
        return data

    if np.min(data) < 0:
        raise ValueError("Label remapping expects non-negative labels.")

    max_val = int(max(data.max(), max(mapping.keys(), default=0), max(mapping.values(), default=0)))
    lut = np.arange(max_val + 1, dtype=data.dtype)

    for src, dst in mapping.items():
        if src <= max_val:
            lut[src] = dst

    return lut[data]


def parse_label_mapping(mapping_text: str | None):
    if not mapping_text:
        return None

    parsed = ast.literal_eval(mapping_text)
    if not isinstance(parsed, dict):
        raise ValueError("--label-mapping must be a Python dict, e.g. '{1: 2, 2: 1}'.")

    return {int(k): int(v) for k, v in parsed.items()}


def _is_integer_like(data: np.ndarray, check_count: int = 200000) -> bool:
    flat = data.reshape(-1)
    if flat.size == 0:
        return True

    stride = max(1, flat.size // check_count)
    sample = flat[::stride]
    return np.all(np.isfinite(sample)) and np.allclose(sample, np.round(sample), atol=1e-6)


def infer_nifti_mode(data1: np.ndarray, data2: np.ndarray) -> str:
    """Heuristic mode selection: mask for integer-like low-cardinality data, else image."""
    if not (_is_integer_like(data1) and _is_integer_like(data2)):
        return "image"

    unique1 = np.unique(data1.astype(np.int64))
    unique2 = np.unique(data2.astype(np.int64))
    if unique1.size <= 32 and unique2.size <= 32:
        return "mask"
    return "image"


def compare_nifti_mask(
    data1: np.ndarray,
    data2: np.ndarray,
    label_mapping: dict[int, int] | None = None,
):
    labels1 = np.round(data1).astype(np.int32)
    labels2 = np.round(data2).astype(np.int32)

    if label_mapping:
        labels2 = remap_labels(labels2, label_mapping)

    if np.array_equal(labels1, labels2):
        print("IDENTICAL")
        return {"changed_voxels": 0.0, "percent_changed": 0.0}

    matrix, changed_voxels = get_label_transition_matrix(labels1, labels2)
    total_voxels = labels1.size
    percent_changed = (changed_voxels / total_voxels) * 100.0

    print(f"DIFFERENT ({changed_voxels} voxels changed, {percent_changed:.4f}%)")
    if matrix is not None:
        print("\nLabel Swap Counts (Row=File1, Col=File2):")
        print(matrix)

    return {
        "changed_voxels": float(changed_voxels),
        "percent_changed": percent_changed,
    }


def compare_nifti_image(data1: np.ndarray, data2: np.ndarray, atol: float, rtol: float):
    if np.array_equal(data1, data2):
        print("IDENTICAL (exact)")
        return {
            "changed_voxels": 0.0,
            "percent_changed": 0.0,
            "max_abs_diff": 0.0,
            "mean_abs_diff": 0.0,
        }

    close_mask = np.isclose(data1, data2, atol=atol, rtol=rtol, equal_nan=True)
    changed_voxels = int(np.size(close_mask) - np.count_nonzero(close_mask))
    percent_changed = (changed_voxels / data1.size) * 100.0
    diff = np.abs(data1 - data2)
    max_abs_diff = float(np.nanmax(diff))
    mean_abs_diff = float(np.nanmean(diff))

    if changed_voxels == 0:
        print("IDENTICAL (within tolerance)")
    else:
        print(
            f"DIFFERENT ({changed_voxels} voxels changed, {percent_changed:.4f}%) | "
            f"max_abs_diff={max_abs_diff:.6g}, mean_abs_diff={mean_abs_diff:.6g}"
        )

    return {
        "changed_voxels": float(changed_voxels),
        "percent_changed": percent_changed,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
    }


def compare_nifti(
    file1_path: str,
    file2_path: str,
    nifti_mode: str,
    label_mapping: dict[int, int] | None,
    atol: float,
    rtol: float,
):
    data1, affine1 = load_medical_image(file1_path)
    data2, affine2 = load_medical_image(file2_path)

    if data1.shape != data2.shape:
        raise ValueError(f"SHAPE MISMATCH: {data1.shape} vs {data2.shape}")

    affine_equal = np.allclose(affine1, affine2, atol=atol, rtol=rtol)
    if not affine_equal:
        print("WARNING: Affine matrices differ.")

    mode = nifti_mode
    if mode == "auto":
        mode = infer_nifti_mode(data1, data2)
        print(f"Auto-selected NIfTI comparison mode: {mode}")

    if mode == "mask":
        result = compare_nifti_mask(data1, data2, label_mapping=label_mapping)
    else:
        result = compare_nifti_image(data1, data2, atol=atol, rtol=rtol)

    result["affine_equal"] = float(1.0 if affine_equal else 0.0)
    return result


def load_medical_image(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load voxel array and affine-like transform for NIfTI/MetaImage files."""
    lower = file_path.lower()

    if lower.endswith((".nii", ".nii.gz")):
        img = nib.load(file_path)
        return img.get_fdata(), img.affine

    if lower.endswith((".mha", ".mhd")):
        if sitk is None:
            raise RuntimeError(
                "SimpleITK is required to read .mha/.mhd files. "
                "Install it with: pip install SimpleITK"
            )

        image = sitk.ReadImage(file_path)
        data = sitk.GetArrayFromImage(image)

        direction = np.array(image.GetDirection(), dtype=np.float64).reshape(3, 3)
        spacing = np.array(image.GetSpacing(), dtype=np.float64)
        origin = np.array(image.GetOrigin(), dtype=np.float64)
        affine = np.eye(4, dtype=np.float64)
        affine[:3, :3] = direction @ np.diag(spacing)
        affine[:3, 3] = origin
        return data, affine

    raise ValueError(f"Unsupported medical image type: {file_path}")


def compare_npz(file1_path: str, file2_path: str, tolerance: float = 1e-5):
    npz1 = np.load(file1_path)
    npz2 = np.load(file2_path)

    keys1, keys2 = set(npz1.files), set(npz2.files)
    if keys1 != keys2:
        raise ValueError(f"KEY MISMATCH: {keys1} vs {keys2}")

    key = "probabilities" if "probabilities" in npz1.files else npz1.files[0]
    arr1 = npz1[key]
    arr2 = npz2[key]

    if np.allclose(arr1, arr2, atol=tolerance, rtol=0.0):
        print("IDENTICAL (within tolerance)")
        return {
            "max_abs_diff": 0.0,
            "mean_abs_diff": 0.0,
            "changed_voxels": 0.0,
        }

    diff = np.abs(arr1 - arr2)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    print(f"DIFFERENT: max_abs_diff={max_diff:.6g}, mean_abs_diff={mean_diff:.6g}")

    changed_voxels = 0
    if arr1.ndim >= 2:
        pred1 = np.argmax(arr1, axis=0)
        pred2 = np.argmax(arr2, axis=0)
        changed_voxels = int(np.sum(pred1 != pred2))
        if changed_voxels:
            print(f"Resulting argmax segmentation changed in {changed_voxels} voxels.")

    return {
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "changed_voxels": float(changed_voxels),
    }


def compare_pkl(file1_path: str, file2_path: str):
    with open(file1_path, "rb") as file1, open(file2_path, "rb") as file2:
        data1 = pickle.load(file1)
        data2 = pickle.load(file2)

    diffs = recursive_dict_compare(data1, data2)
    if not diffs:
        print("IDENTICAL")
        return {"diff_count": 0.0}

    preview = "\n".join(diffs[:5]) + ("\n..." if len(diffs) > 5 else "")
    print(f"DIFFERENT ({len(diffs)} keys):\n{preview}")
    return {"diff_count": float(len(diffs))}


def compare_file_pair(
    path_a: str,
    path_b: str,
    nifti_mode: str,
    label_mapping: dict[int, int] | None,
    atol: float,
    rtol: float,
):
    lower = path_a.lower()
    if lower.endswith((".nii", ".nii.gz", ".mha", ".mhd")):
        return compare_nifti(
            path_a,
            path_b,
            nifti_mode=nifti_mode,
            label_mapping=label_mapping,
            atol=atol,
            rtol=rtol,
        )
    if lower.endswith(".npz"):
        return compare_npz(path_a, path_b, tolerance=atol)
    if lower.endswith(".pkl"):
        return compare_pkl(path_a, path_b)

    print("Skipped (unsupported file type)")
    return None


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compare common files between two directories (NIfTI/NPZ/PKL)."
    )
    parser.add_argument("dir_a", help="First directory")
    parser.add_argument("dir_b", help="Second directory")
    parser.add_argument(
        "--nifti-mode",
        choices=["auto", "mask", "image"],
        default="auto",
        help="How to compare NIfTI files: label masks, image intensities, or infer automatically.",
    )
    parser.add_argument(
        "--label-mapping",
        default=None,
        help="Optional label mapping for mask mode, e.g. '{1: 2, 2: 1}'. Applied to files in dir_b.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for floating-point comparisons.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-8,
        help="Relative tolerance for floating-point comparisons.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".nii.gz", ".nii", ".mha", ".mhd", ".npz", ".pkl"],
        help="Only compare files with these suffixes.",
    )
    return parser


def should_compare(filename: str, extensions: list[str]) -> bool:
    lower = filename.lower()
    return any(lower.endswith(ext.lower()) for ext in extensions)


def main():
    parser = build_parser()
    args = parser.parse_args()

    label_mapping = parse_label_mapping(args.label_mapping)

    common_files = set(os.listdir(args.dir_a)) & set(os.listdir(args.dir_b))
    common_files = sorted([name for name in common_files if should_compare(name, args.extensions)])

    print(f"Found {len(common_files)} common files to compare.")
    print("=" * 50)

    aggregate = {
        "files_compared": 0,
        "changed_voxels_total": 0.0,
        "percent_changed_total": 0.0,
        "max_abs_diff_total": 0.0,
        "mean_abs_diff_total": 0.0,
        "pkl_diff_count_total": 0.0,
    }

    for filename in common_files:
        path_a = os.path.join(args.dir_a, filename)
        path_b = os.path.join(args.dir_b, filename)
        print(f"Comparing: {filename}")

        try:
            result = compare_file_pair(
                path_a,
                path_b,
                nifti_mode=args.nifti_mode,
                label_mapping=label_mapping,
                atol=args.atol,
                rtol=args.rtol,
            )
        except Exception as exc:
            print(f"ERROR: {exc}")
            print("-" * 50)
            continue

        if result is not None:
            aggregate["files_compared"] += 1
            aggregate["changed_voxels_total"] += result.get("changed_voxels", 0.0)
            aggregate["percent_changed_total"] += result.get("percent_changed", 0.0)
            aggregate["max_abs_diff_total"] += result.get("max_abs_diff", 0.0)
            aggregate["mean_abs_diff_total"] += result.get("mean_abs_diff", 0.0)
            aggregate["pkl_diff_count_total"] += result.get("diff_count", 0.0)

        print("-" * 50)

    compared = max(1, aggregate["files_compared"])
    summary = {
        "files_compared": aggregate["files_compared"],
        "avg_changed_voxels": aggregate["changed_voxels_total"] / compared,
        "avg_percent_changed": aggregate["percent_changed_total"] / compared,
        "avg_max_abs_diff": aggregate["max_abs_diff_total"] / compared,
        "avg_mean_abs_diff": aggregate["mean_abs_diff_total"] / compared,
        "avg_pkl_diff_count": aggregate["pkl_diff_count_total"] / compared,
    }

    print("SUMMARY REPORT:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
