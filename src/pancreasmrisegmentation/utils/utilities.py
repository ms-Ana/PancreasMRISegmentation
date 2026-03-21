import torch
import os
import numpy as np
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from pathlib import Path


def define_device(device: str):
    assert device in ["cpu", "cuda", "mps"], (
        f"-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {device}."
    )
    if device == "cpu":
        # let's allow torch to use hella threads
        import multiprocessing

        torch.set_num_threads(multiprocessing.cpu_count())
        return torch.device("cpu")
    elif device == "cuda":
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        return torch.device("cuda")
    else:
        return torch.device("mps")


def define_available_folds(model_dir: str):
    if os.path.isdir(os.path.join(model_dir, "fold_all")):
        return ["all"]
    available_folds = [
        fold
        for fold in range(5)
        if os.path.isdir(os.path.join(model_dir, f"fold_{fold}"))
    ]
    if not available_folds:
        raise RuntimeError(
            f"No fold directories found in {model_dir}. Expected fold_0, fold_1, ..., fold_4 or fold_all."
        )
    return available_folds


def save_segmentation(
    segmentation: np.ndarray,
    output_folder: Path,
    input_filename: str,
    properties: dict,
    prompt_name: str = None,
    suffix: str = ".nii.gz",
) -> None:
    """
    Save segmentation mask to file.

    Args:
        segmentation: Segmentation array to save.
        output_folder: Output folder path.
        input_filename: Original input filename (without extension).
        properties: Image properties from the reader.
        prompt_name: Optional prompt name to include in filename.
        suffix: File extension to use.
    """
    if prompt_name:
        # Clean prompt name for filename
        safe_name = "".join(
            c if c.isalnum() or c in (" ", "_") else "_" for c in prompt_name
        )
        safe_name = safe_name.replace(" ", "_")
        output_file = output_folder / f"{input_filename}_{safe_name}{suffix}"
    else:
        output_file = output_folder / f"{input_filename}{suffix}"

    # Use NIfTI writer
    reader_writer = NibabelIOWithReorient()
    reader_writer.write_seg(segmentation, str(output_file), properties)
    print(f"Saved segmentation to: {output_file}")
