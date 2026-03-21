import torch
import numpy as np
from pathlib import Path
import click


def compare_checkpoints(model_dir1: str, model_dir2: str):
    """
    Compare weights in multiple nnUNet checkpoints across folds.
    """
    dir1, dir2 = Path(model_dir1), Path(model_dir2)

    # 1. Identify common folds present in both directories
    folds1 = sorted([d.name for d in dir1.glob("fold_*") if d.is_dir()])
    folds2 = sorted([d.name for d in dir2.glob("fold_*") if d.is_dir()])
    common_folds = set(folds1).intersection(set(folds2))

    if not common_folds:
        print(f"No matching folds found between {model_dir1} and {model_dir2}")
        return

    print(f"Comparing folds: {sorted(list(common_folds))}")
    overall_identical = True

    for fold in sorted(common_folds):
        path1 = dir1 / fold / "checkpoint_final.pth"
        path2 = dir2 / fold / "checkpoint_final.pth"

        if not path1.exists() or not path2.exists():
            print(f"Skipping {fold}: Checkpoint file missing in one directory.")
            continue

        # Load weights one fold at a time to save RAM
        ckpt1 = torch.load(path1, map_location="cpu", weights_only=False)
        ckpt2 = torch.load(path2, map_location="cpu", weights_only=False)

        # Print latest timestamp for verification (nnUNet structure)
        ts1 = ckpt1.get("logging", {}).get("epoch_end_timestamps", [-1])[-1]
        ts2 = ckpt2.get("logging", {}).get("epoch_end_timestamps", [-1])[-1]
        print(f"\n--- {fold} ---")
        print(f"Timestamp checkpoint 1: {ts1} | Timestamp checkpoint 2: {ts2}")

        sd1 = ckpt1["network_weights"]
        sd2 = ckpt2["network_weights"]

        if sd1.keys() != sd2.keys():
            print(f"Error: Mismatched layers in {fold}.")
            print(f"  Layers in checkpoint 1: {sorted(sd1.keys())}")
            print(f"  Layers in checkpoint 2: {sorted(sd2.keys())}")
            overall_identical = False
            continue

        # Compare weights
        fold_match = True
        for key in sd1.keys():
            w1, w2 = sd1[key].numpy(), sd2[key].numpy()

            if w1.shape != w2.shape:
                print(f"  Shape mismatch in {key}: {w1.shape} vs {w2.shape}")
                fold_match = False
            elif not np.allclose(w1, w2, atol=1e-8):
                print(f"  Weights differ in layer: {key}")
                fold_match = False

        if fold_match:
            print(f"  Result: Fold {fold} weights are identical.")
        else:
            overall_identical = False

    print("\n" + "=" * 30)
    if overall_identical:
        print("FINAL RESULT: All checked folds/layers are identical.")
    else:
        print("FINAL RESULT: Differences found between checkpoints.")


@click.command()
@click.argument(
    "model_dir1", type=click.Path(exists=True, file_okay=False), required=True
)
@click.argument(
    "model_dir2", type=click.Path(exists=True, file_okay=False), required=True
)
def main(model_dir1, model_dir2):
    compare_checkpoints(model_dir1, model_dir2)


if __name__ == "__main__":
    main()
