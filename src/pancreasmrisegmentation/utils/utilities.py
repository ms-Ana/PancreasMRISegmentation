import torch
import os

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
    available_folds = [fold for fold in range(5) if os.path.isdir(os.path.join(model_dir, f"fold_{fold}"))]
    if not available_folds:
        raise RuntimeError(f"No fold directories found in {model_dir}. Expected fold_0, fold_1, ..., fold_4 or fold_all.")
    return available_folds