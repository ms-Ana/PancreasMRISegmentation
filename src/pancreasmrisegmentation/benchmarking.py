from omegaconf import OmegaConf
import click
import torch
import os
from pancreasmrisegmentation.config import MODEL_DIR, RESULTS_DIR
from pancreasmrisegmentation.run.predict_from_raw_data import nnUNetPredictorWrapper

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

@click.command()
@click.option('--config', default='benchmarking_config.yaml', help='Path to the configuration file.')
def benchmarking(config: str):
    conf = OmegaConf.load(config)
    print(OmegaConf.to_yaml(conf))

    device = define_device(conf.device)

    for model_name, model in conf.models.items():
        predictor = nnUNetPredictorWrapper(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        allow_tqdm=True,
        verbose_preprocessing=False,
        predictor=model.predictor,
    )
        
        model_path = model.path if "path" in model else os.path.join(MODEL_DIR, model_name)
        folds = model.folds if "folds" in model else define_available_folds(model_path)
        predictor.initialize_from_trained_model_folder(model_path, folds, "checkpoint_final.pth")

        for dataset_name, dataset in conf.datasets.items():
            output_folder = os.path.join(RESULTS_DIR, model_name, dataset_name)
            os.makedirs(output_folder, exist_ok=True)
            predictor.predict_from_files(
                dataset.path,
                output_folder,
                save_probabilities=False,
                overwrite=False,
                num_processes_preprocessing=3,
                num_processes_segmentation_export=3,
                folder_with_segs_from_prev_stage=None,
                num_parts=1,
                part_id=0,
                file_ending=dataset.file_ending,
            )

if __name__ == "__main__":
    benchmarking()