from omegaconf import OmegaConf
import click
import os
import time
from datetime import timedelta

from pancreasmrisegmentation.config import MODEL_DIR, RESULTS_DIR
from pancreasmrisegmentation.run.predict_from_raw_data import nnUNetPredictorWrapper
from pancreasmrisegmentation.utils.utilities import (
    define_device,
    define_available_folds,
)


@click.command()
@click.option(
    "--config",
    default="benchmarking_config.yaml",
    help="Path to the configuration file.",
)
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

        model_path = (
            model.path
            if "path" in model
            else os.path.join(MODEL_DIR, model_name)
            if "voxtell" not in model_name
            else os.path.join(MODEL_DIR, model_name, "voxtell_v1.1")
        )
        folds = model.folds if "folds" in model else define_available_folds(model_path)
        predictor.initialize_from_trained_model_folder(
            model_path, folds, "checkpoint_final.pth"
        )

        for dataset_name, dataset in conf.datasets.items():
            output_folder = os.path.join(RESULTS_DIR, model_name, dataset_name)
            os.makedirs(output_folder, exist_ok=True)

            print(f"Predicting {model_name} on {dataset_name}...")
            try:
                start_time = time.time()
                predictor.predict_from_files(
                    dataset.path,
                    output_folder,
                    save_probabilities=False,
                    overwrite=False,
                    num_processes_preprocessing=4,
                    num_processes_segmentation_export=4,
                    folder_with_segs_from_prev_stage=None,
                    num_parts=1,
                    part_id=0,
                )
                elapsed_time = time.time() - start_time
                print(
                    f"Finished predicting {model_name} on {dataset_name} in {timedelta(seconds=elapsed_time)}."
                )

            except Exception as e:
                print(
                    f"Error occurred while predicting {model_name} on {dataset_name}: {e}"
                )
                raise e
            finally:
                # clean up
                for meta_file in [
                    "predict_from_raw_data_args.json",
                    "plans.json",
                    "dataset.json",
                    "plans.pkl",
                    "postprocessing.json"
                ]:
                    path = os.path.join(output_folder, meta_file)
                    if os.path.exists(path):
                        os.remove(path)


if __name__ == "__main__":
    benchmarking()
