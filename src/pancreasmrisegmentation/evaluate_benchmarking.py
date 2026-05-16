from omegaconf import OmegaConf
import click
import os
import json
from pancreasmrisegmentation.evaluate import evaluate_segmentation_performance
from pancreasmrisegmentation.config import RESULTS_DIR
from pancreasmrisegmentation.utils.settings import MODEL_SETTINGS, DATASET_SETTINGS

@click.command()
@click.argument(
    "config",
    default="benchmarking_config.yaml",
)
@click.option(
    "--results_dir",
    default="benchmarking_results",
    help="Directory to save the benchmarking results.",
)
def evaluate_benchmarking(config: str, results_dir: str):
    conf = OmegaConf.load(config)
    print(OmegaConf.to_yaml(conf))
    os.makedirs(results_dir, exist_ok=True)

    
    for model_name, model in conf.models.items():
        for dataset_name, dataset in conf.datasets.items():
            try:
                result_path = os.path.join(results_dir, f"{model_name}-{dataset_name}.json")
                if os.path.exists(result_path):
                    print(f"Results for {model_name} on {dataset_name} already exist. Skipping evaluation.")
                    continue
                model_label = model.label if model and "label" in model else MODEL_SETTINGS[model_name]["label"] if model_name in MODEL_SETTINGS else None
                if not model_label:
                    raise AttributeError("Either model from default models should be used or 'label' should be specified.")
                results = evaluate_segmentation_performance(
                    pred_dir=os.path.join(RESULTS_DIR, model_name, dataset_name)
                    if not model or "path" not in model else model.path,
                    gt_dir=dataset.labels,
                    label=model_label,
                    dataset_label=dataset.label if dataset and "label" in dataset else 
                    DATASET_SETTINGS[dataset_name]["label"] if dataset_name in DATASET_SETTINGS else None,
                    verbose=True
                )
                print("Evaluation Metrics:")
                print(results)
                with open(result_path, "w") as f:
                    json.dump(results, f, indent=4)
    
            except Exception as e:
                print(f"Error occurred while evaluating {model_name} on {dataset_name}: {e}")
                

if __name__ == "__main__":
    evaluate_benchmarking()


