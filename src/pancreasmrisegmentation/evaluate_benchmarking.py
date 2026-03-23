from omegaconf import OmegaConf
import click
import os
import json
from pancreasmrisegmentation.evaluate import evaluate_segmentation_performance
from pancreasmrisegmentation.config import RESULTS_DIR

@click.command()
@click.option(
    "--config",
    default="benchmarking_config.yaml",
    help="Path to the configuration file.",
)
@click.option(
    "--results_dir",
    default="benchmarking_results",
    help="Directory to save the benchmarking results.",
)
def evaluate_benchmarking(config: str, results_dir: str):
    conf = OmegaConf.load(config)
    print(OmegaConf.to_yaml(conf))

    
    for model_name, model in conf.models.items():
        for dataset_name, dataset in conf.datasets.items():
            try:
                result_path = os.path.join(results_dir, f"{model_name}_{dataset_name}.json")
                if os.path.exists(result_path):
                    print(f"Results for {model_name} on {dataset_name} already exist. Skipping evaluation.")
                    continue
                results = evaluate_segmentation_performance(
                    pred_dir=os.path.join(RESULTS_DIR, model_name, dataset_name),
                    gt_dir=dataset.labels,
                    label=model.label,
                    dataset_label=dataset.label if "label" in dataset else None,
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


