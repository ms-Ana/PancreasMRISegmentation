import click 
import os 
from pancreasmrisegmentation.utils.download_weights import WEIGHTS_SETTINGS, download_weights
from pancreasmrisegmentation.config import MODEL_DIR

@click.command()
@click.option("--model_name",
              type="mr_segmentator" | "mri_segmenter" | "pansegnet_t1" | "pansegnet_t2" | "total_segmentator" | "umamba_bot" | "umamba_enc" | "umamba_nnunet" |  "umamba_segresnet" | "umamba_swinunetr" | "umamba_unetr" | "voxtell",
              default=None,
              help="")
@click.option(
    "--input",
    type=str,
    help=""
)
def inference(model_name, 
              input):
    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        download_weights(
            MODEL_DIR, 
            False, 
            model_name,
            WEIGHTS_SETTINGS[model_name]["link"], 
            WEIGHTS_SETTINGS[model_name]["download_func"], 
            model_path
        )
    
    