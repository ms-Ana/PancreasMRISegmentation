import requests
from tqdm import tqdm
import os
import zipfile
import click
import glob
import shutil
from huggingface_hub import snapshot_download as _snaphot_download
from pancreasmrisegmentation.utils.settings import WEIGHTS_SETTINGS

def _flatten_weights(target_dir):
    """
    Moves fold_ folders and root-level files to the target_dir root,
    handling the deep nesting without double-moving files.
    """
    search_pattern = os.path.join(target_dir, "**", "fold_*")
    found_folds = [
        f for f in glob.glob(search_pattern, recursive=True) if os.path.isdir(f)
    ]

    all_files = glob.glob(os.path.join(target_dir, "**", "*.*"), recursive=True)
    loose_files = [f for f in all_files if "/fold_" not in f and os.path.isfile(f)]

    items_to_move = found_folds + loose_files

    for item in items_to_move:
        name = os.path.basename(item)
        destination = os.path.join(target_dir, name)

        if os.path.abspath(item) == os.path.abspath(destination):
            continue

        if os.path.exists(destination):
            if os.path.isdir(destination):
                shutil.rmtree(destination)
            else:
                os.remove(destination)

        shutil.move(item, destination)

    for entry in os.listdir(target_dir):
        full_path = os.path.join(target_dir, entry)
        if os.path.isdir(full_path) and not entry.startswith("fold_"):
            shutil.rmtree(full_path)

    print(f"Cleanup complete for {target_dir}")


def snaphot_download(link, download_dir, model_name):
    _snaphot_download(
        repo_id=link,
        allow_patterns=[f"{model_name}/*", "*.json"],
        local_dir=download_dir,
    )


def gdown_download(url, output):
    """
    Download a file from Google Drive using gdown
    """
    import gdown

    gdown.download_folder(id=url, output=output, quiet=False)


def request_download(url, output_dir):
    """
    url: The download link
    output_dir: The directory where you want the unzipped contents
    """
    temp_zip = "TEMP_WEIGHTS.zip"

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for 404/500 errors

        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024

        with tqdm(
            total=total_size, unit="B", unit_scale=True, desc="Downloading"
        ) as progress_bar:
            with open(temp_zip, "wb") as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)

        if not os.path.exists(temp_zip):
            raise FileNotFoundError(f"Failed to create {temp_zip}")

        os.makedirs(output_dir, exist_ok=True)

        print(f"Unzipping to {output_dir}...")
        with zipfile.ZipFile(temp_zip, "r") as z:
            z.extractall(output_dir)
        _flatten_weights(output_dir)
        os.remove(temp_zip)

    except Exception as e:
        if os.path.exists(temp_zip):
            os.remove(temp_zip)
        raise RuntimeError(f"Error during request_download: {e}")


DOWNLOAD_FUNCTIONS = {
    "gdown": gdown_download,
    "requests": request_download,
    "snapshot": snaphot_download,
}



@click.command()
@click.option(
    "--weights_dir",
    type=click.Path(),
    default=".model_weights",
    required=False,
    help="Directory to save or look for weights",
)
@click.option(
    "--redownload",
    is_flag=True,
    default=False,
    help="Redownload weights even if they already exist",
)
@click.option(
    "--model",
    type=str,
    help="Which model weights to download, if not specified, no weights will be downloaded. Can specify multiple models by repeating the --model option. Can be one of "
    + ", ".join(WEIGHTS_SETTINGS.keys())
    + ", or a custom model name if used with --link and --download-function options",
    required=False,
    multiple=True,
)
@click.option(
    "--link",
    type=str,
    required=False,
    help="Download all model weights",
)
@click.option(
    "--download-function",
    type=click.Choice(list(DOWNLOAD_FUNCTIONS.keys())),
    help="What download function to use for downloading weights",
)
@click.option(
    "--all",
    is_flag=True,
    default=False,
    help="Download all model weights (overrides --model and --link options)",
)
def download_weights(
    weights_dir: str,
    redownload: bool,
    model: tuple[str],
    link: str = None,
    download_function: str = None,
    all: bool = False,
):
    """
    download the weights automatically
    """
    os.makedirs(weights_dir, exist_ok=True)

    if all:
        to_download = WEIGHTS_SETTINGS
    else:
        for m in model:
            if m not in WEIGHTS_SETTINGS:
                raise ValueError(
                    f"Model {m} not found in predefined weights settings. Please provide a valid model name or use --link and --download-function options to specify custom weights."
                )
            to_download[m] = {"link": link, "download_func": download_function}
    for name, settings in to_download.items():
        final_path = os.path.join(weights_dir, name)

        if os.path.exists(final_path) and not redownload:
            click.echo(f"Skipping {name}, already exists.")
            continue

        _download_weights_for_model(
            name,
            settings["link"],
            settings["download_func"],
            final_path,
            settings.get("kwargs", {}),
        )


def _download_weights_for_model(
    model: str, link: str, download_function: str, final_path: str, kwargs: dict = {}
) -> tuple[str, str]:
    print(f"Downloading {model}...")
    download_function = DOWNLOAD_FUNCTIONS[download_function]
    try:
        download_function(link, final_path, **kwargs)
    except Exception as e:
        raise RuntimeError(f"Error downloading {model}: {e}")


if __name__ == "__main__":
    download_weights()
