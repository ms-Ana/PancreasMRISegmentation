# Adapted from https://github.com/rsummers11/MRISegmenter/blob/main/mrisegmentator/download_weights.py


import requests
from tqdm import tqdm
import os
import zipfile
import click


@click.command()
@click.option(
    "--weights_dir",
    type=click.Path(),
    default=".mrisegmentator_weights",
    required=False,
    help="Directory to save or look for weights",
)
@click.option(
    "--redownload",
    is_flag=True,
    default=False,
    help="Redownload weights even if they already exist",
)
def download_weights(weights_dir, redownload):
    """
    download the weights automatically
    """
    url = "https://nihcc.app.box.com/index.php?rm=box_download_shared_file&shared_name=q6vl3015hteoufz7jll63u3hdqk79li7&file_id=f_1544045874167"

    filepath = weights_dir + "/MRISegmentator.zip"
    final_path = weights_dir + "/nnUNetTrainer__nnUNetPlans__3d_fullres_New"
    if os.path.exists(final_path) and not redownload:
        print("Weights are already downloaded at", final_path)
        return final_path
    if not os.path.exists(final_path) or redownload:
        print("saving weights to", weights_dir)
        # from https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024
        with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
            with open(filepath, "wb") as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
        if total_size != 0 and progress_bar.n != total_size:
            raise RuntimeError("Could not download weights")
    if os.path.exists(filepath):
        print("unzipping model weights")
        with zipfile.ZipFile(filepath, "r") as zip_ref:
            zip_ref.extractall(weights_dir)
            if os.path.exists(final_path):
                print(
                    "successfully unzipped weights and saved to",
                    final_path,
                    ". Cleaning up...",
                )
                os.remove(filepath)
                return final_path
    else:
        raise FileNotFoundError(
            f"for some reason the MRISegmentator.zip file was not found in your weights directory at: {weights_dir}"
        )


if __name__ == "__main__":
    download_weights()
