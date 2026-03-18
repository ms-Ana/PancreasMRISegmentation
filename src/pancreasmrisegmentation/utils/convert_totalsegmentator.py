
import os
from pathlib import Path
import pandas as pd
import click
import numpy as np
import SimpleITK as sitk
import json 
from tqdm import tqdm

KEEP_SEGMENTATIONS  = ["aorta", "colon", "duodenum", "esophagus", "gallbladder", "heart", "kidney_left", "kidney_right", "liver", "lung_left", "lung_right", "pancreas", "portal_vein_and_splenic_vein", "spleen", "stomach", "urinary_bladder"]

def remove_not_listed_segmentations(folder_path: str, list_of_segmentations: list[str]):
    for ss in os.listdir(folder_path):
        if ss != "meta.csv":
            for ff in os.listdir(os.path.join(folder_path, f"{ss}/segmentations")):
                if ff.split(".")[0] not in list_of_segmentations:
                   os.remove(os.path.join(folder_path, f"{ss}/segmentations", ff))

def combine_segmentation_to_file(segmentations: list[str], 
                                 label_mapping: dict[str, int],
                                 output_path: str
                                 ):
    if not segmentations:
        raise ValueError("No segmentation files provided.")

    first_image = sitk.ReadImage(segmentations[0])
    first_array = sitk.GetArrayFromImage(first_image)

    max_label_value = max(label_mapping.values(), default=0)
    output_dtype = np.uint8 if max_label_value <= np.iinfo(np.uint8).max else np.uint16
    combined_array = np.zeros_like(first_array, dtype=output_dtype)

    def _segmentation_key(file_path: str) -> str:
        return Path(file_path).stem.split(".")[0]

    for segmentation_path in segmentations:
        key = _segmentation_key(segmentation_path)
        if key not in label_mapping:
            raise KeyError(f"Segmentation '{key}' is missing in label_mapping.")

        image = sitk.ReadImage(segmentation_path)
        if image.GetSize() != first_image.GetSize():
            raise ValueError(
                f"Size mismatch for '{segmentation_path}': "
                f"{image.GetSize()} != {first_image.GetSize()}"
            )

        array = sitk.GetArrayFromImage(image)
        mask = array > 0
        combined_array[mask] = label_mapping[key]

    output_image = sitk.GetImageFromArray(combined_array)
    output_image.CopyInformation(first_image)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(output_image, output_path)


@click.command()
@click.option("--input-folder", type=str, required=True, help="Path to the input folder containing the Totalsegmentator dataset.")
@click.option("--output-folder", type=str, required=True, help="Path to the output folder  where the converted dataset will be saved.")
@click.option("--meta-file", type=str, required=True, help="Path to the meta.csv file containing the metadata for the dataset.")
def convert_totalsegmentator_to_unet_dataset(
    input_folder: str,
    output_folder: str,
    meta_file: str | Path,
    list_of_segmentations: list[str] = KEEP_SEGMENTATIONS
):
    remove_not_listed_segmentations(input_folder, list_of_segmentations)
    label_mapping = {segmentation: idx + 1 for idx, segmentation in enumerate(list_of_segmentations)}
    meta_df = pd.read_csv(meta_file, sep=';')

    for info in tqdm(meta_df.itertuples(), total=len(meta_df), desc="Processing dataset"):
        try:
            if info.split == "train":
                segmentation_output = os.path.join(output_folder, "labelsTr", 
                                                    f"{info.image_id}.nii.gz")
                combine_segmentation_to_file([os.path.join(input_folder, f"{info.image_id}/segmentations/{seg}.nii.gz") for seg in list_of_segmentations], 
                                            label_mapping,
                                            segmentation_output
                                            )
                os.rename(os.path.join(input_folder, f"{info.image_id}/mri.nii.gz"),
                        os.path.join(output_folder, "imagesTr", f"{info.image_id}_0000.nii.gz"))
            elif info.split == "test":
                segmentation_output = os.path.join(output_folder, "labelsTs", 
                                                    f"{info.image_id}.nii.gz")
                combine_segmentation_to_file([os.path.join(input_folder, f"{info.image_id}/segmentations/{seg}.nii.gz") for seg in list_of_segmentations], 
                                            label_mapping,
                                            segmentation_output
                                            )
                os.rename(os.path.join(input_folder, f"{info.image_id}/mri.nii.gz"),
                        os.path.join(output_folder, "imagesTs", f"{info.image_id}_0000.nii.gz"))
        except Exception as e:
            print(f"Error processing {info.image_id}: {e}")
            
    with open(os.path.join(output_folder, "dataset.json"), "w") as f:
        json_content = {
            "name": "TotalsegmentatorMRI",
            "description": "Dataset converted from Totalsegmentator to UNet format",
            "numTraining": len(meta_df[meta_df['split'] == 'train']),
            "numTest": len(meta_df[meta_df['split'] == 'test']),
            "labels": label_mapping
        }
        json.dump(json_content, f, indent=4)

if __name__ == "__main__":
    convert_totalsegmentator_to_unet_dataset()
    

