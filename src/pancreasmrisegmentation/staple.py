import SimpleITK as sitk 
from tqdm import tqdm 
import click 
import os 
import yaml 



def run_staple(images: list[sitk.Image], foreground_value: int | list[int] =1, 
               verbose: bool =False, save_statistics: bool =False) -> sitk.Image | tuple[sitk.Image, dict[int, dict[str, float]]]:
    segmentations = []
    if isinstance(foreground_value, int):
        foreground_value = [foreground_value] * len(images)

    reference_image = None

    for i, (label, image) in enumerate(zip(foreground_value, images)):
        binary_img = sitk.Cast(image == label, sitk.sitkUInt8)
        
        if i == 0:
            reference_image = binary_img
        else:
            binary_img.CopyInformation(reference_image)
            
        segmentations.append(binary_img)

    combined_mask = segmentations[0]
    for seg in segmentations[1:]:
        combined_mask = sitk.Or(combined_mask, seg)

    label_stats = sitk.LabelShapeStatisticsImageFilter()
    label_stats.Execute(combined_mask)
    
    if not label_stats.HasLabel(1):
        empty_img = sitk.Image(images[0].GetSize(), sitk.sitkUInt8)
        empty_img.CopyInformation(images[0])
        return empty_img
        
    bbox = label_stats.GetBoundingBox(1)
    start_idx = bbox[:3]
    size = bbox[3:]

    cropped_segs = []
    for seg in segmentations:
        roi = sitk.RegionOfInterest(seg, size=size, index=start_idx)
        cropped_segs.append(roi)

    staple_filter = sitk.STAPLEImageFilter()
    staple_filter.SetForegroundValue(1)
    probability_map = staple_filter.Execute(cropped_segs)
    cropped_consensus = sitk.Cast(probability_map > 0.5, sitk.sitkUInt8)

    full_size_consensus = sitk.Image(images[0].GetSize(), sitk.sitkUInt8)
    full_size_consensus.CopyInformation(images[0])
    
    final_consensus = sitk.Paste(full_size_consensus, cropped_consensus, size, [0,0,0], start_idx)
    observer_statistics = {}


    if verbose or save_statistics:
        sensitivities = staple_filter.GetSensitivity()
        specificities = staple_filter.GetSpecificity()
        iterations = staple_filter.GetElapsedIterations()

    if verbose:
        print(f"STAPLE converged after {iterations} iterations.")

        pbar = tqdm(enumerate(zip(sensitivities, specificities)), 
                    total=len(sensitivities), 
                    desc="Calculating Observer Performance")
        
        for i, (sens, spec) in pbar:
            tqdm.write(f"Observer {i}: Sensitivity = {sens:.4f}, Specificity = {spec:.4f}")
    if save_statistics:
        for i, (sens, spec) in enumerate(zip(sensitivities, specificities)):
            observer_statistics[i] = {"sensitivity": sens, "specificity": spec}
        return final_consensus, observer_statistics 
    return final_consensus

def get_segmentation_paths(segmentations_folders: list[str], strategy: str = "union") -> dict[str, list[str]]:
    folder_contents = {folder: set(os.listdir(folder)) for folder in segmentations_folders}
    
    if strategy == "union":
        target_files = set.union(*folder_contents.values())
    elif strategy == "intersection":
        target_files = set.intersection(*folder_contents.values())
    else:
        raise ValueError(f"Invalid strategy: '{strategy}'. Must be 'union' or 'intersection'.")

    full_paths = {}
    for filename in target_files:
        full_paths[filename] = [
            os.path.join(folder, filename)
            for folder, files_in_folder in folder_contents.items()
            if filename in files_in_folder
        ]

    return full_paths

@click.command()
@click.argument("config_file", type=click.Path(exists=True))
def main(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    os.makedirs(config["output_folder"], exist_ok=True)

    with open(os.path.join(config["output_folder"], "config.yaml"), "w") as f:
        yaml.dump(config, f)

    segmentations_folders = [seg['path'] for seg in config['segmentations'].values()]
    foreground_values = [seg['label'] for seg in config['segmentations'].values()]

    segmentation_paths = get_segmentation_paths(segmentations_folders, strategy=config["strategy"])
    verbose = config.get("verbose", False)
    save_statistics = config.get("save_statistics", False)
    global_stats = {}
    for filename, paths in segmentation_paths.items():
        try:
            if verbose:
                print(f"Processing {filename} with {len(paths)} segmentations...")
            images = [sitk.ReadImage(p) for p in paths]
            if save_statistics:
                consensus_mask, stats = run_staple(images, foreground_value=foreground_values, verbose=verbose, save_statistics=save_statistics)
                global_stats[filename] = {segmentation_name: stat for segmentation_name, stat in zip(config['segmentations'].keys(), stats.values())}
            else:
                consensus_mask = run_staple(images, foreground_value=foreground_values, verbose=verbose, save_statistics=save_statistics)
            output_path = os.path.join(config["output_folder"], filename)
            sitk.WriteImage(consensus_mask, output_path)
            if verbose:
                print(f"Saved consensus segmentation to {output_path}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    if save_statistics:
        stats_output_path = os.path.join(config["output_folder"], "observer_statistics.yaml")
        with open(stats_output_path, 'w') as f:
            yaml.dump(global_stats, f)
        if verbose:
            print(f"Saved observer statistics to {stats_output_path}")

if __name__ == "__main__":
    main()