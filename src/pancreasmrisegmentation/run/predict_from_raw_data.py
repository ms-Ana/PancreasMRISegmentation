from typing import Union, List

from pancreasmrisegmentation.utils.utilities import define_device
import torch
from batchgenerators.utilities.file_and_folder_operations import (
    maybe_mkdir_p,
    isdir
)

from nnunetv2.configuration import default_num_processes
from pancreasmrisegmentation.umamba.inference.predict_from_raw_data import (
    nnUNetPredictor as umamba_nnUNetPredictor,
)
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


class nnUNetPredictorWrapper:
    ### This is a wrapper that allows us to use both the original nnUNetPredictor and the one from umamba with the same interface. This is useful for testing and benchmarking. It also allows us to easily switch between the two predictors without having to change the code that calls them.
    def __init__(
        self,
        tile_step_size: float = 0.5,
        use_gaussian: bool = True,
        use_mirroring: bool = True,
        perform_everything_on_device: bool = True,
        device: torch.device = torch.device("cuda"),
        verbose: bool = False,
        verbose_preprocessing: bool = False,
        allow_tqdm: bool = True,
        predictor: str = "nnUNetv2",
    ):
        if predictor == "nnUNetv2":
            self.predictor = nnUNetPredictor(
                tile_step_size=tile_step_size,
                use_gaussian=use_gaussian,
                use_mirroring=use_mirroring,
                perform_everything_on_device=perform_everything_on_device,
                device=device,
                verbose=verbose,
                verbose_preprocessing=verbose_preprocessing,
                allow_tqdm=allow_tqdm,
            )
        elif predictor == "umamba":
            self.predictor = umamba_nnUNetPredictor(
                tile_step_size=tile_step_size,
                use_gaussian=use_gaussian,
                use_mirroring=use_mirroring,
                perform_everything_on_device=perform_everything_on_device,
                device=device,
                verbose=verbose,
                verbose_preprocessing=verbose_preprocessing,
                allow_tqdm=allow_tqdm,
            )
        else:
            raise ValueError(f"Unknown predictor: {predictor}")

    def initialize_from_trained_model_folder(
        self,
        model_folder: str,
        folds: Union[int, List[int]],
        checkpoint_name: str = "checkpoint_final.pth",
    ):
        self.predictor.initialize_from_trained_model_folder(
            model_folder, folds, checkpoint_name
        )

    def predict_from_files(
        self,
        list_of_lists_or_source_folder: Union[str, List[List[str]]],
        output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
        save_probabilities: bool = False,
        overwrite: bool = True,
        num_processes_preprocessing: int = default_num_processes,
        num_processes_segmentation_export: int = default_num_processes,
        folder_with_segs_from_prev_stage: str = None,
        num_parts: int = 1,
        part_id: int = 0
    ):
        """
        This is nnU-Net's default function for making predictions. It works best for batch predictions
        (predicting many images at once).
        """

        self.predictor.predict_from_files(list_of_lists_or_source_folder=list_of_lists_or_source_folder,
                                          output_folder_or_list_of_truncated_output_files=output_folder_or_list_of_truncated_output_files,
                                          save_probabilities=save_probabilities,
                                          overwrite=overwrite,
                                          num_processes_preprocessing=num_processes_preprocessing,
                                          num_processes_segmentation_export=num_processes_segmentation_export,
                                          folder_with_segs_from_prev_stage=folder_with_segs_from_prev_stage,
                                          num_parts=num_parts,
                                          part_id=part_id)


def predict_entry_point_modelfolder():
    import argparse

    parser = argparse.ArgumentParser(
        description="Inference with nnU-Net. This script allows you to make predictions with a trained nnU-Net model on raw data."
    )
    parser.add_argument(
        "-i",
        type=str,
        required=True,
        help="input folder. Remember to use the correct channel numberings for your files (_0000 etc). "
        "File endings must be the same as the training dataset!",
    )
    parser.add_argument(
        "-o",
        type=str,
        required=True,
        help="Output folder. If it does not exist it will be created. Predicted segmentations will "
        "have the same name as their source images.",
    )
    parser.add_argument(
        "-m",
        type=str,
        required=True,
        help="Folder in which the trained model is. Must have subfolders fold_X for the different "
        "folds you trained",
    )
    parser.add_argument(
        "-f",
        nargs="+",
        type=str,
        required=False,
        default=(0, 1, 2, 3, 4),
        help="Specify the folds of the trained model that should be used for prediction. "
        "Default: (0, 1, 2, 3, 4)",
    )
    parser.add_argument(
        "-step_size",
        type=float,
        required=False,
        default=0.5,
        help="Step size for sliding window prediction. The larger it is the faster but less accurate "
        "the prediction. Default: 0.5. Cannot be larger than 1. We recommend the default.",
    )
    parser.add_argument(
        "--disable_tta",
        action="store_true",
        required=False,
        default=False,
        help="Set this flag to disable test time data augmentation in the form of mirroring. Faster, "
        "but less accurate inference. Not recommended.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Set this if you like being talked to. You will have "
        "to be a good listener/reader.",
    )
    parser.add_argument(
        "--save_probabilities",
        action="store_true",
        help='Set this to export predicted class "probabilities". Required if you want to ensemble '
        "multiple configurations.",
    )
    parser.add_argument(
        "--continue_prediction",
        "--c",
        action="store_true",
        help="Continue an aborted previous prediction (will not overwrite existing files)",
    )
    parser.add_argument(
        "-chk",
        type=str,
        required=False,
        default="checkpoint_final.pth",
        help="Name of the checkpoint you want to use. Default: checkpoint_final.pth",
    )
    parser.add_argument(
        "-npp",
        type=int,
        required=False,
        default=3,
        help="Number of processes used for preprocessing. More is not always better. Beware of "
        "out-of-RAM issues. Default: 3",
    )
    parser.add_argument(
        "-nps",
        type=int,
        required=False,
        default=3,
        help="Number of processes used for segmentation export. More is not always better. Beware of "
        "out-of-RAM issues. Default: 3",
    )
    parser.add_argument(
        "-prev_stage_predictions",
        type=str,
        required=False,
        default=None,
        help="Folder containing the predictions of the previous stage. Required for cascaded models.",
    )
    parser.add_argument(
        "-device",
        type=str,
        default="cuda",
        required=False,
        help="Use this to set the device the inference should run with. Available options are 'cuda' "
        "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
        "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_predict [...] instead!",
    )
    parser.add_argument(
        "--disable_progress_bar",
        action="store_true",
        required=False,
        default=False,
        help="Set this flag to disable progress bar. Recommended for HPC environments (non interactive "
        "jobs)",
    )
    parser.add_argument(
        "--predictor",
        type=str,
        required=False,
        default="nnUNetv2",
        help="Which predictor to use. Available options are 'nnUNetv2' and 'umamba'. Default: nnUNetv2. ",
    )
    args = parser.parse_args()
    args.f = [i if i == "all" else int(i) for i in args.f]

    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    device = define_device(args.device)

    predictor = nnUNetPredictorWrapper(
        tile_step_size=args.step_size,
        use_gaussian=True,
        use_mirroring=not args.disable_tta,
        perform_everything_on_device=True,
        device=device,
        verbose=args.verbose,
        allow_tqdm=not args.disable_progress_bar,
        verbose_preprocessing=args.verbose,
        predictor=args.predictor,
    )

    predictor.initialize_from_trained_model_folder(args.m, args.f, args.chk)
    predictor.predict_from_files(
        args.i,
        args.o,
        save_probabilities=args.save_probabilities,
        overwrite=not args.continue_prediction,
        num_processes_preprocessing=args.npp,
        num_processes_segmentation_export=args.nps,
        folder_with_segs_from_prev_stage=args.prev_stage_predictions,
        num_parts=1,
        part_id=0,
        file_ending=args.file_ending,
    )


if __name__ == "__main__":
    predict_entry_point_modelfolder()
