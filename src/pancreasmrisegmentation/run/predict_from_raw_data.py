import inspect
import os
from copy import deepcopy
from typing import Union, List

import torch
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    isfile,
    maybe_mkdir_p,
    isdir,
    save_json,
)

from nnunetv2.configuration import default_num_processes
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder
from pancreasmrisegmentation.umamba.inference.predict_from_raw_data import (
    nnUNetPredictor as umamba_nnUNetPredictor,
)
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


class nnUNetPredictorWrapper:
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
        part_id: int = 0,
        file_ending: str = ".nii.gz",
    ):
        """
        This is nnU-Net's default function for making predictions. It works best for batch predictions
        (predicting many images at once).
        """
        if file_ending != ".nii.gz":
            rw_map = {"NibabelIOWithReorient": ""}

        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_folder = output_folder_or_list_of_truncated_output_files
        elif isinstance(output_folder_or_list_of_truncated_output_files, list):
            output_folder = os.path.dirname(
                output_folder_or_list_of_truncated_output_files[0]
            )
        else:
            output_folder = None

        ########################
        # let's store the input arguments so that its clear what was used to generate the prediction
        if output_folder is not None:
            my_init_kwargs = {}
            for k in inspect.signature(self.predict_from_files).parameters.keys():
                my_init_kwargs[k] = locals()[k]
            my_init_kwargs = deepcopy(
                my_init_kwargs
            )  # let's not unintentionally change anything in-place. Take this as a
            recursive_fix_for_json_export(my_init_kwargs)
            maybe_mkdir_p(output_folder)
            save_json(
                my_init_kwargs, join(output_folder, "predict_from_raw_data_args.json")
            )

            # we need these two if we want to do things with the predictions like for example apply postprocessing
            save_json(
                self.predictor.dataset_json,
                join(output_folder, "dataset.json"),
                sort_keys=False,
            )
            save_json(
                self.predictor.plans_manager.plans,
                join(output_folder, "plans.json"),
                sort_keys=False,
            )
        #######################

        # sort out input and output filenames
        (
            list_of_lists_or_source_folder,
            output_filename_truncated,
            seg_from_prev_stage_files,
        ) = self._manage_input_and_output_lists(
            list_of_lists_or_source_folder,
            output_folder_or_list_of_truncated_output_files,
            folder_with_segs_from_prev_stage,
            overwrite,
            part_id,
            num_parts,
            save_probabilities,
            file_ending,
        )
        if len(list_of_lists_or_source_folder) == 0:
            return

        data_iterator = (
            self.predictor._internal_get_data_iterator_from_lists_of_filenames(
                list_of_lists_or_source_folder,
                seg_from_prev_stage_files,
                output_filename_truncated,
                num_processes_preprocessing,
            )
        )

        return self.predictor.predict_from_data_iterator(
            data_iterator, save_probabilities, num_processes_segmentation_export
        )

    def _manage_input_and_output_lists(
        self,
        list_of_lists_or_source_folder: Union[str, List[List[str]]],
        output_folder_or_list_of_truncated_output_files: Union[None, str, List[str]],
        folder_with_segs_from_prev_stage: str = None,
        overwrite: bool = True,
        part_id: int = 0,
        num_parts: int = 1,
        save_probabilities: bool = False,
        file_ending: str = ".nii.gz",
    ):
        if isinstance(list_of_lists_or_source_folder, str):
            list_of_lists_or_source_folder = create_lists_from_splitted_dataset_folder(
                list_of_lists_or_source_folder, file_ending
            )
            print(list_of_lists_or_source_folder)
        print(
            f"There are {len(list_of_lists_or_source_folder)} cases in the source folder"
        )
        list_of_lists_or_source_folder = list_of_lists_or_source_folder[
            part_id::num_parts
        ]
        caseids = [
            os.path.basename(i[0])[: -(len(file_ending) + 5)]
            for i in list_of_lists_or_source_folder
        ]
        print(
            f"I am process {part_id} out of {num_parts} (max process ID is {num_parts - 1}, we start counting with 0!)"
        )
        print(f"There are {len(caseids)} cases that I would like to predict")

        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_filename_truncated = [
                join(output_folder_or_list_of_truncated_output_files, i)
                for i in caseids
            ]
        else:
            output_filename_truncated = output_folder_or_list_of_truncated_output_files

        seg_from_prev_stage_files = [
            join(folder_with_segs_from_prev_stage, i + file_ending)
            if folder_with_segs_from_prev_stage is not None
            else None
            for i in caseids
        ]
        # remove already predicted files form the lists
        if not overwrite and output_filename_truncated is not None:
            tmp = [isfile(i + file_ending) for i in output_filename_truncated]
            if save_probabilities:
                tmp2 = [isfile(i + ".npz") for i in output_filename_truncated]
                tmp = [i and j for i, j in zip(tmp, tmp2)]
            not_existing_indices = [i for i, j in enumerate(tmp) if not j]

            output_filename_truncated = [
                output_filename_truncated[i] for i in not_existing_indices
            ]
            list_of_lists_or_source_folder = [
                list_of_lists_or_source_folder[i] for i in not_existing_indices
            ]
            seg_from_prev_stage_files = [
                seg_from_prev_stage_files[i] for i in not_existing_indices
            ]
            print(
                f"overwrite was set to {overwrite}, so I am only working on cases that haven't been predicted yet. "
                f"That's {len(not_existing_indices)} cases."
            )
        return (
            list_of_lists_or_source_folder,
            output_filename_truncated,
            seg_from_prev_stage_files,
        )


def predict_entry_point_modelfolder():
    import argparse

    parser = argparse.ArgumentParser(
        description="Use this to run inference with nnU-Net. This function is used when "
        "you want to manually specify a folder containing a trained nnU-Net "
        "model. This is useful when the nnunet environment variables "
        "(nnUNet_results) are not set."
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
    parser.add_argument(
        "--file_ending",
        type=str,
        required=False,
        default=".nii.gz",
        help="File ending of the files to predict. Default: .nii.gz",
    )

    print(
        "\n#######################################################################\nPlease cite the following paper "
        "when using nnU-Net:\n"
        "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
        "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
        "Nature methods, 18(2), 203-211.\n#######################################################################\n"
    )

    args = parser.parse_args()
    args.f = [i if i == "all" else int(i) for i in args.f]

    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    assert args.device in ["cpu", "cuda", "mps"], (
        f"-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}."
    )
    if args.device == "cpu":
        # let's allow torch to use hella threads
        import multiprocessing

        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device("cpu")
    elif args.device == "cuda":
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device("cuda")
    else:
        device = torch.device("mps")

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
