# PancreasMRISegmentation

Comprehensive benchmarking and inference toolkit for MRI segmentation models.

This repository provides a unified interface to run multiple model families on common pancreas MRI datasets, then evaluate and compare results.

It can also be adapted for any abdominal organ by setting the correct dataset_label and model label.

## What Is Included

- Unified inference wrapper for:
	- nnUNetv2-style models
	- uMamba variants
	- nnUNetv1/PanSegNet-style predictors
	- VoxTell predictor
- Benchmark orchestration from a YAML config
- Evaluation scripts for Dice/surface-based metrics and aggregate reporting
- Utilities for downloading pretrained weights

## Requirements

- Linux
- Python >= 3.12 and < 3.15
- CUDA GPU recommended for practical runtime

## Install

### Option A: PDM (recommended)

```bash
cd /home/derzhana/MEN1/PancreasMRISegmentation
pdm install
```

Install optional uMamba extras:

```bash
pdm add "PancreasMRISegmentation[umamba]"
```

### Option B: pip editable install

```bash
cd /home/derzhana/MEN1/PancreasMRISegmentation
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Environment Variables

The code reads paths from .env:

```dotenv
MODEL_DIR=/absolute/path/to/model_weights
RESULTS_DIR=/absolute/path/to/results
```


## Expected Input Format

For nnUNet-compatible inference, put images in an input folder with channel suffixes, for example:

```text
imagesTs/
	case_001_0000.nii.gz
	case_002_0000.nii.gz
```

Output segmentations are written with the same case IDs.

## Download Pretrained Weights

Download one or more known model weights:

```bash
pdm run python -m pancreasmrisegmentation.utils.download_weights \
	--weights_dir src/pancreasmrisegmentation/.model_weights \
	--model mri_segmentator \
	--model pansegnet_t2
```

Download all predefined weights:

```bash
pdm run python -m pancreasmrisegmentation.utils.download_weights \
	--weights_dir src/pancreasmrisegmentation/.model_weights \
	--all
```

## Single-Model Inference

Console entry point:

```bash
pdm run inference \
	-i /path/to/imagesTs \
	-o /path/to/predictions \
	-m /path/to/model_folder \
	--predictor nnUNetv2 \
	-f 0 1 2 3 4 \
	--device cuda
```

Important notes:

- Use model folders that contain fold directories (for example fold_0 ... fold_4).
- Supported predictors include nnUNetv2, umamba, nnUNetv1, and voxtell.
- If needed, disable test-time augmentation with --disable_tta.

## Benchmark Multiple Models and Datasets

1. Edit benchmark config:

- File: src/pancreasmrisegmentation/benchmarking_config.yaml
- Define datasets.path and datasets.labels
- Define each model with predictor and label

2. Run benchmark:

```bash
pdm run benchmark --config src/pancreasmrisegmentation/benchmarking_config.yaml
```

The benchmark writes predictions under RESULTS_DIR/{model_name}/{dataset_name}.

## Evaluate Predictions

Evaluate one prediction folder against labels:

```bash
pdm run python -m pancreasmrisegmentation.evaluate \
	--pred_dir /path/to/predictions \
	--gt_dir /path/to/labels \
	--label 1 \
	--dataset_label 1 \
	--save_path /path/to/metrics.json
```

Evaluate all model/dataset pairs from the benchmark config:

```bash
pdm run python -m pancreasmrisegmentation.evaluate_benchmarking \
	--config src/pancreasmrisegmentation/benchmarking_config.yaml \
	--results_dir benchmarking_results
```

## Adjusting to Other Organs and Datasets

To use this pipeline for organs other than pancreas or for new datasets:

1. Prepare data in nnUNet-compatible format.
	- Input images should follow channel naming (for example case_001_0000.nii.gz).
	- Ground-truth masks must be aligned with the input images (same case IDs, spacing, and shape).

2. Update dataset entries in src/pancreasmrisegmentation/benchmarking_config.yaml.
	- Set datasets.<name>.path to your image folder.
	- Set datasets.<name>.labels to the corresponding label folder.
	- Set datasets.<name>.label to the label value used in that dataset for your target organ.

3. Update model entries in src/pancreasmrisegmentation/benchmarking_config.yaml.
	- Set models.<name>.predictor to the correct predictor backend (nnUNetv2, umamba, nnUNetv1, or voxtell).
	- Set models.<name>.label to the class value produced by that model for your target organ.

4. Run inference and evaluation.
	- For single-folder evaluation, use --label (model output label) and --dataset_label (ground-truth label).
	- For benchmarking evaluation, ensure model.label and dataset.label are set correctly in the config.

If labels are mismatched between predictions and ground truth, metrics will be misleading even if segmentations look visually correct.

## Testing Results on the Panther Dataset (92 Cases)
Data available: https://zenodo.org/records/15192302
| model_name         |   mean_volumetric_dice |   mean_surface_dice |   mean_hausdorff95 |   mean_masd |    rmse |
|:-------------------|-----------------------:|--------------------:|-------------------:|------------:|--------:|
| mr_segmentator     |         **0.817645**   |           0.918493  |           11.5811  |  **1.82541**| 38493.5 | 
| mri_segmenter      |             0.796488   |           0.923081  |           12.9455  |     2.0095  |**29448.3**|
| pansegnet_t1       |             0.789289   |         **0.935621**|         **10.7063**|     1.82616 | 32689.6 |
| pansegnet_t2       |             0.00372444 |           0.0196079 |           695.92   |   646.54    | 95968.1 |
| total_segmentator  |             0.708104   |           0.848722  |           17.1272  |     3.01063 | 45805.3 |
| umamba_bot         |             0.733533   |           0.873526  |           17.6466  |     2.71619 | 35372.7 |
| umamba_enc         |             0.737339   |           0.872953  |           18.8495  |     2.95857 | 32876.3 |
| umamba_nnunet      |             0.719131   |           0.848584  |           20.7637  |     3.235   | 35188.3 |
| umamba_segresnet   |             0.633627   |           0.785301  |           28.2919  |     4.80271 | 41404.6 |
| umamba_swinunetr   |             0.591129   |           0.724034  |           38.7543  |     6.62151 | 41770.5 |
|:-------------------|-----------------------:|--------------------:|-------------------:|------------:|--------:|
| voxtell            |             0.239834   |           0.285989  |          229.355   |    94.1497  | 78338.8 |

### STAPLE algorithm results (Panther)
| model_name         |   mean_volumetric_dice |   mean_surface_dice |   mean_hausdorff95 |   mean_masd |    rmse |
|:-------------------|-----------------------:|--------------------:|-------------------:|------------:|--------:|
| staple(all)        |             0.801702   |           0.921682  |           13.1322  |     1.97725 | 30997.3 |
| staple_2           |             0.794323   |           0.911073  |           12.0384  |     1.94826 | 40831.5 |
| staple_3   		 |             0.834607   |           0.941814  |            9.46269 |     1.53    | 32749   |
| staple_3_1         |             0.810652   |           0.918725  |           11.5169  |     1.82426 | 37501.6 |
| staple_4           |             **0.835202**|        **0.943715**|          **9.11981**|  **1.50494**| **31866.9**|
| staple_5           |             0.827393   |           0.941717  |            9.61874 |     1.57    | 29719.9 |
| staple_6           |             0.818302   |           0.934624  |           11.4213  |     1.72059 | 30479.2 |
* _2 - mri_segmenter, mr_segmentator
* _3 - mri_segmenter, mr_segmentator, pansegnet_t1
* _3_1 - mri_segmenter, mr_segmentator, total_segmentator
* _4 - mri_segmenter, mr_segmentator, pansegnet_t1, total_segmentator
* _5 - mri_segmenter, mr_segmentator, pansegnet_t1, total_segmentator, umamba_enc
* _6 - mri_segmenter, mr_segmentator, pansegnet_t1, total_segmentator, umamba_enc, umamba_bot

## MRI Segmenter
| model_name         |   mean_volumetric_dice |   mean_surface_dice |   mean_hausdorff95 |   mean_masd |    rmse |
|:-------------------|-----------------------:|--------------------:|-------------------:|------------:|--------:|
| mri_segmenter      |             0.796488   |          0.923081   |           12.9455  |     2.0095  | 29448.3 |
| mri_segmenter0     |             0.796154   |          0.92136    |           13.1898  |     2.02767 | **29100.1** |
| mri_segmenter1     |             0.795457   |          0.92206    |           12.933   |     2.04953 | 29716.7 |
| mri_segmenter2     |             0.794414   |          0.920105   |           13.1079  |     2.0212  | 29216.8 |
| mri_segmenter3     |             0.792031   |          0.920704   |           13.1063  |     2.04918 | 29410.5 |
| mri_segmenter4     |             0.793985   |        **0.92322**  |         **12.7224**|  **2.00333**| 30070.5 |
| staple             |           **0.797045** |          0.922631   |           13.0177  |     2.0236  | 29451.9 |




## AMOS 22 (Pancreas)
| model_name        |   mean_volumetric_dice |   mean_surface_dice |   mean_hausdorff95 |   mean_masd |      rmse |
|:------------------|-----------------------:|--------------------:|-------------------:|------------:|----------:|
| mr_segmentator    |              0.782952  |           0.917004  |          16.4871   |    1.99039  | 23661.1   |
| mri_segmenter     |            **0.876259**|         **0.973724**|        **12.5276** |  **1.23797**|**7186.54**|
| pansegnet_t1      |              0.816202  |           0.946902  |          23.396    |    2.6684   |  9235.66  |
| pansegnet_t2      |              0.0263096 |           0.0457698 |         558.959    |  447.907    | 73039.9   |
| total_segmentator |              0.759583  |           0.888407  |          16.8997   |    2.21504  | 25201.8   |
|:------------------|-----------------------:|--------------------:|-------------------:|------------:|----------:|
| voxtell           |              0.851591  |           0.963806  |           4.98238  |    0.942215 |  9096.59  |

### STAPLE 
| model_name        |   mean_volumetric_dice |   mean_surface_dice |   mean_hausdorff95 |   mean_masd |      rmse |
|:------------------|-----------------------:|--------------------:|-------------------:|------------:|----------:|
| staple_2          |              0.77709   |           0.915644  |          16.5928   |    2.01525  | 25892.5   |
| staple_3          |              0.870844  |           0.977078  |           4.844    |    0.764282 | 10941.9   |
| staple_3_1        |              0.815165  |           0.936473  |          10.1575   |    1.38592  | 19914.9   |
| staple_4          |            **0.875184**|         **0.977499**|         **4.77696**|  **0.745366**|**9217.79**|
* _2 - mri_segmenter, mr_segmentator
* _3 - mri_segmenter, mr_segmentator, pansegnet_t1
* _3_1 - mri_segmenter, mr_segmentator, total_segmentator
* _4 - mri_segmenter, mr_segmentator, pansegnet_t1, total_segmentator

## PanSegData (T2)
| model_name        |   mean_volumetric_dice |   mean_surface_dice |   mean_hausdorff95 |   mean_masd |    rmse |
|:------------------|-----------------------:|--------------------:|-------------------:|------------:|--------:|
| mr_segmentator    |          **0.42257**   |       **0.582964**  |         **224.999**|    **139.861**  | **64171.7** |
| mri_segmenter     |            0.00174548  |         0.0284116   |          1101.09   |    986.631  | 84616   |
| pansegnet_t1      |            0.000773926 |         0.022873    |          1107.24   |    999.331  | 86943.7 |
| total_segmentator |            0.235993    |         0.393382    |           354.533  |    251.409  | 76186.4 |
| umamba_nnunet     |            0.00109078  |         0.00880882  |          1545.26   |   1507.23   | 87842.8 |
| umamba_segresnet  |            0           |         2.32252e-05 |          1711.8    |   1711.15   | 89683.9 |
| umamba_swinunetr  |            0.000402111 |         0.0097217   |          1083.79   |    973.307  | 87861.5 |
|:------------------|-----------------------:|--------------------:|-------------------:|------------:|--------:|
| voxtell           |            0.750038    |         0.941298    |            15.2476 |      1.8899 | 14156.4 |
