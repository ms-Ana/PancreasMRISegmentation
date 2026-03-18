
#!/usr/bin/env bash

# pdm run custom_nnUNetv2_predict_from_modelfolder -i /home/derzhana/PancreasMRISegmentation/tests/test -o /home/derzhana/PancreasMRISegmentation/tests/u-mamba/SegResNet_gpu -m /home/derzhana/models/UMamba/nnUNetTrainerSegResNet_2xFeat__nnUNetPlans__3d_fullres --folds all --disable_tta  --save_probabilities
# pdm run custom_nnUNetv2_predict_from_modelfolder -i /home/derzhana/PancreasMRISegmentation/tests/test1 -o /home/derzhana/PancreasMRISegmentation/tests/u-mamba/SegResNet_gpu -m /home/derzhana/models/UMamba/nnUNetTrainerSegResNet_2xFeat__nnUNetPlans__3d_fullres -f all --disable_tta  --save_probabilities


# pdm run custom_nnUNetv2_predict_from_modelfolder -i /home/derzhana/PancreasMRISegmentation/tests/test -o /home/derzhana/PancreasMRISegmentation/tests/u-mamba/UMambaBot_gpu -m /home/derzhana/models/UMamba/nnUNetTrainerUMambaBot__nnUNetPlans__3d_fullres -f all --disable_tta  --save_probabilities
# pdm run custom_nnUNetv2_predict_from_modelfolder -i /home/derzhana/PancreasMRISegmentation/tests/test1 -o /home/derzhana/PancreasMRISegmentation/tests/u-mamba/UMambaBot_gpu -m /home/derzhana/models/UMamba/nnUNetTrainerUMambaBot__nnUNetPlans__3d_fullres -f all --disable_tta  --save_probabilities

# pdm run custom_nnUNetv2_predict_from_modelfolder -i /home/derzhana/PancreasMRISegmentation/tests/test -o /home/derzhana/PancreasMRISegmentation/tests/u-mamba/UMambaEnc_gpu -m /home/derzhana/models/UMamba/nnUNetTrainerUMambaEnc__nnUNetPlans__3d_fullres -f all --disable_tta  --save_probabilities
# pdm run custom_nnUNetv2_predict_from_modelfolder -i /home/derzhana/PancreasMRISegmentation/tests/test1 -o /home/derzhana/PancreasMRISegmentation/tests/u-mamba/UMambaEnc_gpu -m /home/derzhana/models/UMamba/nnUNetTrainerUMambaEnc__nnUNetPlans__3d_fullres -f all --disable_tta  --save_probabilities
# pdm run custom_nnUNetv2_predict_from_modelfolder -i /home/derzhana/PancreasMRISegmentation/tests/test -o /home/derzhana/PancreasMRISegmentation/tests/u-mamba/UNet_gpu -m /home/derzhana/models/UMamba/nnUNetTrainer__nnUNetPlans__3d_fullres -f all --disable_tta  --save_probabilities
# pdm run custom_nnUNetv2_predict_from_modelfolder -i /home/derzhana/PancreasMRISegmentation/tests/test1 -o /home/derzhana/PancreasMRISegmentation/tests/u-mamba/UNet_gpu -m /home/derzhana/models/UMamba/nnUNetTrainer__nnUNetPlans__3d_fullres -f all --disable_tta  --save_probabilities

# pdm run custom_nnUNetv2_predict_from_modelfolder -i /home/derzhana/PancreasMRISegmentation/tests/test -o /home/derzhana/PancreasMRISegmentation/tests/u-mamba/UNETR_gpu -m /home/derzhana/models/UMamba/nnUNetTrainerUNETR__nnUNetPlans__3d_fullres -f all --disable_tta  --save_probabilities
# pdm run custom_nnUNetv2_predict_from_modelfolder -i /home/derzhana/PancreasMRISegmentation/tests/test1 -o /home/derzhana/PancreasMRISegmentation/tests/u-mamba/UNETR_gpu -m /home/derzhana/models/UMamba/nnUNetTrainerUNETR__nnUNetPlans__3d_fullres -f all --disable_tta  --save_probabilities


time pdm run custom_nnUNetv2_predict_from_modelfolder -i /home/derzhana/MEN1/external_data/Panther/imagesTr -o /home/derzhana/MEN1/PancreasMRISegmentation/results/panther/umamba/SegResNet -m /home/derzhana/MEN1/models/UMamba/nnUNetTrainerSegResNet_2xFeat__nnUNetPlans__3d_fullres -f all --disable_tta  
time pdm run custom_nnUNetv2_predict_from_modelfolder -i /home/derzhana/MEN1/external_data/Panther/imagesTr -o /home/derzhana/MEN1/PancreasMRISegmentation/results/panther/umamba/SwinUNETR -m /home/derzhana/MEN1/models/UMamba/nnUNetTrainerSwinUNETR__nnUNetPlans__3d_fullres -f all --disable_tta  
time pdm run custom_nnUNetv2_predict_from_modelfolder -i /home/derzhana/MEN1/external_data/Panther/imagesTr -o /home/derzhana/MEN1/PancreasMRISegmentation/results/panther/umamba/UNETR -m /home/derzhana/MEN1/models/UMamba/nnUNetTrainerUNETR__nnUNetPlans__3d_fullres -f all --disable_tta  
time pdm run custom_nnUNetv2_predict_from_modelfolder -i /home/derzhana/MEN1/external_data/Panther/imagesTr -o /home/derzhana/MEN1/PancreasMRISegmentation/results/panther/umamba/UNet -m /home/derzhana/MEN1/models/UMamba/nnUNetTrainer__nnUNetPlans__3d_fullres -f all --disable_tta  
