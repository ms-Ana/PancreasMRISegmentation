## U-Mamba

The source code: https://github.com/bowang-lab/U-Mamba 

pdm run umamba_nnUNetv2_predict_from_modelfolder  -i /home/anastasiia/MasterThesis/external_data/Imaging/MRISegmenter/ImageTs -o /home/anastasiia/MasterThesis/PancreasMRISegmentation/results/umamba_segresnet/mrisegmenter -m /home/anastasiia/MasterThesis/PancreasMRISegmentation/src/pancreasmrisegmentation/.model_weights/umamba_segresnet -f all --disable_tta -c