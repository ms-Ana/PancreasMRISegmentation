## Origin
code: 
paper: 

# 

# Disclaimer 
The original repo requirements 'torch>=2.0.0', 'nnunetv2>=2.2.1'. 
This repos environment is torch>=2.10.0, nnunetv2>=2.6.2
Comparing with original repo with environment: "torch==2.2.0", "nnunetv2==2.3.1"
There 2 versions of weights are available: checkpoint from 20240812 and 20240216

Comparing on cpu:
20240216:
average_percent_change: 1.530662225705329e-06 (mostly between labels and background)
average_max_diff: 0.004151545464992523
average_mean_diff: 6.593202095395156e-10

20240812: 
average_percent_change: 4.4528355656882305e-06
average_max_diff: 0.0020871541928499937
average_mean_diff: 2.8477991076414355e-09

Comparison between checkpoints:
average_percent_change: 15.250664760935503 (labels and background, spleen and liver)


Comparing on gpu:
20240216:
average_percent_change: 0.0017563350501900539
average_max_diff: 0.010215417481958866
average_mean_diff: 5.621444074677129e-07

20240812:
average_percent_change: 0.0032616079960227935
average_max_diff: 0.006359149236232042
average_mean_diff: 1.6665994735376444e-06

Comparison between checkpoints:
average_percent_change: 13.772960589475707 (labels and background, spleen and liver)