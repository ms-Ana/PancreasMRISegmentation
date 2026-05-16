
WEIGHTS_SETTINGS = {
    # umamba weights
    "umamba_segresnet": {
        "link": "1MlMZ96e5BpZ9Co5KnX71zMNk3ejNAzfA",
        "download_func": "gdown",
    },
    "umamba_nnunet": {
        "link": "1NKwuFLe6EWfiClM32MctVwWnu7FvD9N1",
        "download_func": "gdown",
    },
    "umamba_enc": {
        "link": "1ptCcenH5mYK-kh_WnpovR1959JcfzYj8",
        "download_func": "gdown",
    },
    "umamba_bot": {
        "link": "1smPeaDOw0GHZ9SvI_L63-o-iSRE_Yg_e",
        "download_func": "gdown",
    },
    "umamba_unetr": {
        "link": "1lEg8WTBduIBNT9bI7tJ_yEdSsEdeLy41",
        "download_func": "gdown",
    },
    "umamba_swinunetr": {
        "link": "1zkut-0Te3hn3P5hBYAll-KziQ2wMJ6gv",
        "download_func": "gdown",
    },
    # total segmentator weights
    "total_segmentator": {
        "link": "https://github.com/wasserth/TotalSegmentator/releases/download/v2.5.0-weights/Dataset850_TotalSegMRI_part1_organs_1088subj.zip",
        "download_func": "requests",
    },
    # pansegnet weights
    "pansegnet_t1": {
        "link": "1F4UAZW3neRbXmB_T2ZA2jpCkKUM6eLmG",
        "download_func": "gdown",
    },
    "pansegnet_t2": {
        "link": "1yAxnpjAeoyPiQd-_VNfKrjvB6ZZOY5XK",
        "download_func": "gdown",
    },
    # mri segmenter weights
    "mri_segmenter": {
        "link": "https://nihcc.app.box.com/index.php?rm=box_download_shared_file&shared_name=q6vl3015hteoufz7jll63u3hdqk79li7&file_id=f_1544045874167",
        "download_func": "requests",
    },
    # mr segmentator weights
    "mr_segmentator": {
        "link": "https://github.com/hhaentze/MRSegmentator/releases/download/v1.2.0/weights.zip",
        "download_func": "requests",
    },
    "voxtell": {
        "link": "mrokuss/VoxTell",
        "kwargs": {"model_name": "voxtell_v1.1"},
        "download_func": "snapshot",
    },
}

# default settings with pancreas labels
MODEL_SETTINGS = {
    "umamba_nnunet":
    {"predictor": "nnUNetv2",
     "label": 4},
  "umamba_swinunetr":
    {"predictor": "umamba",
    "label": 4},
  "umamba_segresnet":
    {
    "predictor": "umamba",
    "label": 4 },
  
  "umamba_bot":
    {"predictor": "umamba",
    "label": 4},
  
  "umamba_enc":
    {"predictor": "umamba",
    "label": 4},

  "total_segmentator":
    {"predictor": "nnUNetv2",
    "label": 7},

  "mri_segmenter":
    {"predictor": "nnUNetv2",
    "label": 11},

  "mr_segmentator":
    {"predictor": "nnUNetv2", 
    "label": 7}, 

  "pansegnet_t1":
    {"predictor": "nnUNetv1",
    "label": 1},

  "pansegnet_t2":
   {"predictor": "nnUNetv1",
    "label": 1},

  "voxtell":
  {
    "predictor": "voxtell",
    "label": 1
  }
}

#dataset_settings
DATASET_SETTINGS = {
    "panther": {
        "label": None, 
    }, 
    "amos22": {
        "label": 10,
    },
    "mrisegmenter": {
        "label": 11
    }, 
    "pansegnet_t2": {
        "label": 1
    }
}