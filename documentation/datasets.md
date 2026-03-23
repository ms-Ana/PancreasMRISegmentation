# AMOS: A Large-Scale Abdominal Multi-Organ Benchmark for Versatile Medical Image Segmentation (2022)

[paper](https://arxiv.org/pdf/2206.08023) \
[dataset](https://zenodo.org/records/7262581)

AMOS provides 500 CT and 100 MRI scans collected from multi-center, multi-vendor, multi-modality,
multi-phase, and multi-disease cohorts. Each scan includes voxel-level annotations for 15 abdominal
organs, making AMOS a challenging test bed for robust segmentation algorithms across diverse clinical
scenarios. The original work benchmarks several state-of-the-art medical segmentation models and
publishes datasets, benchmark servers, and baselines to support further research.

AMOS data come from real-world clinical settings, where patients with different abdominal
cancers and abnormalities were scanned on eight different CT or MRI scanners at two medical centers.

Annotation workflow in AMOS:
coarse annotations are first generated automatically by pretrained segmenters and then refined over
multiple rounds by human annotators, including 5 junior radiologists in the initial stage and 3 senior
specialists in the second review stage.

AMOS is designed to facilitate abdominal multi-organ segmentation in diverse, clinical, and complex
scenarios. Data from two institutes (2018-2021) were selected using the following criteria:

1. Patients were diagnosed with abdominal tumors or abnormalities; normal abdomens were excluded.
2. Scan quality was sufficient for expert review and annotation.
3. Samples were collected from different scanners and scanning phases to ensure diversity.
4. Scans covered as many target abdominal organs as possible.

The dataset covers 15 organ categories: spleen, right kidney, left kidney, gallbladder, esophagus,
liver, stomach, aorta, inferior vena cava, pancreas, right adrenal gland, left adrenal gland, duodenum,
bladder, and prostate/uterus.

MRI scanners: Ingenia -/-/29, Prisma 33/20/11, Signa HDe 7/-/-

Results in Table 5 show that the jointly trained model consistently outperforms the individually
trained model on both AMOS-CT (+0.55% mDice and +0.81% mNSD) and AMOS-MRI
(+2.14% mDice and +2.16% mNSD). To reduce the chance that this improvement comes only from
having more training data, two comparative experiments were performed with randomly selected
training subsets (160CT+40MRI and 10CT+30MRI). Improvements were still observed, supporting the
effectiveness of cross-modality training.

For the 100 MRI scans, there are 55 male and 45 female patients. The minimum, maximum, median,
and mean ages are 22, 85, 50, and 48.71 years, respectively.

Normalization: z-score normalization.

In addition to the labeled 600 CT and MRI scans, AMOS also provides unlabeled data to support
additional learning paradigms (semi-supervised learning, unsupervised learning, domain adaptation,
etc.): 2000 CT scans and 1200 MRI scans.

Unlabeled MRI available: 1200 scans.

labelsTs are unavailable on Zenodo.

# PANTHER Challenge

[dataset](https://zenodo.org/records/15192302)

The public training release contains 92 T1-weighted, contrast-enhanced arterial-phase scans and 367
additional scans across multiple MRI sequences.

# PanSegData

[paper](https://www.sciencedirect.com/science/article/pii/S1361841524003074) \
[dataset](https://osf.io/kysnj/overview) \
[code](https://github.com/NUBagciLab/PaNSegNet)

PanSegData includes 767 abdominal MRI scans: 385 contrast-enhanced venous T1w and 382 T2w.

# MRISegmenter

[paper](https://pubs.rsna.org/doi/full/10.1148/radiol.241979) \
[dataset](https://nihcc.app.box.com/s/zbrocs18g9ctrl0gb4n3hq7dlotpuwg5) \
[code](https://github.com/rsummers11/MRISegmenter)

MRISegmenter includes annotations for 62 abdominal organs and structures, including the pancreas.

It contains 780 cases with four scans per patient: pre-contrast and contrast-enhanced arterial, portal
venous, delayed phases, and T1w sequences.