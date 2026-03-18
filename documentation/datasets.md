AMOS: A Large-Scale Abdominal Multi-Organ
Benchmark for Versatile Medical Image
Segmentation (2022)
[paper](https://arxiv.org/pdf/2206.08023)
[dataset](https://zenodo.org/records/7262581)

AMOS provides 500
CT and 100 MRI scans collected from multi-center, multi-vendor, multi-modality,
multi-phase, multi-disease patients, each with voxel-level annotations of 15 abdominal organs, providing challenging examples and test-bed for studying robust
segmentation algorithms under diverse targets and scenarios. We further benchmark several state-of-the-art medical segmentation models to evaluate the status of
the existing methods on this new challenging dataset. We have made our datasets,
benchmark servers, and baselines publicly available, and hope to inspire future research. 

AMOS acquires data from the real-world
clinical settings, where the patients with different abdominal cancers/abnormalities are tested from
eight different CT or MRI scanners at two medical centers.

Annotation workflow of AMOS. The coarse annotations automatically labeled by pretrained segmentors will be further refined by human annotators for multiple times, including 5 junior
radiologists for the initial stage and 3 senior specialists for the second checking stage. 

AMOS is designed to facilitate abdominal multi-organ segmentation in a more
diverse, clinical, and complex scenarios. To meet this purpose, we cautiously select data, generated
by two institutes from 2018 to 2021, based on the following criteria: 1) Patients should be diagnosed
with abdominal tumors/abnormalities, while the ones with normal abdomen will be excluded. 2)
The imaging quality of the scanned data should be high-quality enough for the radiologists to review
and annotate. 3) To ensure the data diversity, the collected samples should be derived from different
scanners, as well as different scanning stages. 4) The scan data should cover as many of the specified
abdominal organs as possible.

covering 15 organ categories, including spleen, right kidney, left kidney,
gallbladder, esophagus, liver, stomach, aorta, inferior vena cava, pancreas, right adrenal gland, left
adrenal gland, duodenum, bladder, prostate/uterus.

MRI scanners: Ingenia -/-/29, Prisma 33/20/11, Signa HDe 7/-/-

Results in Table 5 show that the jointly trained model consistently improves
the individually trained model on AMOS-CT (i.e.,+0.55% mDice and +0.81% mNSD) and AMOS-MRI
(i.e., +2.14% mDice and +2.16% mNSD). To mitigate the effect that the improvement is caused by
more training data, we conduct two comparative experiments by selecting training samples randomly
(i.e., 160CT+40MRI, 10CT+30MRI). The improvement could also be observed, which validates the
effectiveness of cross-modalities training.
. For the 100 MRI scans, the number of males and females are 55
and 45, and the patients’ minimum, maximum, median, and mean ages are 22, 85, 50, and 48.71
years old, respectively. 
Z-score data normalization

In addition to providing the labeled 600 CT and MRI scans, we expect to provide 2000 CT and 1200 MRI scans without labels to support more learning tasks (semi-supervised, un-supervised, domain adaption, ...). The link can be found in:

also unlabeled data available 1200 MRI

### Structure

```bash
amos
│   dataset.json
└───imagesTr
│   │   amos_xxxx.nii.gz
│   │   ...
└───imagesVa
└───labelsTr
└───labelsVa

```

labelsTs are unavailable on Zenodo. 

