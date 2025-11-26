# Data Preparation

:link: For all the datasets we used in our experiments, you can access them from the following public link:

- msmt17 [paper](https://arxiv.org/abs/1711.08565), [Link](https://www.pkuvmc.com/dataset.html)
- DukeMTMC-reID [paper](https://arxiv.org/abs/1609.01775), [Link](https://www.kaggle.com/datasets/whurobin/dukemtmcreid)
- Market1501 [paper](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.html), [Link](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view?resourcekey=0-8nyl7K9_x37HlQm34MmrYQ)
- CUHK03 [paper](https://arxiv.org/abs/2101.10774), [Link](https://www.kaggle.com/datasets/priyanagda/cuhk03)

## File Tree

```text
<DATADIR>/
  ├── DanceTrack/
  │ ├── train/
  │ ├── val/
  │ ├── test/
  │ ├── train_seqmap.txt
  │ ├── val_seqmap.txt
  │ └── test_seqmap.txt
  ├── SportsMOT/
  │ ├── train/
  │ ├── val/
  │ ├── test/
  │ ├── train_seqmap.txt
  │ ├── val_seqmap.txt
  │ └── test_seqmap.txt
  ├── BFT/
  │ ├── train/
  │ ├── val/
  │ ├── test/
  │ ├── annotations_mot/    # used for generate gts for BFT
  │ ├── train_seqmap.txt
  │ ├── val_seqmap.txt
  │ └── test_seqmap.txt
  └── CrowdHuman/
    ├── images/
    │ ├── train/     # unzip from CrowdHuman
    │ └── val/       # unzip from CrowdHuman
    └── gts/
      ├── train/     # generate by ./data/gen_crowdhuman_gts.py
      └── val/       # generate by ./data/gen_crowdhuman_gts.py
```

## Q & A

- Q: Lack the `val_seqmap.txt` file of SportsMOT? </br>
  A: Refer to [The 'val_seqmap.txt' file of SportsMOT dataset · Issue #13](https://github.com/MCG-NJU/MOTIP/issues/13)
