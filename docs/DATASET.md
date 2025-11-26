# Data Preparation

:link: For all the datasets we used in our experiments, you can access them from the following public link:

- [msmt17](https://github.com/DanceTrack/DanceTrack)
- [DukeMTMC-re](https://github.com/MCG-NJU/SportsMOT)
- [market1501](https://george-zhuang.github.io/nettrack/)
- [CUHK03](https://www.crowdhuman.org/)

## Generate GT files

For the BFT and CrowdHuman datasets, you’ll need to use the provided script to convert their ground truth files to the format we require:

- For BFT: [gen_bft_gts.py](../tools/gen_bft_gts.py)
- For CrowdHuman: [gen_crowdhuman_gts.py](../tools/gen_crowdhuman_gts.py)

:pushpin: You need to modify the paths in the script according to your requirements.

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
